"""src/inference/kv_cache.py — KV cache management and efficient decode for LoopLM.

Following Section 5.4.2 & Table 14 of the Ouro paper (arXiv:2510.25741):

    "full"       — separate KV cache per recurrent step (exact, T×L memory)
    "last_step"  — all decode steps reuse the final recurrent step's KV cache
                   (T× memory reduction, near-lossless: GSM8K 78.92 -> 78.85)
    "first_step" — all decode steps reuse the first step's KV cache
                   (ablation baseline: collapses performance, GSM8K 78.92 -> 18.73)
    "averaged"   — all decode steps reuse the averaged KV cache across steps
                   (T× memory reduction, near-lossless: GSM8K 78.92 -> 78.73)

Public API::

    # Prefill prompt
    logits, step_caches = prefill(model, input_ids)

    # Decode single token (functional / legacy interface)
    last_cache = extract_decode_cache(step_caches, strategy="last_step")
    logits, last_cache = decode_one_token_last_step(model, token, last_cache, pos)

    # End-to-end autoregressive generation
    result = generate(model, input_ids, max_new_tokens=64, strategy="last_step")
    # result.sequences: (B, prompt_len + generated_len)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from src.model.looplm import LoopLM
from src.model.transformer import TransformerBlock, PostNormTransformerBlock
from src.model.rope import apply_rope


KVCacheStrategy = Literal["last_step", "full", "first_step", "averaged"]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LayerKVCache:
    """Accumulated K and V tensors for one transformer layer."""
    k: Tensor  # (B, num_heads, S, head_dim) — with RoPE already applied
    v: Tensor  # (B, num_heads, S, head_dim)

    def to(self, device: torch.device | str) -> LayerKVCache:
        return LayerKVCache(k=self.k.to(device), v=self.v.to(device))


# One StepCache holds per-layer KV for a single recurrent step.
StepCache = list[LayerKVCache]


@dataclass
class GenerateResult:
    """Output from autoregressive generation."""
    sequences: Tensor             # (B, total_len) full sequence containing prompt + generated tokens
    new_tokens: Tensor            # (B, num_gen) generated token IDs only
    exit_steps: list[int] | None = None  # recurrent step at which each token exited (if early exit used)


class KVCache:
    """Pre-allocated contiguous KV-Cache buffer for high-throughput decoding.

    Avoids O(S^2) memory reallocation and copy overhead caused by calling
    torch.cat at every decoding step.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        self.k = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype
        )
        self.v = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype
        )
        self.seq_len = 0

    @classmethod
    def from_tensors(cls, k: Tensor, v: Tensor, max_seq_len: int | None = None) -> KVCache:
        """Create and populate a KVCache buffer from existing K and V tensors."""
        B, H, S, D = k.shape
        capacity = max(max_seq_len or S, S + 64)
        cache = cls(
            batch_size=B,
            num_heads=H,
            head_dim=D,
            max_seq_len=capacity,
            device=k.device,
            dtype=k.dtype,
        )
        cache.k[:, :, :S, :] = k
        cache.v[:, :, :S, :] = v
        cache.seq_len = S
        return cache

    def _grow(self, new_capacity: int) -> None:
        """Double or expand capacity when max_seq_len is exceeded."""
        old_k, old_v = self.k, self.v
        self.k = torch.zeros(
            self.batch_size,
            self.num_heads,
            new_capacity,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.v = torch.zeros(
            self.batch_size,
            self.num_heads,
            new_capacity,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.k[:, :, :self.seq_len, :] = old_k[:, :, :self.seq_len, :]
        self.v[:, :, :self.seq_len, :] = old_v[:, :, :self.seq_len, :]
        self.max_seq_len = new_capacity

    def append(self, k_new: Tensor, v_new: Tensor) -> None:
        """Append a new token's key and value (B, H, 1, D)."""
        if self.seq_len >= self.max_seq_len:
            self._grow(self.max_seq_len * 2)
        S_new = k_new.shape[2]
        self.k[:, :, self.seq_len : self.seq_len + S_new, :].copy_(k_new)
        self.v[:, :, self.seq_len : self.seq_len + S_new, :].copy_(v_new)
        self.seq_len += S_new

    def get_view(self) -> tuple[Tensor, Tensor]:
        """Return zero-copy views of active keys and values."""
        return self.k[:, :, :self.seq_len, :], self.v[:, :, :self.seq_len, :]

    def to_layer_kv(self) -> LayerKVCache:
        """Export as immutable LayerKVCache."""
        k_view, v_view = self.get_view()
        return LayerKVCache(k=k_view.clone(), v=v_view.clone())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_layer_kv(
    block: TransformerBlock | PostNormTransformerBlock,
    h: Tensor,   # (B, S, hidden) — block input
    cos: Tensor,
    sin: Tensor,
) -> LayerKVCache:
    """Compute K,V for `block`'s attention sub-layer (applies pre-norm + RoPE)."""
    attn = block.attn
    x = block.pre_attn_norm(h) if hasattr(block, "pre_attn_norm") else h
    B, S, _ = x.shape
    k = attn.k_proj(x).view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = apply_rope(k, cos, sin)
    return LayerKVCache(k=k, v=v)


def _run_block_decode(
    block: TransformerBlock | PostNormTransformerBlock,
    h: Tensor,          # (B, 1, hidden) — single new token hidden state
    cos_new: Tensor,    # (1, head_dim/2) — RoPE for position `start_pos`
    sin_new: Tensor,
    past_kv: LayerKVCache,
) -> tuple[Tensor, LayerKVCache]:
    """Run `block` on a single new token with KV cache.

    Appends the new token's K,V to `past_kv` and attends over the full history.

    Returns:
        (h_out, updated_kv)  — h_out has shape (B, 1, hidden)
    """
    attn = block.attn
    x = block.pre_attn_norm(h) if hasattr(block, "pre_attn_norm") else h
    B, S_new, _ = x.shape  # S_new == 1

    q = attn.q_proj(x).view(B, S_new, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(B, S_new, attn.num_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(B, S_new, attn.num_heads, attn.head_dim).transpose(1, 2)

    q = apply_rope(q, cos_new, sin_new)
    k = apply_rope(k, cos_new, sin_new)

    # Extend cache with the new token
    k_full = torch.cat([past_kv.k, k], dim=2)  # (B, H, S_past+1, D)
    v_full = torch.cat([past_kv.v, v], dim=2)
    updated_kv = LayerKVCache(k=k_full, v=v_full)

    # Single-query attention over full history — is_causal=False is correct here
    # because the new query is already at the rightmost position.
    out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)
    out = out.transpose(1, 2).contiguous().view(B, S_new, -1)
    attn_out = attn.o_proj(out)

    # Normalization & residual connection
    if hasattr(block, "post_attn_norm"):
        # Sandwich norm (default LoopLM)
        h = h + block.post_attn_norm(attn_out)
        h = h + block.post_ffn_norm(block.ffn(block.pre_ffn_norm(h)))
    else:
        # Post-norm (HRM-style)
        h = block.attn_norm(h + attn_out)
        h = block.ffn_norm(h + block.ffn(h))

    return h, updated_kv


# ---------------------------------------------------------------------------
# Public API — Prefill and Cache Extraction
# ---------------------------------------------------------------------------

def prefill(
    model: LoopLM,
    input_ids: Tensor,
) -> tuple[Tensor, list[StepCache]]:
    """Run all T recurrent steps on the prompt, extracting per-step KV caches.

    As shown in Ouro paper Section 5.4.2, all recurrent steps require their
    own separate KV caches during prefill to prevent degradation.

    Args:
        model:     LoopLM (eval mode recommended)
        input_ids: (B, S) prompt token ids

    Returns:
        (logits, step_caches) where
          logits      — (B, S, vocab_size) from the last recurrent step
          step_caches — list of T StepCaches; step_caches[t][l] is the
                        LayerKVCache for recurrent step t, layer l
    """
    B, S = input_ids.shape
    cos, sin = model.rope.get_cos_sin(S, input_ids.device)

    h = model.embed(input_ids)  # (B, S, hidden)
    step_caches: list[StepCache] = []
    logits: Tensor | None = None

    for _ in range(model.config.max_recurrent_steps):
        step_cache: StepCache = []
        for block in model.layers:
            kv = _extract_layer_kv(block, h, cos, sin)
            step_cache.append(kv)
            h = block(h, cos, sin)
        logits = model.lm_head(model.final_norm(h))  # (B, S, V)
        step_caches.append(step_cache)

    assert logits is not None
    return logits, step_caches


def extract_decode_cache(
    step_caches: list[StepCache],
    strategy: KVCacheStrategy = "last_step",
) -> StepCache | list[StepCache]:
    """Extract initial decode cache from prefill output based on sharing strategy.

    Strategies:
        "last_step":  Returns step_caches[-1] (L layers, T× memory savings, recommended)
        "first_step": Returns step_caches[0] (L layers, ablation baseline)
        "averaged":   Returns per-layer averaged KV across all T steps (L layers)
        "full":       Returns the full list of step_caches (T × L layers)
    """
    if strategy == "last_step":
        return step_caches[-1]

    elif strategy == "first_step":
        return step_caches[0]

    elif strategy == "averaged":
        T = len(step_caches)
        L = len(step_caches[0])
        averaged_cache: StepCache = []
        for l in range(L):
            k_avg = torch.stack([step_caches[t][l].k for t in range(T)], dim=0).mean(dim=0)
            v_avg = torch.stack([step_caches[t][l].v for t in range(T)], dim=0).mean(dim=0)
            averaged_cache.append(LayerKVCache(k=k_avg, v=v_avg))
        return averaged_cache

    elif strategy == "full":
        return step_caches

    else:
        raise ValueError(
            f"Unknown KV cache strategy: '{strategy}'. "
            f"Expected one of: 'last_step', 'full', 'first_step', 'averaged'."
        )


# ---------------------------------------------------------------------------
# Public API — Single-Token Decode
# ---------------------------------------------------------------------------

def decode_one_token_full(
    model: LoopLM,
    new_token: Tensor,            # (B, 1) — next input token id
    step_caches: list[StepCache], # T × L — from prefill or previous decode
    start_pos: int,               # 0-indexed position of `new_token` in the sequence
) -> tuple[Tensor, list[StepCache]]:
    """Decode one token using a separate KV cache for each recurrent step.

    This is the exact baseline mode: step t uses only the K,V accumulated for step t.

    Returns:
        (logits, updated_step_caches) where logits is (B, 1, vocab_size)
    """
    device = new_token.device
    cos_full, sin_full = model.rope.get_cos_sin(start_pos + 1, device)
    cos_new = cos_full[start_pos : start_pos + 1]  # (1, head_dim/2)
    sin_new = sin_full[start_pos : start_pos + 1]

    h = model.embed(new_token)  # (B, 1, hidden)
    new_step_caches: list[StepCache] = []

    for step_cache in step_caches:
        new_layer_kvs: StepCache = []
        for block, past_kv in zip(model.layers, step_cache):
            h, new_kv = _run_block_decode(block, h, cos_new, sin_new, past_kv)
            new_layer_kvs.append(new_kv)
        new_step_caches.append(new_layer_kvs)

    logits = model.lm_head(model.final_norm(h))  # (B, 1, V)
    return logits, new_step_caches


def decode_one_token_last_step(
    model: LoopLM,
    new_token: Tensor,           # (B, 1)
    last_step_cache: StepCache,  # L — KV from the last recurrent step
    start_pos: int,
    early_exit_strategy: Any | None = None,
) -> tuple[Tensor, StepCache]:
    """Decode one token; all T recurrent steps reuse the last step's KV cache.

    This is the recommended Ouro mode that achieves ~T× memory reduction vs. full cache.
    All T steps use `last_step_cache` as their past context. Only the last step's
    newly computed K,V are stored back, so the cache stays at a single copy (L layers).

    For T=1 this is identical to decode_one_token_full.

    Returns:
        (logits, updated_last_step_cache) where logits is (B, 1, vocab_size)
    """
    device = new_token.device
    cos_full, sin_full = model.rope.get_cos_sin(start_pos + 1, device)
    cos_new = cos_full[start_pos : start_pos + 1]
    sin_new = sin_full[start_pos : start_pos + 1]

    h = model.embed(new_token)  # (B, 1, hidden)
    T = model.config.max_recurrent_steps
    updated_cache: StepCache | None = None

    for t in range(T):
        new_layer_kvs: StepCache = []
        for block, past_kv in zip(model.layers, last_step_cache):
            h, new_kv = _run_block_decode(block, h, cos_new, sin_new, past_kv)
            new_layer_kvs.append(new_kv)

        if early_exit_strategy is not None:
            lam = torch.sigmoid(model.exit_gate(h)).squeeze(-1)  # (B, 1)
            is_last = (t == T - 1)
            if hasattr(early_exit_strategy, "update_cdf"):
                cdf = early_exit_strategy.update_cdf(lam, is_last_step=is_last)
            else:
                cdf = 0.0
            if early_exit_strategy.should_exit(t + 1, hidden=h, exit_lambda=lam, cdf=cdf):
                updated_cache = new_layer_kvs
                break

        if t == T - 1:
            updated_cache = new_layer_kvs  # keep only last step

    assert updated_cache is not None
    logits = model.lm_head(model.final_norm(h))
    return logits, updated_cache


def decode_one_token_first_step(
    model: LoopLM,
    new_token: Tensor,            # (B, 1)
    first_step_cache: StepCache,  # L — KV from the first recurrent step
    start_pos: int,
) -> tuple[Tensor, StepCache]:
    """Decode one token reusing the first step's KV cache (ablation baseline).

    Paper Table 14: This strategy collapses performance (GSM8K: 78.92 -> 18.73)
    because early representations lack the depth needed for multi-step reasoning.

    Returns:
        (logits, updated_first_step_cache) where logits is (B, 1, vocab_size)
    """
    device = new_token.device
    cos_full, sin_full = model.rope.get_cos_sin(start_pos + 1, device)
    cos_new = cos_full[start_pos : start_pos + 1]
    sin_new = sin_full[start_pos : start_pos + 1]

    h = model.embed(new_token)  # (B, 1, hidden)
    T = model.config.max_recurrent_steps
    current_cache = first_step_cache

    for t in range(T):
        new_layer_kvs: StepCache = []
        for block, past_kv in zip(model.layers, current_cache):
            h, new_kv = _run_block_decode(block, h, cos_new, sin_new, past_kv)
            new_layer_kvs.append(new_kv)
        if t == 0:
            # First step stores back its newly computed KV
            current_cache = new_layer_kvs

    logits = model.lm_head(model.final_norm(h))
    return logits, current_cache


def decode_one_token_averaged(
    model: LoopLM,
    new_token: Tensor,          # (B, 1)
    averaged_cache: StepCache,  # L — averaged KV across recurrent steps
    start_pos: int,
) -> tuple[Tensor, StepCache]:
    """Decode one token reusing an averaged KV cache across all recurrent steps.

    Paper Table 14: Averaged reuse achieves near-lossless accuracy (GSM8K: 78.73)
    and 4× memory reduction.

    Returns:
        (logits, updated_averaged_cache) where logits is (B, 1, vocab_size)
    """
    device = new_token.device
    cos_full, sin_full = model.rope.get_cos_sin(start_pos + 1, device)
    cos_new = cos_full[start_pos : start_pos + 1]
    sin_new = sin_full[start_pos : start_pos + 1]

    h = model.embed(new_token)  # (B, 1, hidden)
    T = model.config.max_recurrent_steps
    L = len(model.layers)

    # Collect the new token's (k, v) from each recurrent step
    step_new_ks: list[list[Tensor]] = [[] for _ in range(L)]
    step_new_vs: list[list[Tensor]] = [[] for _ in range(L)]

    for t in range(T):
        for l, (block, past_kv) in enumerate(zip(model.layers, averaged_cache)):
            h, new_kv = _run_block_decode(block, h, cos_new, sin_new, past_kv)
            # Extract only the newly added token slice at the rightmost position
            step_new_ks[l].append(new_kv.k[:, :, -1:, :])
            step_new_vs[l].append(new_kv.v[:, :, -1:, :])

    # Average the new token representations across all T recurrent steps
    updated_cache: StepCache = []
    for l in range(L):
        avg_k_new = torch.stack(step_new_ks[l], dim=0).mean(dim=0)  # (B, H, 1, D)
        avg_v_new = torch.stack(step_new_vs[l], dim=0).mean(dim=0)
        past_kv = averaged_cache[l]
        updated_cache.append(
            LayerKVCache(
                k=torch.cat([past_kv.k, avg_k_new], dim=2),
                v=torch.cat([past_kv.v, avg_v_new], dim=2),
            )
        )

    logits = model.lm_head(model.final_norm(h))
    return logits, updated_cache


def decode_one_token(
    model: LoopLM,
    new_token: Tensor,
    cache: StepCache | list[StepCache],
    start_pos: int,
    strategy: KVCacheStrategy = "last_step",
    early_exit_strategy: Any | None = None,
) -> tuple[Tensor, StepCache | list[StepCache]]:
    """Unified single-token decode dispatcher based on `strategy`."""
    if strategy == "last_step":
        assert isinstance(cache, list) and isinstance(cache[0], LayerKVCache)
        return decode_one_token_last_step(
            model, new_token, cache, start_pos, early_exit_strategy=early_exit_strategy
        )
    elif strategy == "first_step":
        assert isinstance(cache, list) and isinstance(cache[0], LayerKVCache)
        return decode_one_token_first_step(model, new_token, cache, start_pos)
    elif strategy == "averaged":
        assert isinstance(cache, list) and isinstance(cache[0], LayerKVCache)
        return decode_one_token_averaged(model, new_token, cache, start_pos)
    elif strategy == "full":
        assert isinstance(cache, list) and isinstance(cache[0], list)
        return decode_one_token_full(model, new_token, cache, start_pos)
    else:
        raise ValueError(f"Unknown KV cache strategy: '{strategy}'")


# ---------------------------------------------------------------------------
# Sampling Helpers
# ---------------------------------------------------------------------------

def _filter_top_k_top_p(
    logits: Tensor,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or top-p (nucleus) filtering."""
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


def _sample_next_token(
    logits: Tensor,  # (B, vocab_size)
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Tensor:
    """Sample or argmax the next token given unnormalized logits."""
    if temperature <= 0.0 or (top_k is None and top_p is None and temperature == 0.0):
        # Greedy decoding
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits_scaled = logits / max(temperature, 1e-5)
    filtered = _filter_top_k_top_p(logits_scaled, top_k=top_k, top_p=top_p)
    probs = F.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Public API — End-to-End Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: LoopLM,
    input_ids: Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    strategy: KVCacheStrategy = "last_step",
    early_exit_strategy: Any | None = None,
    use_cache: bool = True,
) -> GenerateResult:
    """Autoregressive generation for LoopLM using KV cache sharing.

    Args:
        model:                LoopLM instance
        input_ids:            (B, S) prompt token IDs
        max_new_tokens:       maximum number of new tokens to generate
        temperature:          sampling temperature (0.0 = greedy)
        top_k:                top-k filtering parameter
        top_p:                top-p (nucleus) filtering parameter
        eos_token_id:         optional token ID to terminate generation
        strategy:             KV cache sharing strategy: "last_step" (default),
                              "full", "first_step", or "averaged"
        early_exit_strategy:  optional EarlyExitStrategy instance (e.g. QExit, StaticExit)
        use_cache:            if True, uses efficient KV-cache decoding;
                              if False, performs full unrolled re-computation per token

    Returns:
        GenerateResult with full sequence and newly generated tokens
    """
    model.eval()
    B, prompt_len = input_ids.shape
    device = input_ids.device

    if not use_cache:
        # Reference non-cached decoding (slow baseline for verification)
        curr_ids = input_ids.clone()
        generated_list: list[Tensor] = []
        for _ in range(max_new_tokens):
            out = model(curr_ids)
            next_logits = out.logits[-1][:, -1, :]  # (B, V)
            next_tok = _sample_next_token(next_logits, temperature, top_k, top_p)
            generated_list.append(next_tok)
            curr_ids = torch.cat([curr_ids, next_tok], dim=1)
            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break
        new_tokens = torch.cat(generated_list, dim=1) if generated_list else torch.empty((B, 0), dtype=torch.long, device=device)
        return GenerateResult(sequences=curr_ids, new_tokens=new_tokens)

    # 1. Prefill phase
    prefill_logits, step_caches = prefill(model, input_ids)
    next_logits = prefill_logits[:, -1, :]  # (B, vocab_size)

    # 2. Extract initial decode cache
    cache = extract_decode_cache(step_caches, strategy=strategy)

    generated_tokens: list[Tensor] = []
    exit_steps: list[int] | None = [] if early_exit_strategy is not None else None

    # Track sequence completion per batch element
    done = torch.zeros(B, dtype=torch.bool, device=device)

    for i in range(max_new_tokens):
        next_tok = _sample_next_token(next_logits, temperature, top_k, top_p)
        generated_tokens.append(next_tok)

        if eos_token_id is not None:
            done = done | (next_tok.squeeze(-1) == eos_token_id)
            if done.all():
                break

        if i == max_new_tokens - 1:
            break

        # Decode the single newly generated token
        curr_pos = prompt_len + i
        step_logits, cache = decode_one_token(
            model=model,
            new_token=next_tok,
            cache=cache,
            start_pos=curr_pos,
            strategy=strategy,
            early_exit_strategy=early_exit_strategy,
        )
        next_logits = step_logits.squeeze(1)  # (B, V)

    new_tokens_tensor = torch.cat(generated_tokens, dim=1) if generated_tokens else torch.empty((B, 0), dtype=torch.long, device=device)
    full_sequences = torch.cat([input_ids, new_tokens_tensor], dim=1)

    return GenerateResult(
        sequences=full_sequences,
        new_tokens=new_tokens_tensor,
        exit_steps=exit_steps,
    )
