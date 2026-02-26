"""src/inference/kv_cache.py — KV cache management for LoopLM generation.

Two modes are supported:

    "full"       — separate KV cache per recurrent step (exact, more memory)
    "last_step"  — all decode steps reuse the last recurrent step's KV cache
                   (~T× memory reduction; near-identical outputs for trained models)

Public API::

    logits, step_caches = prefill(model, input_ids)
    logits, step_caches = decode_one_token_full(model, token, step_caches, pos)

    last_cache = step_caches[-1]
    logits, last_cache = decode_one_token_last_step(model, token, last_cache, pos)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from src.model.looplm import LoopLM
from src.model.transformer import TransformerBlock
from src.model.rope import apply_rope


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LayerKVCache:
    """Accumulated K and V tensors for one transformer layer."""
    k: Tensor  # (B, num_heads, S, head_dim) — with RoPE already applied
    v: Tensor  # (B, num_heads, S, head_dim)


# One StepCache holds per-layer KV for a single recurrent step.
StepCache = list[LayerKVCache]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_layer_kv(
    block: TransformerBlock,
    h: Tensor,   # (B, S, hidden) — block input (pre-sandwich)
    cos: Tensor,
    sin: Tensor,
) -> LayerKVCache:
    """Compute K,V for `block`'s attention sub-layer (applies pre_attn_norm + RoPE)."""
    attn = block.attn
    x = block.pre_attn_norm(h)
    B, S, _ = x.shape
    k = attn.k_proj(x).view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = apply_rope(k, cos, sin)
    return LayerKVCache(k=k, v=v)


def _run_block_decode(
    block: TransformerBlock,
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
    x = block.pre_attn_norm(h)
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

    # Sandwich norm residuals (matches TransformerBlock.forward)
    h = h + block.post_attn_norm(attn_out)
    h = h + block.post_ffn_norm(block.ffn(block.pre_ffn_norm(h)))
    return h, updated_kv


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prefill(
    model: LoopLM,
    input_ids: Tensor,
) -> tuple[Tensor, list[StepCache]]:
    """Run all T recurrent steps on the prompt, extracting per-step KV caches.

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


def decode_one_token_full(
    model: LoopLM,
    new_token: Tensor,            # (B, 1) — next input token id
    step_caches: list[StepCache], # T × L — from prefill or previous decode
    start_pos: int,               # 0-indexed position of `new_token` in the sequence
) -> tuple[Tensor, list[StepCache]]:
    """Decode one token using a separate KV cache for each recurrent step.

    This is the exact mode: step t uses only the K,V accumulated for step t.

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
    new_token: Tensor,         # (B, 1)
    last_step_cache: StepCache,  # L — KV from the last recurrent step
    start_pos: int,
) -> tuple[Tensor, StepCache]:
    """Decode one token; all T recurrent steps reuse the last step's KV cache.

    This is the approximate mode that achieves ~T× memory reduction vs. full cache.
    All T steps use `last_step_cache` as their past context. Only the last step's
    newly computed K,V are stored back, so the cache stays at a single copy.

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
        if t == T - 1:
            updated_cache = new_layer_kvs  # keep only last step

    assert updated_cache is not None
    logits = model.lm_head(model.final_norm(h))
    return logits, updated_cache
