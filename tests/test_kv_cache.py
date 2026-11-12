"""tests/test_kv_cache.py — Tests for KV cache management (Sprint 4.2)."""

from __future__ import annotations

import torch
import pytest

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.inference.kv_cache import (
    LayerKVCache,
    StepCache,
    KVCache,
    GenerateResult,
    prefill,
    extract_decode_cache,
    decode_one_token,
    decode_one_token_full,
    decode_one_token_last_step,
    decode_one_token_first_step,
    decode_one_token_averaged,
    generate,
)



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model(num_recurrent_steps: int = 4) -> LoopLM:
    cfg = LoopLMConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=128,
        max_seq_len=32,
        max_recurrent_steps=num_recurrent_steps,
    )
    model = LoopLM(cfg)
    model.eval()
    return model


@pytest.fixture
def model() -> LoopLM:
    return _make_model(num_recurrent_steps=4)


@pytest.fixture
def model_t1() -> LoopLM:
    return _make_model(num_recurrent_steps=1)


@pytest.fixture
def prompt() -> torch.Tensor:
    return torch.randint(0, 256, (1, 8))  # batch=1, seq_len=8


@pytest.fixture
def next_token() -> torch.Tensor:
    return torch.randint(0, 256, (1, 1))  # one new token


# ---------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------

class TestPrefill:
    def test_logits_shape(self, model, prompt):
        logits, _ = prefill(model, prompt)
        B, S = prompt.shape
        assert logits.shape == (B, S, model.config.vocab_size)

    def test_logits_finite(self, model, prompt):
        logits, _ = prefill(model, prompt)
        assert torch.isfinite(logits).all()

    def test_returns_T_step_caches(self, model, prompt):
        _, step_caches = prefill(model, prompt)
        assert len(step_caches) == model.config.max_recurrent_steps

    def test_each_step_cache_has_L_layers(self, model, prompt):
        _, step_caches = prefill(model, prompt)
        for sc in step_caches:
            assert len(sc) == model.config.num_layers

    def test_kv_shapes(self, model, prompt):
        B, S = prompt.shape
        _, step_caches = prefill(model, prompt)
        H = model.config.num_heads
        D = model.config.hidden_size // model.config.num_heads
        for sc in step_caches:
            for lkv in sc:
                assert lkv.k.shape == (B, H, S, D)
                assert lkv.v.shape == (B, H, S, D)


# ---------------------------------------------------------------------------
# decode_one_token_full
# ---------------------------------------------------------------------------

class TestDecodeOneFull:
    def test_logits_shape(self, model, prompt, next_token):
        _, step_caches = prefill(model, prompt)
        logits, _ = decode_one_token_full(model, next_token, step_caches, start_pos=8)
        assert logits.shape == (1, 1, model.config.vocab_size)

    def test_logits_finite(self, model, prompt, next_token):
        _, step_caches = prefill(model, prompt)
        logits, _ = decode_one_token_full(model, next_token, step_caches, start_pos=8)
        assert torch.isfinite(logits).all()

    def test_cache_grows_by_one(self, model, prompt, next_token):
        B, S = prompt.shape
        H = model.config.num_heads
        D = model.config.hidden_size // model.config.num_heads
        _, step_caches = prefill(model, prompt)
        _, new_caches = decode_one_token_full(model, next_token, step_caches, start_pos=S)
        # After decoding one token, each layer's cache should have S+1 positions
        for sc in new_caches:
            for lkv in sc:
                assert lkv.k.shape == (B, H, S + 1, D)

    def test_returns_T_step_caches(self, model, prompt, next_token):
        _, step_caches = prefill(model, prompt)
        _, new_caches = decode_one_token_full(model, next_token, step_caches, start_pos=8)
        assert len(new_caches) == model.config.max_recurrent_steps

    def test_two_decode_steps(self, model, prompt):
        _, step_caches = prefill(model, prompt)
        tok1 = torch.randint(0, 256, (1, 1))
        tok2 = torch.randint(0, 256, (1, 1))
        logits1, step_caches = decode_one_token_full(model, tok1, step_caches, start_pos=8)
        logits2, step_caches = decode_one_token_full(model, tok2, step_caches, start_pos=9)
        assert logits1.shape == (1, 1, model.config.vocab_size)
        assert logits2.shape == (1, 1, model.config.vocab_size)


# ---------------------------------------------------------------------------
# decode_one_token_last_step
# ---------------------------------------------------------------------------

class TestDecodeOneLastStep:
    def test_logits_shape(self, model, prompt, next_token):
        _, step_caches = prefill(model, prompt)
        last_cache = step_caches[-1]
        logits, _ = decode_one_token_last_step(model, next_token, last_cache, start_pos=8)
        assert logits.shape == (1, 1, model.config.vocab_size)

    def test_logits_finite(self, model, prompt, next_token):
        _, step_caches = prefill(model, prompt)
        last_cache = step_caches[-1]
        logits, _ = decode_one_token_last_step(model, next_token, last_cache, start_pos=8)
        assert torch.isfinite(logits).all()

    def test_cache_grows_by_one(self, model, prompt, next_token):
        B, S = prompt.shape
        H = model.config.num_heads
        D = model.config.hidden_size // model.config.num_heads
        _, step_caches = prefill(model, prompt)
        last_cache = step_caches[-1]
        _, new_cache = decode_one_token_last_step(model, next_token, last_cache, start_pos=S)
        for lkv in new_cache:
            assert lkv.k.shape == (B, H, S + 1, D)

    def test_returns_one_step_cache(self, model, prompt, next_token):
        _, step_caches = prefill(model, prompt)
        last_cache = step_caches[-1]
        _, new_cache = decode_one_token_last_step(model, next_token, last_cache, start_pos=8)
        # last-step mode returns a single StepCache (L layers), not T of them
        assert len(new_cache) == model.config.num_layers

    def test_two_decode_steps(self, model, prompt):
        _, step_caches = prefill(model, prompt)
        last_cache = step_caches[-1]
        tok1 = torch.randint(0, 256, (1, 1))
        tok2 = torch.randint(0, 256, (1, 1))
        logits1, last_cache = decode_one_token_last_step(model, tok1, last_cache, start_pos=8)
        logits2, last_cache = decode_one_token_last_step(model, tok2, last_cache, start_pos=9)
        assert logits1.shape == (1, 1, model.config.vocab_size)
        assert logits2.shape == (1, 1, model.config.vocab_size)


# ---------------------------------------------------------------------------
# T=1: last-step reuse is identical to full cache
# ---------------------------------------------------------------------------

class TestT1Equivalence:
    """With a single recurrent step, both cache modes must give identical outputs."""

    def test_prefill_logits_identical(self, model_t1, prompt):
        logits_kv, _ = prefill(model_t1, prompt)
        # Compare against regular forward pass
        with torch.no_grad():
            out = model_t1(prompt, num_steps=1)
        assert torch.allclose(logits_kv, out.logits[-1], atol=1e-5), (
            "prefill logits should match regular forward when T=1"
        )

    def test_decode_full_vs_last_step_identical(self, model_t1, prompt, next_token):
        S = prompt.shape[1]
        _, step_caches = prefill(model_t1, prompt)

        with torch.no_grad():
            logits_full, _ = decode_one_token_full(
                model_t1, next_token, step_caches, start_pos=S
            )

        # Re-run prefill to get a fresh cache for last-step mode
        _, step_caches2 = prefill(model_t1, prompt)
        last_cache = step_caches2[-1]
        with torch.no_grad():
            logits_last, _ = decode_one_token_last_step(
                model_t1, next_token, last_cache, start_pos=S
            )

        assert torch.allclose(logits_full, logits_last, atol=1e-5), (
            "Full cache and last-step cache must give identical logits when T=1"
        )


# ---------------------------------------------------------------------------
# Memory savings check
# ---------------------------------------------------------------------------

class TestMemorySavings:
    def test_last_step_cache_is_smaller_than_full(self, model, prompt):
        """Full cache stores T×L layer caches; last-step stores only L."""
        _, step_caches = prefill(model, prompt)
        T = model.config.max_recurrent_steps
        L = model.config.num_layers

        full_cache_layers = sum(len(sc) for sc in step_caches)
        last_step_cache_layers = len(step_caches[-1])

        assert full_cache_layers == T * L
        assert last_step_cache_layers == L
        assert full_cache_layers == T * last_step_cache_layers


# ---------------------------------------------------------------------------
# extract_decode_cache
# ---------------------------------------------------------------------------

class TestExtractDecodeCache:
    def test_extract_last_step(self, model, prompt):
        _, step_caches = prefill(model, prompt)
        cache = extract_decode_cache(step_caches, strategy="last_step")
        assert len(cache) == model.config.num_layers
        assert cache is step_caches[-1]

    def test_extract_first_step(self, model, prompt):
        _, step_caches = prefill(model, prompt)
        cache = extract_decode_cache(step_caches, strategy="first_step")
        assert len(cache) == model.config.num_layers
        assert cache is step_caches[0]

    def test_extract_averaged(self, model, prompt):
        _, step_caches = prefill(model, prompt)
        cache = extract_decode_cache(step_caches, strategy="averaged")
        assert len(cache) == model.config.num_layers
        # Check averaged shape matches
        for lkv in cache:
            assert lkv.k.shape == step_caches[0][0].k.shape
            assert lkv.v.shape == step_caches[0][0].v.shape

    def test_extract_full(self, model, prompt):
        _, step_caches = prefill(model, prompt)
        cache = extract_decode_cache(step_caches, strategy="full")
        assert len(cache) == model.config.max_recurrent_steps

    def test_extract_unknown_strategy_raises(self, model, prompt):
        _, step_caches = prefill(model, prompt)
        with pytest.raises(ValueError):
            extract_decode_cache(step_caches, strategy="invalid_strategy")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# decode_one_token_first_step
# ---------------------------------------------------------------------------

class TestDecodeOneFirstStep:
    def test_logits_shape(self, model, prompt, next_token):
        _, step_caches = prefill(model, prompt)
        first_cache = step_caches[0]
        logits, new_cache = decode_one_token_first_step(model, next_token, first_cache, start_pos=8)
        assert logits.shape == (1, 1, model.config.vocab_size)
        assert len(new_cache) == model.config.num_layers
        assert new_cache[0].k.shape[2] == 9

    def test_two_decode_steps(self, model, prompt):
        _, step_caches = prefill(model, prompt)
        cache = step_caches[0]
        tok1 = torch.randint(0, 256, (1, 1))
        tok2 = torch.randint(0, 256, (1, 1))
        logits1, cache = decode_one_token_first_step(model, tok1, cache, start_pos=8)
        logits2, cache = decode_one_token_first_step(model, tok2, cache, start_pos=9)
        assert logits1.shape == (1, 1, model.config.vocab_size)
        assert logits2.shape == (1, 1, model.config.vocab_size)


# ---------------------------------------------------------------------------
# decode_one_token_averaged
# ---------------------------------------------------------------------------

class TestDecodeOneAveraged:
    def test_logits_shape(self, model, prompt, next_token):
        _, step_caches = prefill(model, prompt)
        avg_cache = extract_decode_cache(step_caches, strategy="averaged")
        logits, new_cache = decode_one_token_averaged(model, next_token, avg_cache, start_pos=8)
        assert logits.shape == (1, 1, model.config.vocab_size)
        assert len(new_cache) == model.config.num_layers
        assert new_cache[0].k.shape[2] == 9

    def test_two_decode_steps(self, model, prompt):
        _, step_caches = prefill(model, prompt)
        avg_cache = extract_decode_cache(step_caches, strategy="averaged")
        tok1 = torch.randint(0, 256, (1, 1))
        tok2 = torch.randint(0, 256, (1, 1))
        logits1, avg_cache = decode_one_token_averaged(model, tok1, avg_cache, start_pos=8)
        logits2, avg_cache = decode_one_token_averaged(model, tok2, avg_cache, start_pos=9)
        assert logits1.shape == (1, 1, model.config.vocab_size)
        assert logits2.shape == (1, 1, model.config.vocab_size)


# ---------------------------------------------------------------------------
# Unified decode_one_token dispatcher
# ---------------------------------------------------------------------------

class TestUnifiedDecodeOneToken:
    def test_dispatch_all_strategies(self, model, prompt, next_token):
        _, step_caches = prefill(model, prompt)
        for strategy in ["last_step", "first_step", "averaged", "full"]:
            cache = extract_decode_cache(step_caches, strategy=strategy)  # type: ignore[arg-type]
            logits, updated_cache = decode_one_token(
                model, next_token, cache, start_pos=8, strategy=strategy  # type: ignore[arg-type]
            )
            assert logits.shape == (1, 1, model.config.vocab_size)
            assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# KVCache pre-allocated buffer
# ---------------------------------------------------------------------------

class TestKVCacheBuffer:
    def test_init_and_get_view(self):
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)
        cache = KVCache.from_tensors(k, v, max_seq_len=32)
        assert cache.seq_len == 8
        k_view, v_view = cache.get_view()
        assert torch.allclose(k_view, k)
        assert torch.allclose(v_view, v)

    def test_append_grows_capacity(self):
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        cache = KVCache.from_tensors(k, v, max_seq_len=6)
        k_new = torch.randn(1, 2, 1, 8)
        v_new = torch.randn(1, 2, 1, 8)
        cache.append(k_new, v_new)
        assert cache.seq_len == 5
        # Append beyond max_seq_len -> triggers _grow
        cache.append(k_new, v_new)
        cache.append(k_new, v_new)
        assert cache.seq_len == 7
        assert cache.max_seq_len >= 7


# ---------------------------------------------------------------------------
# generate end-to-end
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_generate_shape(self, model, prompt):
        B, S = prompt.shape
        max_new = 5
        res = generate(model, prompt, max_new_tokens=max_new, strategy="last_step")
        assert isinstance(res, GenerateResult)
        assert res.sequences.shape == (B, S + max_new)
        assert res.new_tokens.shape == (B, max_new)

    def test_generate_all_strategies(self, model, prompt):
        for strategy in ["last_step", "full", "averaged", "first_step"]:
            res = generate(model, prompt, max_new_tokens=3, strategy=strategy)  # type: ignore[arg-type]
            assert res.new_tokens.shape == (1, 3)

    def test_generate_eos_early_stop(self, model):
        prompt = torch.tensor([[10, 20]])
        # Find the argmax token to simulate immediate eos stop
        with torch.no_grad():
            out = model(prompt)
            target_eos = out.logits[-1][0, -1].argmax().item()

        res = generate(
            model,
            prompt,
            max_new_tokens=10,
            temperature=0.0,
            eos_token_id=target_eos,
            strategy="last_step",
        )
        # Should stop after generating the target_eos token (1 token)
        assert res.new_tokens.shape == (1, 1)
        assert res.new_tokens[0, 0].item() == target_eos

    def test_model_generate_method(self, model, prompt):
        res = model.generate(prompt, max_new_tokens=4, strategy="last_step")
        assert res.new_tokens.shape == (1, 4)

    def test_t1_cached_vs_non_cached_greedy_identical(self, model_t1, prompt):
        """When T=1, cached decode must match non-cached full forward pass exactly."""
        res_cached = generate(
            model_t1, prompt, max_new_tokens=4, temperature=0.0, use_cache=True, strategy="last_step"
        )
        res_uncached = generate(
            model_t1, prompt, max_new_tokens=4, temperature=0.0, use_cache=False
        )
        assert torch.equal(res_cached.sequences, res_uncached.sequences), (
            "Greedy sequences should match between cached and uncached forward pass for T=1"
        )

    def test_sampling_options(self, model, prompt):
        # Sampling with temperature, top_k, top_p
        res = generate(
            model,
            prompt,
            max_new_tokens=4,
            temperature=0.8,
            top_k=20,
            top_p=0.9,
            strategy="last_step",
        )
        assert res.new_tokens.shape == (1, 4)

