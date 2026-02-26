"""tests/test_kv_cache.py — Tests for KV cache management (Sprint 4.2)."""

from __future__ import annotations

import torch
import pytest

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.inference.kv_cache import (
    LayerKVCache,
    StepCache,
    prefill,
    decode_one_token_full,
    decode_one_token_last_step,
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
