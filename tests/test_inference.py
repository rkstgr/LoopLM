"""tests/test_inference.py — Unit tests for early exit strategies (Sprint 4.1)."""

from __future__ import annotations

import torch
import pytest

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.inference.early_exit import (
    EarlyExitResult,
    StaticExit,
    HiddenStateDiffExit,
    QExit,
    run_with_early_exit,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model() -> LoopLM:
    """A minimal LoopLM (2 layers, hidden 64, 4 recurrent steps) for fast tests."""
    cfg = LoopLMConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=128,
        max_seq_len=16,
        max_recurrent_steps=4,
    )
    model = LoopLM(cfg)
    model.eval()
    return model


@pytest.fixture
def input_ids() -> torch.Tensor:
    return torch.randint(0, 256, (1, 8))  # batch=1, seq_len=8


# ---------------------------------------------------------------------------
# StaticExit
# ---------------------------------------------------------------------------

class TestStaticExit:
    def test_exits_at_configured_step(self, tiny_model, input_ids):
        for exit_at in range(1, 5):
            result = run_with_early_exit(tiny_model, input_ids, StaticExit(exit_at))
            assert result.exit_step == exit_at

    def test_clamps_to_max_steps(self, tiny_model, input_ids):
        # step > max_recurrent_steps → exits at max_recurrent_steps
        result = run_with_early_exit(tiny_model, input_ids, StaticExit(100))
        assert result.exit_step == tiny_model.config.max_recurrent_steps

    def test_logits_shape(self, tiny_model, input_ids):
        result = run_with_early_exit(tiny_model, input_ids, StaticExit(2))
        B, S = input_ids.shape
        assert result.logits.shape == (B, S, tiny_model.config.vocab_size)

    def test_invalid_step_raises(self):
        with pytest.raises(ValueError):
            StaticExit(0)

    def test_reset_is_idempotent(self):
        s = StaticExit(2)
        s.reset()  # should not raise
        s.reset()


# ---------------------------------------------------------------------------
# HiddenStateDiffExit
# ---------------------------------------------------------------------------

class TestHiddenStateDiffExit:
    def test_never_exits_at_step_1(self, tiny_model, input_ids):
        # With any threshold, step 1 has no previous state → never exits early
        strategy = HiddenStateDiffExit(threshold=1e9)
        result = run_with_early_exit(tiny_model, input_ids, strategy)
        assert result.exit_step >= 2

    def test_exits_at_step_2_with_huge_threshold(self, tiny_model, input_ids):
        # threshold so large any diff qualifies → exits at the first comparison (step 2)
        strategy = HiddenStateDiffExit(threshold=1e9)
        result = run_with_early_exit(tiny_model, input_ids, strategy)
        assert result.exit_step == 2

    def test_runs_all_steps_with_zero_threshold(self, tiny_model, input_ids):
        # threshold=0 → diff is always > 0 → never triggers → runs all steps
        strategy = HiddenStateDiffExit(threshold=0.0)
        result = run_with_early_exit(tiny_model, input_ids, strategy)
        assert result.exit_step == tiny_model.config.max_recurrent_steps

    def test_logits_shape(self, tiny_model, input_ids):
        strategy = HiddenStateDiffExit(threshold=1e9)
        result = run_with_early_exit(tiny_model, input_ids, strategy)
        B, S = input_ids.shape
        assert result.logits.shape == (B, S, tiny_model.config.vocab_size)

    def test_reset_clears_previous_hidden(self, tiny_model, input_ids):
        strategy = HiddenStateDiffExit(threshold=1e9)
        # First run accumulates state
        run_with_early_exit(tiny_model, input_ids, strategy)
        assert strategy._prev_hidden is not None

        # reset() clears it
        strategy.reset()
        assert strategy._prev_hidden is None


# ---------------------------------------------------------------------------
# QExit
# ---------------------------------------------------------------------------

class TestQExit:
    def test_q0_exits_at_step_1(self, tiny_model, input_ids):
        """Q-exit with q=0 should always exit at step 1."""
        result = run_with_early_exit(tiny_model, input_ids, QExit(0.0))
        assert result.exit_step == 1

    def test_q1_uses_all_steps(self, tiny_model, input_ids):
        """Q-exit with q=1 should use all recurrent steps."""
        result = run_with_early_exit(tiny_model, input_ids, QExit(1.0))
        assert result.exit_step == tiny_model.config.max_recurrent_steps

    def test_intermediate_q_exits_between_bounds(self, tiny_model, input_ids):
        result = run_with_early_exit(tiny_model, input_ids, QExit(0.5))
        max_T = tiny_model.config.max_recurrent_steps
        assert 1 <= result.exit_step <= max_T

    def test_logits_shape(self, tiny_model, input_ids):
        result = run_with_early_exit(tiny_model, input_ids, QExit(0.0))
        B, S = input_ids.shape
        assert result.logits.shape == (B, S, tiny_model.config.vocab_size)

    def test_reset_clears_cdf_and_survival(self):
        s = QExit(0.5)
        s._cdf = 0.9
        s._survival = 0.1
        s.reset()
        assert s._cdf == 0.0
        assert s._survival == 1.0

    def test_invalid_q_raises(self):
        with pytest.raises(ValueError):
            QExit(-0.1)
        with pytest.raises(ValueError):
            QExit(1.1)

    def test_increasing_q_gives_nondecreasing_exit_step(self, tiny_model, input_ids):
        """Higher q threshold should not produce an earlier exit."""
        steps = []
        for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = run_with_early_exit(tiny_model, input_ids, QExit(q))
            steps.append(result.exit_step)
        assert steps == sorted(steps), f"Expected non-decreasing, got {steps}"


# ---------------------------------------------------------------------------
# run_with_early_exit — general contract
# ---------------------------------------------------------------------------

class TestRunWithEarlyExit:
    def test_returns_early_exit_result(self, tiny_model, input_ids):
        result = run_with_early_exit(tiny_model, input_ids, StaticExit(2))
        assert isinstance(result, EarlyExitResult)

    def test_logits_are_finite(self, tiny_model, input_ids):
        result = run_with_early_exit(tiny_model, input_ids, StaticExit(1))
        assert torch.isfinite(result.logits).all()

    def test_max_steps_override(self, tiny_model, input_ids):
        # Force max_steps=2; StaticExit(10) should now exit at step 2
        result = run_with_early_exit(tiny_model, input_ids, StaticExit(10), max_steps=2)
        assert result.exit_step == 2

    def test_step1_and_step4_logits_differ(self, tiny_model, input_ids):
        r1 = run_with_early_exit(tiny_model, input_ids, StaticExit(1))
        r4 = run_with_early_exit(tiny_model, input_ids, StaticExit(4))
        assert not torch.allclose(r1.logits, r4.logits), (
            "Logits should differ between T=1 and T=4"
        )

    def test_no_gradient_in_eval_mode(self, tiny_model, input_ids):
        with torch.no_grad():
            result = run_with_early_exit(tiny_model, input_ids, StaticExit(2))
        assert result.logits is not None
