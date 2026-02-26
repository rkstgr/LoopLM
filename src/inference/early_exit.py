"""src/inference/early_exit.py — Early exit strategies for LoopLM inference.

Three strategies are provided:

    StaticExit(step)            — always exit at a fixed recurrent step
    HiddenStateDiffExit(thresh) — exit when consecutive hidden states converge
    QExit(q)                    — exit when the CDF of exit probabilities >= q

All strategies share the same interface: implement `should_exit()` and `reset()`.
Use `run_with_early_exit()` to run a LoopLM model under any strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from src.model.looplm import LoopLM


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EarlyExitResult:
    logits: Tensor    # (B, S, vocab_size) from the exit step
    exit_step: int    # 1-indexed step at which the model stopped


# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------

class EarlyExitStrategy(ABC):
    """Predicate called after each recurrent step to decide whether to stop."""

    @abstractmethod
    def should_exit(
        self,
        step: int,
        *,
        hidden: Tensor,
        exit_lambda: Tensor,
        cdf: float,
    ) -> bool:
        """Return True to stop and use the current step's output.

        Args:
            step:        1-indexed current recurrent step
            hidden:      final-layer hidden state (B, S, hidden_size)
            exit_lambda: gate output λ_t averaged across batch/positions (scalar float tensor)
            cdf:         cumulative exit probability up to and including this step
        """
        ...

    def reset(self) -> None:
        """Reset accumulated state. Called at the start of each new sequence."""


# ---------------------------------------------------------------------------
# Strategy 1: StaticExit
# ---------------------------------------------------------------------------

class StaticExit(EarlyExitStrategy):
    """Always exit at a fixed recurrent step regardless of model state."""

    def __init__(self, step: int) -> None:
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        self.step = step

    def should_exit(self, step: int, *, hidden: Tensor, exit_lambda: Tensor, cdf: float) -> bool:
        return step >= self.step


# ---------------------------------------------------------------------------
# Strategy 2: HiddenStateDiffExit
# ---------------------------------------------------------------------------

class HiddenStateDiffExit(EarlyExitStrategy):
    """Exit when the RMS difference between consecutive hidden states < threshold.

    The difference is normalised by the number of elements (RMS), so the
    threshold is scale-independent with respect to sequence and batch size.

    At step 1 there is no previous hidden state, so the model never exits
    early at step 1 under this strategy.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self._prev_hidden: Tensor | None = None

    def reset(self) -> None:
        self._prev_hidden = None

    def should_exit(self, step: int, *, hidden: Tensor, exit_lambda: Tensor, cdf: float) -> bool:
        if self._prev_hidden is None:
            self._prev_hidden = hidden.detach().clone()
            return False

        diff = hidden.detach() - self._prev_hidden
        rms = (diff.pow(2).mean()).sqrt().item()
        self._prev_hidden = hidden.detach().clone()
        return rms < self.threshold


# ---------------------------------------------------------------------------
# Strategy 3: QExit
# ---------------------------------------------------------------------------

class QExit(EarlyExitStrategy):
    """Exit when the cumulative exit probability (CDF) reaches q_threshold.

    The CDF is computed from the exit gate's λ_t values averaged across
    all batch elements and token positions:

        p(t) = mean(λ_t) · survival_{t-1}
        CDF_t = Σ_{j=1}^{t} p(j)

    Special cases:
        q=0  → always exit after step 1 (CDF after step 1 > 0)
        q=1  → always run all steps (CDF reaches 1 only at the last step)
    """

    def __init__(self, q_threshold: float) -> None:
        if not 0.0 <= q_threshold <= 1.0:
            raise ValueError(f"q_threshold must be in [0, 1], got {q_threshold}")
        self.q_threshold = q_threshold
        self._cdf = 0.0
        self._survival = 1.0

    def reset(self) -> None:
        self._cdf = 0.0
        self._survival = 1.0

    def update_cdf(self, lam_mean: float, is_last_step: bool) -> float:
        """Advance the CDF by one step and return the new value."""
        if is_last_step:
            self._cdf += self._survival  # absorb remaining mass
        else:
            p_t = lam_mean * self._survival
            self._cdf += p_t
            self._survival *= 1.0 - lam_mean
        return self._cdf

    def should_exit(self, step: int, *, hidden: Tensor, exit_lambda: Tensor, cdf: float) -> bool:
        return cdf >= self.q_threshold


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

def run_with_early_exit(
    model: LoopLM,
    input_ids: Tensor,
    strategy: EarlyExitStrategy,
    max_steps: int | None = None,
) -> EarlyExitResult:
    """Run LoopLM recurrently, stopping when `strategy.should_exit()` returns True.

    Args:
        model:      trained LoopLM (should be in eval mode)
        input_ids:  (B, S) token ids
        strategy:   early-exit strategy instance
        max_steps:  maximum recurrent steps; defaults to model.config.max_recurrent_steps

    Returns:
        EarlyExitResult with final logits and the step at which we exited
    """
    strategy.reset()
    max_steps = max_steps or model.config.max_recurrent_steps

    B, S = input_ids.shape
    cos, sin = model.rope.get_cos_sin(S, input_ids.device)

    h = model.embed(input_ids)  # (B, S, hidden_size)

    logits: Tensor | None = None

    for step in range(1, max_steps + 1):
        # One recurrent pass through the shared layer stack
        for layer in model.layers:
            h = layer(h, cos, sin)

        logits = model.lm_head(model.final_norm(h))      # (B, S, vocab_size)
        lam = torch.sigmoid(model.exit_gate(h)).squeeze(-1)  # (B, S)

        # CDF update (scalar — mean over batch and positions)
        lam_mean = lam.mean().item()
        is_last = step == max_steps
        if isinstance(strategy, QExit):
            cdf = strategy.update_cdf(lam_mean, is_last_step=is_last)
        else:
            cdf = 0.0  # not used by other strategies

        if strategy.should_exit(step, hidden=h, exit_lambda=lam, cdf=cdf):
            return EarlyExitResult(logits=logits, exit_step=step)

    # Reached max_steps without an early exit
    assert logits is not None
    return EarlyExitResult(logits=logits, exit_step=max_steps)
