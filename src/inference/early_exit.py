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
    logits: Tensor    # (B, S, vocab_size) from the exit step (gathered per-token or expectation)
    exit_step: int    # 1-indexed step at which the model stopped (or uniform step / round(mean))
    exit_steps: Tensor | None = None  # (B, S) tensor of 1-indexed exit step per token
    exit_pdf: Tensor | None = None    # (B, S, T) per-token exit probability distribution
    exit_cdf: Tensor | None = None    # (B, S, T) per-token cumulative exit probabilities
    mean_exit_step: float | None = None  # average exit step across tokens

    def __post_init__(self) -> None:
        if self.mean_exit_step is None:
            if self.exit_steps is not None:
                self.mean_exit_step = float(self.exit_steps.float().mean().item())
            else:
                self.mean_exit_step = float(self.exit_step)
        if self.exit_steps is None:
            B, S = self.logits.shape[:2]
            self.exit_steps = torch.full(
                (B, S),
                int(self.exit_step),
                dtype=torch.long,
                device=self.logits.device,
            )


# ---------------------------------------------------------------------------
# Helper: token-level exit probability distribution
# ---------------------------------------------------------------------------

def compute_token_exit_distribution(gate_list: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Compute per-token discrete exit PDF and CDF from gate predictions across loop steps.

    Matches the official ByteDance Ouro reference implementation:
    https://huggingface.co/ByteDance/Ouro-1.4B/blob/main/modeling_ouro.py#L730-L741

    Args:
        gate_list: list of N tensors of shape (B, S), where gate_list[t] is λ_{t+1} ∈ [0, 1].

    Returns:
        (stacked_exit_pdf, stacked_exit_cdf) both of shape (B, S, N),
        where pdf sums to 1.0 along dim 2 for each token, and cdf[..., -1] == 1.0.
    """
    N = len(gate_list)
    if N == 0:
        raise ValueError("gate_list cannot be empty")

    pdf_list: list[Tensor] = []
    remaining_prob = torch.ones_like(gate_list[0])  # (B, S)

    for idx, lam in enumerate(gate_list):
        if idx < N - 1:
            p_i = lam * remaining_prob
            remaining_prob = remaining_prob * (1.0 - lam)
        else:
            p_i = remaining_prob  # absorb all remaining probability mass at final step
        pdf_list.append(p_i)

    stacked_exit_pdf = torch.stack(pdf_list, dim=2)  # (B, S, N)
    stacked_exit_cdf = torch.cumsum(stacked_exit_pdf, dim=2)  # (B, S, N)
    return stacked_exit_pdf, stacked_exit_cdf


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
        cdf: float | Tensor,
    ) -> bool:
        """Return True to stop running further recurrent steps.

        Args:
            step:        1-indexed current recurrent step
            hidden:      final-layer hidden state (B, S, hidden_size)
            exit_lambda: gate output λ_t (B, S)
            cdf:         cumulative exit probability up to and including this step (scalar or (B, S))
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

    def should_exit(
        self,
        step: int,
        *,
        hidden: Tensor,
        exit_lambda: Tensor,
        cdf: float | Tensor,
    ) -> bool:
        return step >= self.step


# ---------------------------------------------------------------------------
# Strategy 2: HiddenStateDiffExit
# ---------------------------------------------------------------------------

class HiddenStateDiffExit(EarlyExitStrategy):
    """Exit when consecutive hidden states converge within a threshold.

    Supports token-level diff tracking as well as sequence RMS.
    At step 1 there is no previous hidden state, so the model never exits
    early at step 1 under this strategy.
    """

    def __init__(self, threshold: float, mode: str = "token") -> None:
        self.threshold = threshold
        self.mode = mode
        self._prev_hidden: Tensor | None = None

    def reset(self) -> None:
        self._prev_hidden = None

    def should_exit(
        self,
        step: int,
        *,
        hidden: Tensor,
        exit_lambda: Tensor,
        cdf: float | Tensor,
    ) -> bool:
        if self._prev_hidden is None:
            self._prev_hidden = hidden.detach().clone()
            return False

        diff = hidden.detach() - self._prev_hidden
        self._prev_hidden = hidden.detach().clone()

        if self.mode == "token":
            # Per-token RMS diff over hidden dimension: shape (B, S)
            token_rms = diff.pow(2).mean(dim=-1).sqrt()
            return bool((token_rms < self.threshold).all().item())
        else:
            rms = (diff.pow(2).mean()).sqrt().item()
            return rms < self.threshold


# ---------------------------------------------------------------------------
# Strategy 3: QExit
# ---------------------------------------------------------------------------

class QExit(EarlyExitStrategy):
    """Exit when cumulative exit probability (CDF) reaches q_threshold.

    Supports token-level exit (default, matching ByteDance Ouro-1.4B),
    sample-level exit, and global exit.

    In token-level mode:
        Each token (b, s) has an exit probability:
            p_t(b, s) = λ_t(b, s) · survival_{t-1}(b, s)
            CDF_t(b, s) = Σ_{j=1}^{t} p_j(b, s)
        Tokens independently exit when CDF_t(b, s) >= q_threshold.

    Special cases:
        q=0  → always exit after step 1 (CDF after step 1 >= 0)
        q=1  → always run all steps (CDF reaches 1.0 only at the last absorbing step)
    """

    def __init__(
        self,
        q_threshold: float,
        mode: str = "token",
        weighted: bool = False,
    ) -> None:
        if not 0.0 <= q_threshold <= 1.0:
            raise ValueError(f"q_threshold must be in [0, 1], got {q_threshold}")
        if mode not in ("token", "sample", "global"):
            raise ValueError(f"mode must be one of 'token', 'sample', 'global', got {mode}")
        self.q_threshold = q_threshold
        self.mode = mode
        self.weighted = weighted
        self._cdf: float | Tensor = 0.0
        self._survival: float | Tensor = 1.0

    def reset(self) -> None:
        self._cdf = 0.0
        self._survival = 1.0

    def update_cdf(self, lam: float | Tensor, is_last_step: bool) -> float | Tensor:
        """Advance the CDF by one step and return the updated value."""
        if isinstance(lam, Tensor):
            if isinstance(self._survival, (int, float)):
                self._survival = torch.ones_like(lam)
                self._cdf = torch.zeros_like(lam)
            if is_last_step:
                self._cdf = self._cdf + self._survival
                self._survival = torch.zeros_like(self._survival)
            else:
                p_t = lam * self._survival
                self._cdf = self._cdf + p_t
                self._survival = self._survival * (1.0 - lam)
            return self._cdf
        else:
            if is_last_step:
                self._cdf += self._survival
                self._survival = 0.0
            else:
                p_t = lam * self._survival
                self._cdf += p_t
                self._survival *= 1.0 - lam
            return self._cdf

    def should_exit(
        self,
        step: int,
        *,
        hidden: Tensor,
        exit_lambda: Tensor,
        cdf: float | Tensor,
    ) -> bool:
        if isinstance(cdf, Tensor):
            if self.mode == "token":
                # True when all tokens in the batch have met their threshold
                return bool((cdf >= self.q_threshold).all().item())
            elif self.mode == "sample":
                # True when the final (generated) token of each sample has met threshold
                return bool((cdf[:, -1] >= self.q_threshold).all().item())
            else:
                return float(cdf.mean().item()) >= self.q_threshold
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
    """Run LoopLM recurrently with adaptive token-level early exit.

    Each token position (b, s) can exit at the step where its cumulative exit
    probability reaches the threshold, matching ByteDance Ouro-1.4B.
    Hidden states are gathered per-token from each token's selected exit step.

    Args:
        model:      trained LoopLM (in eval mode)
        input_ids:  (B, S) token ids
        strategy:   early-exit strategy instance (e.g. QExit, StaticExit, HiddenStateDiffExit)
        max_steps:  maximum recurrent steps; defaults to model.config.max_recurrent_steps

    Returns:
        EarlyExitResult with per-token gathered logits and exit statistics.
    """
    strategy.reset()
    max_steps = max_steps or model.config.max_recurrent_steps

    B, S = input_ids.shape
    cos, sin = model.rope.get_cos_sin(S, input_ids.device)

    h = model.embed(input_ids)  # (B, S, hidden_size)

    hidden_states_list: list[Tensor] = []
    logits_list: list[Tensor] = []
    gate_list: list[Tensor] = []

    for step in range(1, max_steps + 1):
        # One recurrent pass through the shared layer stack
        for layer in model.layers:
            h = layer(h, cos, sin)

        hidden_states_list.append(h)
        norm_h = model.final_norm(h)
        step_logits = model.lm_head(norm_h)  # (B, S, vocab_size)
        logits_list.append(step_logits)

        lam = torch.sigmoid(model.exit_gate(h)).squeeze(-1)  # (B, S)
        gate_list.append(lam)

        is_last = (step == max_steps)
        if isinstance(strategy, QExit):
            cdf = strategy.update_cdf(lam, is_last_step=is_last)
        else:
            cdf = 0.0

        if strategy.should_exit(step, hidden=h, exit_lambda=lam, cdf=cdf):
            break

    N = len(hidden_states_list)

    # Compute exact PDF and CDF over the executed steps
    stacked_exit_pdf, stacked_exit_cdf = compute_token_exit_distribution(gate_list)

    # Determine per-token exit steps (1-indexed, shape (B, S))
    if isinstance(strategy, StaticExit):
        exit_step_val = min(strategy.step, N)
        exit_steps = torch.full((B, S), exit_step_val, dtype=torch.long, device=input_ids.device)
    elif isinstance(strategy, QExit):
        if strategy.mode == "global":
            # Global mean exit step
            threshold_mask = (stacked_exit_cdf.mean(dim=(0, 1)) >= strategy.q_threshold)
            step_idx = int(torch.argmax(threshold_mask.float()).item()) if threshold_mask.any() else (N - 1)
            exit_steps = torch.full((B, S), step_idx + 1, dtype=torch.long, device=input_ids.device)
        elif strategy.mode == "sample":
            # Per-sample exit based on sequence's last token
            last_tok_cdf = stacked_exit_cdf[:, -1, :]  # (B, N)
            threshold_mask = last_tok_cdf >= strategy.q_threshold  # (B, N)
            exit_idx = torch.argmax(threshold_mask.float(), dim=1)  # (B,)
            never_exceeded = ~threshold_mask.any(dim=1)
            exit_idx[never_exceeded] = N - 1
            exit_steps = (exit_idx + 1).unsqueeze(-1).expand(B, S)
        else:
            # Token-level threshold exit (matching ByteDance Ouro-1.4B modeling_ouro.py#L788-798)
            threshold_mask = stacked_exit_cdf >= strategy.q_threshold  # (B, S, N)
            exit_steps_idx = torch.argmax(threshold_mask.float(), dim=2)  # (B, S) 0-indexed
            never_exceeded = ~threshold_mask.any(dim=2)
            exit_steps_idx[never_exceeded] = N - 1
            exit_steps = exit_steps_idx + 1  # 1-indexed (B, S)
    elif isinstance(strategy, HiddenStateDiffExit):
        if N == 1:
            exit_steps = torch.ones((B, S), dtype=torch.long, device=input_ids.device)
        else:
            diff_mask = torch.zeros((B, S, N), dtype=torch.bool, device=input_ids.device)
            for s_idx in range(1, N):
                diff = (hidden_states_list[s_idx] - hidden_states_list[s_idx - 1]).pow(2).mean(dim=-1).sqrt()
                diff_mask[..., s_idx] = (diff < strategy.threshold)
            exit_steps_idx = torch.argmax(diff_mask.float(), dim=2)
            never_met = ~diff_mask.any(dim=2)
            exit_steps_idx[never_met] = N - 1
            exit_steps = exit_steps_idx + 1
    else:
        exit_steps = torch.full((B, S), N, dtype=torch.long, device=input_ids.device)

    # Gather logits per token or compute expectation
    if isinstance(strategy, QExit) and strategy.weighted:
        # Expected logits: sum_t p_t * logits_t (modeling_ouro.py#L756-760)
        final_logits = torch.zeros_like(logits_list[0])
        for step_idx in range(N):
            weight = stacked_exit_pdf[..., step_idx].unsqueeze(-1)  # (B, S, 1)
            final_logits = final_logits + logits_list[step_idx] * weight
    else:
        # Per-token gathered hidden state and logits (modeling_ouro.py#L799-808)
        stacked_hidden = torch.stack(hidden_states_list, dim=2)  # (B, S, N, D)
        gather_index = (exit_steps - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, stacked_hidden.size(-1))
        final_hidden = torch.gather(stacked_hidden, 2, gather_index).squeeze(2)  # (B, S, D)
        final_logits = model.lm_head(model.final_norm(final_hidden))  # (B, S, vocab_size)

    # Determine scalar exit_step representation
    if (exit_steps == exit_steps.flatten()[0]).all():
        exit_step_scalar = int(exit_steps.flatten()[0].item())
    else:
        exit_step_scalar = int(round(exit_steps.float().mean().item()))

    return EarlyExitResult(
        logits=final_logits,
        exit_step=exit_step_scalar,
        exit_steps=exit_steps,
        exit_pdf=stacked_exit_pdf,
        exit_cdf=stacked_exit_cdf,
        mean_exit_step=float(exit_steps.float().mean().item()),
    )
