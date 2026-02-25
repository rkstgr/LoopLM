import torch
import torch.nn.functional as F
from torch import Tensor

from src.model.looplm import compute_exit_distribution


def compute_looplm_loss(
    logits_per_step: list[Tensor],
    exit_lambdas: list[Tensor],
    targets: Tensor,
    beta: float = 0.1,
    ignore_index: int = -100,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    """Stage I entropy-regularized training objective (paper Eq. 4).

    L = Σ_t p(t|x) · L^(t) - β · H(p(·|x))

    The entropy term prevents the exit distribution from collapsing to always
    use the last step. With a uniform prior this is equivalent to an ELBO.

    Args:
        logits_per_step: T tensors of shape (B, S, vocab_size)
        exit_lambdas:    T tensors of shape (B, S) — raw sigmoid gate outputs λ_t
        targets:         (B, S) ground-truth token ids; use ignore_index for padding
        beta:            entropy regularization weight (0.1 in Stage I)
        ignore_index:    token id to ignore in CE loss (default -100)

    Returns:
        total_loss:  scalar tensor, differentiable w.r.t. both LM and gate params
        diagnostics: dict with detached scalars for logging
    """
    T = len(logits_per_step)
    B, S, V = logits_per_step[0].shape

    # ── Per-step cross-entropy losses ────────────────────────────────────────
    # L^(t): (B, S) — per-token CE at each recurrent step
    ce_per_step: list[Tensor] = []
    for logits in logits_per_step:
        ce = F.cross_entropy(
            logits.reshape(B * S, V),
            targets.reshape(B * S),
            ignore_index=ignore_index,
            reduction="none",
        ).view(B, S)
        ce_per_step.append(ce)

    # ── Exit distribution ────────────────────────────────────────────────────
    # p(t|x): (T, B, S) — sums to 1 over dim 0
    exit_probs = compute_exit_distribution(exit_lambdas)

    # ── Valid-token mask ─────────────────────────────────────────────────────
    mask = (targets != ignore_index).float()  # (B, S)
    n_valid = mask.sum().clamp(min=1.0)

    # ── Weighted task loss ───────────────────────────────────────────────────
    # Σ_t p(t|x) · L^(t), averaged over valid tokens
    ce_stack = torch.stack(ce_per_step, dim=0)         # (T, B, S)
    weighted_ce = (exit_probs * ce_stack).sum(dim=0)   # (B, S)
    task_loss = (weighted_ce * mask).sum() / n_valid

    # ── Entropy regularization ───────────────────────────────────────────────
    # H(p) = -Σ_t p(t) · log p(t), averaged over valid tokens
    log_probs = torch.log(exit_probs.clamp(min=1e-10))
    entropy = -(exit_probs * log_probs).sum(dim=0)     # (B, S)
    mean_entropy = (entropy * mask).sum() / n_valid

    # ── Total loss ───────────────────────────────────────────────────────────
    total_loss = task_loss - beta * mean_entropy

    # ── Diagnostics (detached) ───────────────────────────────────────────────
    with torch.no_grad():
        step_indices = torch.arange(1, T + 1, dtype=torch.float32, device=exit_probs.device)
        avg_exit_step = (exit_probs * step_indices.view(T, 1, 1)).sum(dim=0)  # (B, S)
        mean_avg_exit_step = (avg_exit_step * mask).sum() / n_valid

    diagnostics: dict[str, Tensor | float] = {
        "loss": total_loss.detach(),
        "task_loss": task_loss.detach(),
        "entropy": mean_entropy.detach(),
        "avg_exit_step": mean_avg_exit_step.detach(),
        "per_step_losses": [ce.mean().detach() for ce in ce_per_step],
    }

    return total_loss, diagnostics


def compute_adaptive_gate_loss(
    logits_per_step: list[Tensor],
    exit_lambdas: list[Tensor],
    targets: Tensor,
    k: float = 50.0,
    gamma: float = 0.005,
    ignore_index: int = -100,
) -> tuple[Tensor, dict[str, Tensor | list]]:
    """Stage II adaptive gate training objective (paper Eq. 6).

    LM parameters are frozen by the caller; this loss only flows gradients
    through exit_lambdas (the gate). All CE computations use detached logits.

    For each step t = 2..T_max:
      I^(t)  = max(0, L^(t-1)_detached − L^(t)_detached)   per-token improvement
      w^(t)  = sigmoid(k · (I^(t) − γ))                    ideal continuation label
      loss_t = BCE(1 − λ^(t), w^(t))                        gate should continue iff w≈1

    Args:
        logits_per_step: T tensors of shape (B, S, vocab_size)
        exit_lambdas:    T tensors of shape (B, S) — raw sigmoid gate outputs λ_t
        targets:         (B, S) ground-truth token ids
        k:               sharpness of the sigmoid label (paper: 50.0)
        gamma:           improvement threshold below which gate should exit (paper: 0.005)
        ignore_index:    token id to ignore in CE loss

    Returns:
        total_loss:  scalar, differentiable w.r.t. gate params only (logits are detached)
        diagnostics: dict with detached scalars for logging
    """
    T = len(logits_per_step)
    assert T >= 2, "Adaptive gate loss requires at least 2 recurrent steps"
    B, S, V = logits_per_step[0].shape

    # ── Detached per-step CE losses ───────────────────────────────────────────
    # All CE is computed with detached logits — LM gets no gradients here
    ce_per_step: list[Tensor] = []
    for logits in logits_per_step:
        ce = F.cross_entropy(
            logits.detach().reshape(B * S, V),
            targets.reshape(B * S),
            ignore_index=ignore_index,
            reduction="none",
        ).view(B, S)
        ce_per_step.append(ce)

    # ── Valid-token mask ──────────────────────────────────────────────────────
    mask = (targets != ignore_index).float()  # (B, S)
    n_valid = mask.sum().clamp(min=1.0)

    # ── Per-step BCE losses (t = 2..T, i.e. index 1..T-1) ────────────────────
    bce_per_step: list[Tensor] = []
    mean_w_per_step: list[float] = []

    for t in range(1, T):
        # Improvement: how much did step t reduce the loss vs step t-1?
        improvement = (ce_per_step[t - 1] - ce_per_step[t]).clamp(min=0.0)  # (B, S)

        # Ideal continuation label: ≈1 when improvement is large, ≈0 otherwise
        w = torch.sigmoid(k * (improvement - gamma))  # (B, S)

        # Gate should predict "continue" (high 1−λ) when w≈1
        continuation = (1.0 - exit_lambdas[t]).clamp(min=1e-7, max=1 - 1e-7)
        bce = F.binary_cross_entropy(continuation, w, reduction="none")  # (B, S)
        bce_per_step.append(bce)

        with torch.no_grad():
            mean_w_per_step.append(((w * mask).sum() / n_valid).item())

    # ── Average over steps and valid tokens ───────────────────────────────────
    bce_stack = torch.stack(bce_per_step, dim=0)  # (T-1, B, S)
    total_loss = (bce_stack * mask.unsqueeze(0)).sum() / (n_valid * (T - 1))

    diagnostics: dict[str, Tensor | list] = {
        "loss": total_loss.detach(),
        "mean_w_per_step": mean_w_per_step,   # avg ideal-continuation label per step
        "per_step_bce": [b.mean().detach() for b in bce_per_step],
    }

    return total_loss, diagnostics
