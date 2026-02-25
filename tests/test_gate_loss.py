import torch
import pytest
from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.training.objectives import compute_adaptive_gate_loss


@pytest.fixture
def cfg():
    return LoopLMConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        max_seq_len=16,
        max_recurrent_steps=4,
    )


@pytest.fixture
def model(cfg):
    torch.manual_seed(0)
    return LoopLM(cfg)


def make_batch(cfg, B=2, S=8, seed=42):
    torch.manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (B, S))


# ── Basic output structure ────────────────────────────────────────────────────

def test_loss_is_scalar(cfg, model):
    ids = make_batch(cfg)
    out = model(ids)
    loss, _ = compute_adaptive_gate_loss(out.logits, out.exit_lambdas, ids)
    assert loss.shape == ()


def test_loss_is_finite(cfg, model):
    ids = make_batch(cfg)
    out = model(ids)
    loss, _ = compute_adaptive_gate_loss(out.logits, out.exit_lambdas, ids)
    assert torch.isfinite(loss)


def test_diagnostics_keys(cfg, model):
    ids = make_batch(cfg)
    out = model(ids)
    _, diag = compute_adaptive_gate_loss(out.logits, out.exit_lambdas, ids)
    for key in ("loss", "mean_w_per_step", "per_step_bce"):
        assert key in diag


def test_per_step_bce_count(cfg, model):
    """BCE losses are computed for steps 2..T, so T-1 entries."""
    ids = make_batch(cfg)
    out = model(ids)
    _, diag = compute_adaptive_gate_loss(out.logits, out.exit_lambdas, ids)
    assert len(diag["per_step_bce"]) == cfg.max_recurrent_steps - 1
    assert len(diag["mean_w_per_step"]) == cfg.max_recurrent_steps - 1


def test_requires_at_least_two_steps(cfg, model):
    ids = make_batch(cfg)
    out = model(ids, num_steps=1)
    with pytest.raises(AssertionError):
        compute_adaptive_gate_loss(out.logits, out.exit_lambdas, ids)


# ── Gate learning signal ──────────────────────────────────────────────────────

def test_large_improvement_produces_high_w():
    """When step t greatly reduces loss, the ideal label w should be close to 1."""
    B, S, V, T = 1, 4, 16, 2
    targets = torch.zeros(B, S, dtype=torch.long)

    # Step 1: very high loss (wrong predictions)
    bad_logits = torch.full((B, S, V), -10.0)
    bad_logits[..., 1] = 10.0  # predicts token 1, target is 0

    # Step 2: very low loss (correct predictions)
    good_logits = torch.full((B, S, V), -10.0)
    good_logits[..., 0] = 10.0  # predicts token 0 = target

    lams = [torch.full((B, S), 0.5)] * T
    _, diag = compute_adaptive_gate_loss(
        [bad_logits, good_logits], lams, targets
    )
    # Large improvement → w ≈ 1 → gate should be told to continue
    assert diag["mean_w_per_step"][0] > 0.9


def test_no_improvement_produces_low_w():
    """When step t doesn't improve over step t-1, w should lean toward exit (< 0.5).

    With zero improvement: I=0, w = sigmoid(50*(0 - 0.005)) = sigmoid(-0.25) ≈ 0.44.
    The clamping at 0 means we can't get lower than ~0.44 with identical logits,
    but it's still clearly below 0.5 (the neutral point).
    """
    B, S, V, T = 1, 4, 16, 2
    targets = torch.zeros(B, S, dtype=torch.long)

    # Both steps: same wrong logits (no improvement)
    bad_logits = torch.full((B, S, V), -10.0)
    bad_logits[..., 1] = 10.0

    lams = [torch.full((B, S), 0.5)] * T
    _, diag = compute_adaptive_gate_loss(
        [bad_logits, bad_logits], lams, targets
    )
    # No improvement → I=0 → w < 0.5 (leaning toward exit)
    assert diag["mean_w_per_step"][0] < 0.5


def test_gate_learns_to_continue_when_improving():
    """BCE loss should be lower when gate predicts continuation for an improving step."""
    B, S, V = 1, 4, 16
    targets = torch.zeros(B, S, dtype=torch.long)

    bad_logits = torch.full((B, S, V), -10.0)
    bad_logits[..., 1] = 10.0
    good_logits = torch.full((B, S, V), -10.0)
    good_logits[..., 0] = 10.0  # correct

    # Gate that wants to continue: λ ≈ 0 → (1 - λ) ≈ 1
    lams_continue = [torch.full((B, S), 0.5), torch.full((B, S), 0.01)]
    # Gate that wants to exit: λ ≈ 1 → (1 - λ) ≈ 0
    lams_exit = [torch.full((B, S), 0.5), torch.full((B, S), 0.99)]

    loss_continue, _ = compute_adaptive_gate_loss(
        [bad_logits, good_logits], lams_continue, targets
    )
    loss_exit, _ = compute_adaptive_gate_loss(
        [bad_logits, good_logits], lams_exit, targets
    )
    # Continuing is correct here (large improvement), so its loss should be lower
    assert loss_continue < loss_exit


def test_gate_learns_to_exit_when_not_improving():
    """BCE loss should be lower when gate predicts exit for a non-improving step."""
    B, S, V = 1, 4, 16
    targets = torch.zeros(B, S, dtype=torch.long)

    bad_logits = torch.full((B, S, V), -10.0)
    bad_logits[..., 1] = 10.0

    lams_continue = [torch.full((B, S), 0.5), torch.full((B, S), 0.01)]
    lams_exit = [torch.full((B, S), 0.5), torch.full((B, S), 0.99)]

    loss_continue, _ = compute_adaptive_gate_loss(
        [bad_logits, bad_logits], lams_continue, targets
    )
    loss_exit, _ = compute_adaptive_gate_loss(
        [bad_logits, bad_logits], lams_exit, targets
    )
    # No improvement → exit is correct → exit loss should be lower
    assert loss_exit < loss_continue


# ── Gradient flow ─────────────────────────────────────────────────────────────

def test_no_gradient_through_lm_params(cfg, model):
    """LM parameters must not receive gradients when frozen (as done in Stage II).

    The loss function detaches logits; gradient isolation of other LM params is
    the trainer's responsibility via requires_grad=False. This test simulates that.
    """
    # Stage II: freeze everything except the exit gate
    for name, p in model.named_parameters():
        if "exit_gate" not in name:
            p.requires_grad_(False)

    ids = make_batch(cfg)
    out = model(ids)
    loss, _ = compute_adaptive_gate_loss(out.logits, out.exit_lambdas, ids)
    loss.backward()

    lm_params_with_grad = [
        name for name, p in model.named_parameters()
        if p.grad is not None
        and p.grad.abs().sum() > 0
        and "exit_gate" not in name
    ]
    assert lm_params_with_grad == [], (
        f"Frozen LM params received gradients: {lm_params_with_grad}"
    )


def test_gradient_flows_to_gate(cfg, model):
    ids = make_batch(cfg)
    out = model(ids)
    loss, _ = compute_adaptive_gate_loss(out.logits, out.exit_lambdas, ids)
    loss.backward()

    assert model.exit_gate.weight.grad is not None
    assert model.exit_gate.weight.grad.abs().sum() > 0
