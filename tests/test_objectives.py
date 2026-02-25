import torch
import pytest
from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.training.objectives import compute_looplm_loss


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
    loss, _ = compute_looplm_loss(out.logits, out.exit_lambdas, ids)
    assert loss.shape == ()


def test_loss_is_finite(cfg, model):
    ids = make_batch(cfg)
    out = model(ids)
    loss, _ = compute_looplm_loss(out.logits, out.exit_lambdas, ids)
    assert torch.isfinite(loss)


def test_diagnostics_keys(cfg, model):
    ids = make_batch(cfg)
    out = model(ids)
    _, diag = compute_looplm_loss(out.logits, out.exit_lambdas, ids)
    for key in ("loss", "task_loss", "entropy", "avg_exit_step", "per_step_losses"):
        assert key in diag, f"Missing diagnostic key: {key}"


def test_per_step_losses_count(cfg, model):
    ids = make_batch(cfg)
    out = model(ids)
    _, diag = compute_looplm_loss(out.logits, out.exit_lambdas, ids)
    assert len(diag["per_step_losses"]) == cfg.max_recurrent_steps


# ── Correctness of the objective ─────────────────────────────────────────────

def test_correct_logits_give_lower_loss(cfg):
    """Loss should be lower when logits strongly predict the correct token."""
    torch.manual_seed(0)
    B, S, V, T = 2, 6, cfg.vocab_size, 4
    targets = torch.randint(0, V, (B, S))

    # Build near-perfect logits: large score on correct token
    perfect_logits = torch.full((B, S, V), -10.0)
    perfect_logits.scatter_(2, targets.unsqueeze(-1), 20.0)

    random_logits = torch.randn(B, S, V)

    # Uniform exit lambdas (doesn't affect relative comparison)
    lams = [torch.full((B, S), 0.5) for _ in range(T)]

    loss_perfect, _ = compute_looplm_loss(
        [perfect_logits] * T, lams, targets
    )
    loss_random, _ = compute_looplm_loss(
        [random_logits] * T, lams, targets
    )
    assert loss_perfect < loss_random


def test_task_loss_equals_weighted_ce(cfg):
    """task_loss must equal Σ_t p(t) · L^(t) computed manually."""
    from src.model.looplm import compute_exit_distribution
    import torch.nn.functional as F

    torch.manual_seed(1)
    B, S, V, T = 2, 6, cfg.vocab_size, 3
    targets = torch.randint(0, V, (B, S))
    logits = [torch.randn(B, S, V) for _ in range(T)]
    lams = [torch.rand(B, S) for _ in range(T)]

    _, diag = compute_looplm_loss(logits, lams, targets, beta=0.0)

    # Manual computation
    exit_probs = compute_exit_distribution(lams)  # (T, B, S)
    ce_stack = torch.stack([
        F.cross_entropy(l.reshape(B * S, V), targets.reshape(B * S), reduction="none").view(B, S)
        for l in logits
    ], dim=0)
    expected = (exit_probs * ce_stack).sum(dim=0).mean()

    torch.testing.assert_close(diag["task_loss"], expected, atol=1e-5, rtol=1e-5)


def test_entropy_term_sign():
    """Subtracting entropy should lower the total loss vs. task_loss alone (β > 0)."""
    B, S, V, T = 1, 4, 8, 3
    targets = torch.zeros(B, S, dtype=torch.long)
    logits = [torch.zeros(B, S, V) for _ in range(T)]
    lams = [torch.full((B, S), 0.5) for _ in range(T)]

    loss_with, diag = compute_looplm_loss(logits, lams, targets, beta=0.5)
    loss_without = diag["task_loss"]

    assert loss_with < loss_without, "entropy bonus should reduce total loss"


def test_avg_exit_step_in_range(cfg, model):
    ids = make_batch(cfg)
    out = model(ids)
    _, diag = compute_looplm_loss(out.logits, out.exit_lambdas, ids)
    avg = diag["avg_exit_step"].item()
    assert 1.0 <= avg <= cfg.max_recurrent_steps


# ── Ignore index (padding) ────────────────────────────────────────────────────

def test_ignore_index_does_not_affect_loss(cfg):
    """Padding tokens (ignore_index=-100) must be excluded from the loss."""
    torch.manual_seed(2)
    B, S, V, T = 2, 8, cfg.vocab_size, 3
    targets_full = torch.randint(0, V, (B, S))
    targets_padded = targets_full.clone()
    targets_padded[:, -2:] = -100  # last two tokens are padding

    logits = [torch.randn(B, S, V) for _ in range(T)]
    lams = [torch.rand(B, S) for _ in range(T)]

    # Loss computed on the first S-2 tokens with matching logits should be the
    # same whether we pass padded or truncated targets
    loss_padded, _ = compute_looplm_loss(logits, lams, targets_padded)
    loss_trunc, _ = compute_looplm_loss(
        [l[:, :-2, :] for l in logits],
        [g[:, :-2] for g in lams],
        targets_full[:, :-2],
    )
    torch.testing.assert_close(loss_padded, loss_trunc, atol=1e-5, rtol=1e-5)


# ── Gradient flow ─────────────────────────────────────────────────────────────

def test_gradients_flow_to_lm_params(cfg, model):
    ids = make_batch(cfg)
    out = model(ids)
    loss, _ = compute_looplm_loss(out.logits, out.exit_lambdas, ids)
    loss.backward()

    lm_params_with_grad = [
        name for name, p in model.named_parameters()
        if p.grad is not None and name != "lm_head.weight"  # tied alias
    ]
    assert len(lm_params_with_grad) > 0


def test_gradients_flow_to_gate_params(cfg, model):
    ids = make_batch(cfg)
    out = model(ids)
    loss, _ = compute_looplm_loss(out.logits, out.exit_lambdas, ids)
    loss.backward()

    assert model.exit_gate.weight.grad is not None
    assert not torch.all(model.exit_gate.weight.grad == 0)
