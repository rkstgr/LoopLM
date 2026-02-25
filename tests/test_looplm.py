import torch
import pytest
from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM, LoopLMOutput, compute_exit_distribution


@pytest.fixture
def cfg():
    # Tiny config to keep tests fast
    return LoopLMConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        intermediate_size=128,
        max_seq_len=32,
        max_recurrent_steps=4,
    )


@pytest.fixture
def model(cfg):
    torch.manual_seed(0)
    return LoopLM(cfg)


@pytest.fixture
def input_ids(cfg):
    torch.manual_seed(1)
    return torch.randint(0, cfg.vocab_size, (2, 8))  # (B=2, S=8)


# ── Output structure ─────────────────────────────────────────────────────────

def test_returns_correct_number_of_steps(model, cfg, input_ids):
    out = model(input_ids)
    assert len(out.logits) == cfg.max_recurrent_steps
    assert len(out.exit_lambdas) == cfg.max_recurrent_steps


def test_logits_shape(model, cfg, input_ids):
    out = model(input_ids)
    B, S = input_ids.shape
    for t, logits in enumerate(out.logits):
        assert logits.shape == (B, S, cfg.vocab_size), f"Wrong shape at step {t}"


def test_exit_lambdas_shape(model, cfg, input_ids):
    out = model(input_ids)
    B, S = input_ids.shape
    for t, lam in enumerate(out.exit_lambdas):
        assert lam.shape == (B, S), f"Wrong lambda shape at step {t}"


def test_exit_lambdas_in_unit_interval(model, input_ids):
    out = model(input_ids)
    for lam in out.exit_lambdas:
        assert lam.min() >= 0.0 and lam.max() <= 1.0


# ── Exit distribution ────────────────────────────────────────────────────────

def test_exit_distribution_sums_to_one(model, input_ids):
    out = model(input_ids)
    probs = compute_exit_distribution(out.exit_lambdas)  # (T, B, S)
    total = probs.sum(dim=0)  # (B, S)
    torch.testing.assert_close(total, torch.ones_like(total), atol=1e-5, rtol=1e-5)


def test_exit_distribution_non_negative(model, input_ids):
    out = model(input_ids)
    probs = compute_exit_distribution(out.exit_lambdas)
    assert probs.min() >= 0.0


def test_exit_distribution_single_step():
    """With T=1, all probability mass must go to the last step."""
    lam = [torch.full((1, 4), 0.8)]  # T=1, λ=0.8 (doesn't matter for T=1)
    probs = compute_exit_distribution(lam)
    torch.testing.assert_close(probs, torch.ones_like(probs), atol=1e-6, rtol=1e-6)


# ── Custom num_steps ─────────────────────────────────────────────────────────

def test_custom_num_steps(model, cfg, input_ids):
    for T in [1, 2, cfg.max_recurrent_steps]:
        out = model(input_ids, num_steps=T)
        assert len(out.logits) == T
        assert len(out.exit_lambdas) == T


def test_t1_produces_valid_logits(model, cfg, input_ids):
    """With a single recurrent step the model should behave like a standard LM."""
    out = model(input_ids, num_steps=1)
    assert len(out.logits) == 1
    logits = out.logits[0]
    assert logits.shape == (input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)
    assert torch.isfinite(logits).all()


# ── Gradient flow ─────────────────────────────────────────────────────────────

def test_gradients_flow_to_all_params(model, input_ids):
    out = model(input_ids)
    # Simple loss: sum of all logits across all steps
    loss = sum(l.sum() for l in out.logits) + sum(g.sum() for g in out.exit_lambdas)
    loss.backward()
    for name, p in model.named_parameters():
        # lm_head.weight is tied to embed.weight — skip the alias
        if name == "lm_head.weight":
            continue
        assert p.grad is not None, f"No grad for {name}"
        assert not torch.all(p.grad == 0), f"Zero grad for {name}"


# ── Weight sharing ────────────────────────────────────────────────────────────

def test_lm_head_weight_tied_to_embedding(model):
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()


def test_layer_weights_are_independent(model):
    """Each layer in the stack should be a distinct module with its own weights."""
    if len(model.layers) < 2:
        pytest.skip("Need at least 2 layers")
    w0 = model.layers[0].attn.q_proj.weight
    w1 = model.layers[1].attn.q_proj.weight
    assert w0.data_ptr() != w1.data_ptr()


# ── Recurrent steps improve loss ──────────────────────────────────────────────

def test_later_steps_have_different_logits(model, input_ids):
    """Hidden state must change between recurrent steps."""
    out = model(input_ids, num_steps=2)
    assert not torch.allclose(out.logits[0], out.logits[1])
