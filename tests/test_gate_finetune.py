"""Tests for GateFinetuner (Sprint 5.2 — Stage II gate fine-tuning)."""

import tempfile
from pathlib import Path

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from src.model.config import LoopLMConfig
from src.training.trainer import GateFinetuneConfig, GateFinetuner


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_cfg() -> LoopLMConfig:
    cfg = LoopLMConfig.small()
    cfg.num_layers = 2
    cfg.hidden_size = 64
    cfg.num_heads = 4
    cfg.intermediate_size = 128
    cfg.max_seq_len = 32
    cfg.max_recurrent_steps = 4
    return cfg


def _gate_cfg(**kwargs) -> GateFinetuneConfig:
    defaults = dict(
        max_steps=5,
        log_every=100,
        save_every=1000,
        checkpoint_dir="",
        device="cpu",
    )
    defaults.update(kwargs)
    return GateFinetuneConfig(**defaults)


def _random_batch(vocab_size: int = 100, seq_len: int = 16, batch_size: int = 2):
    return torch.randint(0, vocab_size, (batch_size, seq_len + 1))


def _fake_dataloader(vocab_size=100, seq_len=16, n=20, batch_size=2):
    tokens = torch.randint(0, vocab_size, (n, seq_len + 1))
    return DataLoader(TensorDataset(tokens), batch_size=batch_size, drop_last=True)


# ── Freeze behavior ───────────────────────────────────────────────────────────

def test_only_gate_params_are_trainable():
    finetuner = GateFinetuner(_tiny_cfg(), _gate_cfg())
    for name, param in finetuner.model.named_parameters():
        if "exit_gate" in name:
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_all_lm_params_frozen():
    finetuner = GateFinetuner(_tiny_cfg(), _gate_cfg())
    lm_param_names = [
        n for n, p in finetuner.model.named_parameters()
        if "exit_gate" not in n
    ]
    assert len(lm_param_names) > 0, "There should be LM params to freeze"
    for name in lm_param_names:
        param = dict(finetuner.model.named_parameters())[name]
        assert not param.requires_grad


def test_gate_params_count_matches_optimizer():
    finetuner = GateFinetuner(_tiny_cfg(), _gate_cfg())
    gate_numel = sum(p.numel() for p in finetuner.model.exit_gate.parameters())
    opt_numel = sum(
        p.numel() for pg in finetuner.optimizer.param_groups for p in pg["params"]
    )
    assert gate_numel == opt_numel


# ── Checkpoint loading ────────────────────────────────────────────────────────

def test_load_checkpoint_restores_weights():
    """Weights loaded from a saved Stage I checkpoint match the original."""
    model_cfg = _tiny_cfg()
    finetuner1 = GateFinetuner(model_cfg, _gate_cfg())

    with tempfile.TemporaryDirectory() as tmp:
        # Save via Trainer-compatible format
        path = Path(tmp) / "fake_ckpt.pt"
        torch.save(
            {
                "step": 42,
                "model_state_dict": finetuner1.model.state_dict(),
                "optimizer_state_dict": finetuner1.optimizer.state_dict(),
            },
            path,
        )

        finetuner2 = GateFinetuner(model_cfg, _gate_cfg())
        loaded_step = finetuner2.load_checkpoint(path)

        assert loaded_step == 42
        for (n1, p1), (n2, p2) in zip(
            finetuner1.model.named_parameters(),
            finetuner2.model.named_parameters(),
        ):
            assert torch.allclose(p1, p2), f"Mismatch in {n1}"


def test_load_checkpoint_refreezes_params():
    """After load_checkpoint, LM params must still be frozen."""
    model_cfg = _tiny_cfg()
    finetuner = GateFinetuner(model_cfg, _gate_cfg())

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        torch.save({"step": 0, "model_state_dict": finetuner.model.state_dict()}, path)
        finetuner.load_checkpoint(path)

    for name, param in finetuner.model.named_parameters():
        if "exit_gate" not in name:
            assert not param.requires_grad, f"{name} should still be frozen after load"


# ── train_step ────────────────────────────────────────────────────────────────

def test_train_step_returns_diagnostics():
    finetuner = GateFinetuner(_tiny_cfg(), _gate_cfg())
    diag = finetuner.train_step(_random_batch())
    assert "loss" in diag
    assert "per_step_bce" in diag
    assert "mean_w_per_step" in diag


def test_train_step_increments_step():
    finetuner = GateFinetuner(_tiny_cfg(), _gate_cfg())
    assert finetuner.step == 0
    finetuner.train_step(_random_batch())
    assert finetuner.step == 1


def test_train_step_loss_is_finite():
    finetuner = GateFinetuner(_tiny_cfg(), _gate_cfg())
    diag = finetuner.train_step(_random_batch())
    assert torch.isfinite(diag["loss"])


def test_train_step_only_gate_params_change():
    """LM parameters must not change after a gate training step."""
    model_cfg = _tiny_cfg()
    finetuner = GateFinetuner(model_cfg, _gate_cfg())

    # Snapshot frozen LM params before step
    lm_before = {
        n: p.clone() for n, p in finetuner.model.named_parameters()
        if "exit_gate" not in n
    }

    finetuner.train_step(_random_batch())

    for name, before in lm_before.items():
        after = dict(finetuner.model.named_parameters())[name]
        assert torch.equal(before, after), f"LM param {name} changed during gate step"


def test_train_step_gate_params_do_change():
    """Exit gate parameters must be updated after a training step."""
    model_cfg = _tiny_cfg()
    finetuner = GateFinetuner(model_cfg, _gate_cfg(lr=1e-2))

    gate_before = {
        n: p.clone() for n, p in finetuner.model.exit_gate.named_parameters()
    }

    # Run multiple steps to ensure the gate actually moves
    for _ in range(5):
        finetuner.train_step(_random_batch())

    changed = False
    for name, before in gate_before.items():
        after = dict(finetuner.model.exit_gate.named_parameters())[name]
        if not torch.equal(before, after):
            changed = True
            break
    assert changed, "No gate parameter changed after 5 training steps"


def test_train_step_accepts_tuple_batch():
    """Trainer-style (list/tuple) batch wrapping must be unwrapped correctly."""
    finetuner = GateFinetuner(_tiny_cfg(), _gate_cfg())
    batch = [_random_batch()]  # TensorDataset returns list
    diag = finetuner.train_step(batch)
    assert torch.isfinite(diag["loss"])


# ── Checkpoint save ───────────────────────────────────────────────────────────

def test_save_checkpoint_filename():
    model_cfg = _tiny_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _gate_cfg(checkpoint_dir=tmp)
        finetuner = GateFinetuner(model_cfg, cfg)
        finetuner.train_step(_random_batch())  # step → 1
        path = finetuner.save_checkpoint()
        assert "gate_step_" in path.name
        assert path.exists()


def test_save_load_round_trip():
    model_cfg = _tiny_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _gate_cfg(checkpoint_dir=tmp)
        finetuner = GateFinetuner(model_cfg, cfg)
        for _ in range(3):
            finetuner.train_step(_random_batch())
        path = finetuner.save_checkpoint()

        finetuner2 = GateFinetuner(model_cfg, cfg)
        finetuner2.load_checkpoint(path)

        for (n1, p1), (n2, p2) in zip(
            finetuner.model.named_parameters(),
            finetuner2.model.named_parameters(),
        ):
            assert torch.allclose(p1, p2), f"Param mismatch after round-trip: {n1}"


# ── eval_avg_exit_step ────────────────────────────────────────────────────────

def test_eval_avg_exit_step_returns_expected_keys():
    finetuner = GateFinetuner(_tiny_cfg(), _gate_cfg(num_recurrent_steps=4))
    dl = _fake_dataloader()
    result = finetuner.eval_avg_exit_step(dl, n_batches=2, q_thresholds=(0.5, 0.9))
    assert "avg_exit_q0.5" in result
    assert "avg_exit_q0.9" in result


def test_eval_avg_exit_step_within_bounds():
    cfg = _gate_cfg(num_recurrent_steps=4)
    finetuner = GateFinetuner(_tiny_cfg(), cfg)
    dl = _fake_dataloader()
    result = finetuner.eval_avg_exit_step(dl, n_batches=3, q_thresholds=(0.5,))
    avg = result["avg_exit_q0.5"]
    assert 1.0 <= avg <= 4.0, f"avg_exit_step={avg} out of [1, 4]"


def test_eval_higher_q_gives_more_steps():
    """Higher q_threshold should (on average) use more recurrent steps."""
    cfg = _gate_cfg(num_recurrent_steps=4)
    finetuner = GateFinetuner(_tiny_cfg(), cfg)
    dl = _fake_dataloader(n=40)
    result = finetuner.eval_avg_exit_step(
        dl, n_batches=10, q_thresholds=(0.0, 1.0)
    )
    # q=0 exits at step 1, q=1 uses all steps
    assert result["avg_exit_q0.0"] <= result["avg_exit_q1.0"]


def test_eval_model_returns_to_train_mode():
    """eval_avg_exit_step must restore the model to train mode."""
    finetuner = GateFinetuner(_tiny_cfg(), _gate_cfg())
    finetuner.model.train()
    dl = _fake_dataloader()
    finetuner.eval_avg_exit_step(dl, n_batches=1)
    assert finetuner.model.training
