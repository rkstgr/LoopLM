"""Tests for multi-stage training support in Trainer."""

import math
import tempfile
from pathlib import Path

import torch
import pytest

from src.model.config import LoopLMConfig
from src.training.trainer import (
    StageConfig,
    TrainerConfig,
    Trainer,
    _cosine_lr,
    make_stage_1a,
    make_stage_1b,
    make_stage_2,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_config() -> LoopLMConfig:
    """Minimal model config for fast tests."""
    cfg = LoopLMConfig.small()
    cfg.num_layers = 2
    cfg.hidden_size = 64
    cfg.num_heads = 4
    cfg.intermediate_size = 128
    cfg.max_seq_len = 32
    cfg.max_recurrent_steps = 8
    return cfg


def _random_batch(seq_len: int = 16, batch_size: int = 2) -> list[torch.Tensor]:
    """Return a batch list (wrapping mimics DataLoader output)."""
    return [torch.randint(0, 100, (batch_size, seq_len + 1))]


def _get_optimizer_lr(trainer: Trainer) -> float:
    return trainer.optimizer.param_groups[0]["lr"]


# ── StageConfig factories ─────────────────────────────────────────────────────

def test_make_stage_1a_fields():
    s = make_stage_1a(1000)
    assert s.name == "1a"
    assert s.max_steps == 1000
    assert s.lr == pytest.approx(3e-4)
    assert s.lr_schedule == "constant"
    assert s.num_recurrent_steps == 8
    assert s.beta_kl == pytest.approx(0.1)
    assert s.rope_base == pytest.approx(10_000.0)


def test_make_stage_1b_fields():
    s = make_stage_1b(500)
    assert s.name == "1b"
    assert s.max_steps == 500
    assert s.lr == pytest.approx(3e-4)
    assert s.lr_schedule == "constant"
    assert s.num_recurrent_steps == 4
    assert s.beta_kl == pytest.approx(0.1)
    assert s.rope_base == pytest.approx(10_000.0)


def test_make_stage_2_fields():
    s = make_stage_2(200)
    assert s.name == "2"
    assert s.max_steps == 200
    assert s.lr == pytest.approx(3e-5)
    assert s.lr_schedule == "cosine"
    assert s.num_recurrent_steps == 4
    assert s.beta_kl == pytest.approx(0.05)
    assert s.rope_base == pytest.approx(40_000.0)


# ── _cosine_lr helper ─────────────────────────────────────────────────────────

def test_cosine_lr_at_step_zero():
    base_lr = 1e-3
    result = _cosine_lr(base_lr, step=0, total_steps=100)
    assert result == pytest.approx(base_lr)


def test_cosine_lr_at_total_steps():
    base_lr = 1e-3
    result = _cosine_lr(base_lr, step=100, total_steps=100)
    assert result == pytest.approx(base_lr * 0.1)


def test_cosine_lr_beyond_total_steps():
    base_lr = 1e-3
    result = _cosine_lr(base_lr, step=200, total_steps=100)
    assert result == pytest.approx(base_lr * 0.1)


def test_cosine_lr_decreases_monotonically():
    base_lr = 1e-3
    T = 50
    lrs = [_cosine_lr(base_lr, t, T) for t in range(T + 1)]
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1], f"LR not monotone at step {i}: {lrs[i]} < {lrs[i+1]}"


def test_cosine_lr_midpoint():
    base_lr = 1e-3
    T = 100
    mid = _cosine_lr(base_lr, step=50, total_steps=T)
    # cos(π/2) = 0, so mid = min_lr + 0.5*(base-min) * (1+0) = min_lr + 0.5*(base-min)
    min_lr = base_lr * 0.1
    expected = min_lr + 0.5 * (base_lr - min_lr)
    assert mid == pytest.approx(expected, rel=1e-5)


# ── Single-stage (backward compat) ────────────────────────────────────────────

def test_no_stages_no_regression():
    """stages=[] path must behave identically to old single-stage trainer."""
    model_cfg = _tiny_config()
    trainer_cfg = TrainerConfig(
        max_steps=5,
        log_every=100,
        save_every=1000,
    )
    trainer = Trainer(model_cfg, trainer_cfg)
    assert trainer.stage_idx == 0
    assert trainer._stage_start_step == 0
    assert trainer._stage_schedule is None

    for _ in range(5):
        trainer.train_step(_random_batch())

    assert trainer.step == 5
    assert trainer.stage_idx == 0  # unchanged


# ── Stage transition ──────────────────────────────────────────────────────────

def test_stage_transition_fires_at_correct_step():
    """After stage1a_steps steps, stage_idx should advance to 1."""
    model_cfg = _tiny_config()
    stages = [make_stage_1a(4), make_stage_1b(4)]
    trainer_cfg = TrainerConfig(stages=stages, log_every=100, save_every=1000)
    trainer = Trainer(model_cfg, trainer_cfg)

    assert trainer.config.max_steps == 8
    assert trainer.stage_idx == 0

    for _ in range(4):
        trainer.train_step(_random_batch())

    assert trainer.stage_idx == 1, f"Expected stage 1, got {trainer.stage_idx}"
    assert trainer._stage_start_step == 4


def test_stage_transition_updates_hyperparams():
    """After transitioning 1a→1b, T and β must reflect stage 1b values."""
    model_cfg = _tiny_config()
    stages = [make_stage_1a(3), make_stage_1b(3)]
    trainer_cfg = TrainerConfig(stages=stages, log_every=100, save_every=1000)
    trainer = Trainer(model_cfg, trainer_cfg)

    # During stage 1a
    assert trainer.config.num_recurrent_steps == 8
    assert trainer.config.beta_kl == pytest.approx(0.1)

    for _ in range(3):
        trainer.train_step(_random_batch())

    # After transition to stage 1b
    assert trainer.config.num_recurrent_steps == 4
    assert trainer.config.beta_kl == pytest.approx(0.1)
    assert trainer.model.rope.base == pytest.approx(10_000.0)
    assert _get_optimizer_lr(trainer) == pytest.approx(3e-4)


def test_stage_transition_updates_rope_and_lr():
    """1b→2 transition must update RoPE base and LR."""
    model_cfg = _tiny_config()
    stages = [make_stage_1b(3), make_stage_2(3)]
    trainer_cfg = TrainerConfig(stages=stages, log_every=100, save_every=1000)
    trainer = Trainer(model_cfg, trainer_cfg)

    for _ in range(3):
        trainer.train_step(_random_batch())

    assert trainer.stage_idx == 1
    assert trainer.config.num_recurrent_steps == 4
    assert trainer.config.beta_kl == pytest.approx(0.05)
    assert trainer.model.rope.base == pytest.approx(40_000.0)
    # LR at step 0 of cosine = base_lr
    assert _get_optimizer_lr(trainer) == pytest.approx(3e-5)


def test_three_stage_transitions():
    """All three stages fire in order: 1a → 1b → 2."""
    model_cfg = _tiny_config()
    stages = [make_stage_1a(2), make_stage_1b(2), make_stage_2(2)]
    trainer_cfg = TrainerConfig(stages=stages, log_every=100, save_every=1000)
    trainer = Trainer(model_cfg, trainer_cfg)

    assert trainer.stage_idx == 0
    assert trainer.config.max_steps == 6

    # Run stage 1a
    trainer.train_step(_random_batch())
    trainer.train_step(_random_batch())
    assert trainer.stage_idx == 1

    # Run stage 1b
    trainer.train_step(_random_batch())
    trainer.train_step(_random_batch())
    assert trainer.stage_idx == 2

    # Run stage 2
    trainer.train_step(_random_batch())
    trainer.train_step(_random_batch())
    assert trainer.stage_idx == 2  # no stage 3, stays at 2
    assert trainer.step == 6


def test_no_extra_transition_at_end():
    """stage_idx must not go out of bounds after all stages complete."""
    model_cfg = _tiny_config()
    stages = [make_stage_1a(2), make_stage_1b(2)]
    trainer_cfg = TrainerConfig(stages=stages, log_every=100, save_every=1000)
    trainer = Trainer(model_cfg, trainer_cfg)

    for _ in range(4):
        trainer.train_step(_random_batch())

    assert trainer.stage_idx == 1
    assert trainer.step == 4


# ── Cosine LR within a stage ─────────────────────────────────────────────────

def test_cosine_lr_decreases_during_stage2():
    """LR must decrease monotonically within stage 2."""
    model_cfg = _tiny_config()
    stages = [make_stage_2(10)]
    trainer_cfg = TrainerConfig(stages=stages, log_every=100, save_every=1000)
    trainer = Trainer(model_cfg, trainer_cfg)

    lrs = [_get_optimizer_lr(trainer)]
    for _ in range(10):
        trainer.train_step(_random_batch())
        lrs.append(_get_optimizer_lr(trainer))

    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1], f"LR not decreasing at step {i}: {lrs[i]:.2e} < {lrs[i+1]:.2e}"


# ── Checkpoint save/restore ───────────────────────────────────────────────────

def test_checkpoint_saves_stage_state():
    model_cfg = _tiny_config()
    stages = [make_stage_1a(3), make_stage_1b(3)]
    trainer_cfg = TrainerConfig(stages=stages, log_every=100, save_every=1000,
                                 checkpoint_dir="")

    with tempfile.TemporaryDirectory() as tmp:
        trainer_cfg.checkpoint_dir = tmp
        trainer = Trainer(model_cfg, trainer_cfg)

        # Advance into stage 1b
        for _ in range(3):
            trainer.train_step(_random_batch())
        assert trainer.stage_idx == 1

        path = trainer.save_checkpoint()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["stage_idx"] == 1
        assert ckpt["stage_start_step"] == 3
        assert ckpt["step"] == 3


def test_checkpoint_restores_stage_state():
    """After load_checkpoint, stage settings must be re-applied."""
    model_cfg = _tiny_config()
    stages = [make_stage_1a(3), make_stage_1b(3)]
    trainer_cfg = TrainerConfig(stages=stages, log_every=100, save_every=1000,
                                 checkpoint_dir="")

    with tempfile.TemporaryDirectory() as tmp:
        trainer_cfg.checkpoint_dir = tmp

        # Create and advance trainer to stage 1b
        trainer = Trainer(model_cfg, trainer_cfg)
        for _ in range(3):
            trainer.train_step(_random_batch())
        path = trainer.save_checkpoint()

        # Create a fresh trainer and load the checkpoint
        trainer_cfg2 = TrainerConfig(stages=stages, log_every=100, save_every=1000,
                                      checkpoint_dir=tmp)
        trainer2 = Trainer(model_cfg, trainer_cfg2)
        loaded_step = trainer2.load_checkpoint(path)

        assert loaded_step == 3
        assert trainer2.stage_idx == 1
        assert trainer2._stage_start_step == 3
        # stage 1b settings must be active
        assert trainer2.config.num_recurrent_steps == 4
        assert trainer2.config.beta_kl == pytest.approx(0.1)
        assert trainer2.model.rope.base == pytest.approx(10_000.0)


def test_resume_mid_stage_cosine():
    """Resuming into stage 2 mid-way must set correct cosine LR."""
    model_cfg = _tiny_config()
    total_stage2 = 10
    stages = [make_stage_2(total_stage2)]
    trainer_cfg = TrainerConfig(stages=stages, log_every=100, save_every=1000,
                                 checkpoint_dir="")

    with tempfile.TemporaryDirectory() as tmp:
        trainer_cfg.checkpoint_dir = tmp

        # Advance 5 steps into stage 2
        trainer = Trainer(model_cfg, trainer_cfg)
        for _ in range(5):
            trainer.train_step(_random_batch())
        path = trainer.save_checkpoint()
        lr_at_5 = _get_optimizer_lr(trainer)

        # Restore
        trainer_cfg2 = TrainerConfig(stages=stages, log_every=100, save_every=1000,
                                      checkpoint_dir=tmp)
        trainer2 = Trainer(model_cfg, trainer_cfg2)
        trainer2.load_checkpoint(path)

        restored_lr = _get_optimizer_lr(trainer2)
        expected_lr = _cosine_lr(3e-5, step=5, total_steps=total_stage2)

        assert restored_lr == pytest.approx(expected_lr, rel=1e-4)
        assert restored_lr == pytest.approx(lr_at_5, rel=1e-4)
