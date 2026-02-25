import os
import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from src.model.config import LoopLMConfig
from src.training.trainer import Trainer, TrainerConfig


@pytest.fixture
def tiny_cfg():
    return LoopLMConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        max_seq_len=16,
        max_recurrent_steps=2,
    )


@pytest.fixture
def tiny_trainer_cfg(tmp_path):
    return TrainerConfig(
        lr=1e-3,
        max_steps=10,
        log_every=5,
        save_every=999,  # don't auto-save during short tests
        checkpoint_dir=str(tmp_path / "checkpoints"),
        use_wandb=False,
        device="cpu",
    )


def make_dataloader(vocab_size: int, seq_len: int = 9, n_batches: int = 20, B: int = 4, seed: int = 0):
    """Returns a DataLoader of random token batches of shape (B, seq_len)."""
    torch.manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (n_batches * B, seq_len))
    return DataLoader(TensorDataset(tokens), batch_size=B, shuffle=False)


# ── Initialization ────────────────────────────────────────────────────────────

def test_trainer_creates_model(tiny_cfg, tiny_trainer_cfg):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    from src.model.looplm import LoopLM
    assert isinstance(trainer.model, LoopLM)


def test_trainer_step_starts_at_zero(tiny_cfg, tiny_trainer_cfg):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    assert trainer.step == 0


# ── Single train step ─────────────────────────────────────────────────────────

def test_train_step_returns_diagnostics(tiny_cfg, tiny_trainer_cfg):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    batch = torch.randint(0, tiny_cfg.vocab_size, (4, 9))  # (B, S+1)
    diag = trainer.train_step(batch)
    for key in ("loss", "task_loss", "entropy", "avg_exit_step", "grad_norm"):
        assert key in diag


def test_train_step_increments_step(tiny_cfg, tiny_trainer_cfg):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    batch = torch.randint(0, tiny_cfg.vocab_size, (4, 9))
    trainer.train_step(batch)
    assert trainer.step == 1


def test_train_step_produces_finite_loss(tiny_cfg, tiny_trainer_cfg):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    batch = torch.randint(0, tiny_cfg.vocab_size, (4, 9))
    diag = trainer.train_step(batch)
    assert torch.isfinite(diag["loss"])


# ── Smoke test: loss decreases when overfitting on a fixed batch ──────────────

def test_loss_decreases_overfitting(tiny_cfg, tmp_path):
    """Train 30 steps on the same batch — loss must decrease (overfitting check)."""
    trainer_cfg = TrainerConfig(
        lr=1e-2,
        max_steps=30,
        log_every=999,
        save_every=999,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        use_wandb=False,
        device="cpu",
        num_recurrent_steps=2,
    )
    torch.manual_seed(0)
    trainer = Trainer(tiny_cfg, trainer_cfg)
    fixed_batch = torch.randint(0, tiny_cfg.vocab_size, (4, 9))

    first_loss = trainer.train_step(fixed_batch)["loss"].item()
    for _ in range(28):
        trainer.train_step(fixed_batch)
    last_loss = trainer.train_step(fixed_batch)["loss"].item()

    assert last_loss < first_loss, (
        f"Loss did not decrease: {first_loss:.4f} → {last_loss:.4f}"
    )


# ── Checkpointing ─────────────────────────────────────────────────────────────

def test_save_and_load_checkpoint(tiny_cfg, tiny_trainer_cfg, tmp_path):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    batch = torch.randint(0, tiny_cfg.vocab_size, (4, 9))
    for _ in range(3):
        trainer.train_step(batch)

    path = trainer.save_checkpoint()
    assert path.exists()

    # Create a fresh trainer and load the checkpoint
    trainer2 = Trainer(tiny_cfg, tiny_trainer_cfg)
    loaded_step = trainer2.load_checkpoint(path)

    assert loaded_step == 3
    assert trainer2.step == 3

    # Weights should match
    for (n1, p1), (n2, p2) in zip(
        trainer.model.named_parameters(), trainer2.model.named_parameters()
    ):
        torch.testing.assert_close(p1, p2, msg=f"Mismatch in {n1}")


def test_checkpoint_filename_contains_step(tiny_cfg, tiny_trainer_cfg, tmp_path):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    batch = torch.randint(0, tiny_cfg.vocab_size, (4, 9))
    trainer.train_step(batch)
    path = trainer.save_checkpoint()
    assert "step_0000001" in path.name


# ── Full training loop ────────────────────────────────────────────────────────

def test_train_loop_runs_max_steps(tiny_cfg, tiny_trainer_cfg):
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    dl = make_dataloader(tiny_cfg.vocab_size, seq_len=9, n_batches=5)
    trainer.train(dl)
    assert trainer.step == tiny_trainer_cfg.max_steps


def test_train_loop_cycles_dataloader(tiny_cfg, tiny_trainer_cfg):
    """Loop must continue past the end of the dataloader by cycling."""
    trainer = Trainer(tiny_cfg, tiny_trainer_cfg)
    # Dataloader with only 2 batches, but max_steps=10
    dl = make_dataloader(tiny_cfg.vocab_size, seq_len=9, n_batches=1, B=4)
    trainer.train(dl)  # should not raise StopIteration
    assert trainer.step == tiny_trainer_cfg.max_steps
