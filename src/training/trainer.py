import dataclasses
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.training.objectives import compute_looplm_loss


@dataclass
class StageConfig:
    """Configuration for a single training stage."""
    name: str
    max_steps: int
    lr: float = 3e-4
    lr_schedule: str = "constant"   # "constant" | "cosine"
    num_recurrent_steps: int = 4
    beta_kl: float = 0.1
    rope_base: float = 10_000.0


def make_stage_1a(max_steps: int) -> StageConfig:
    return StageConfig("1a", max_steps, lr=3e-4, lr_schedule="constant",
                       num_recurrent_steps=8, beta_kl=0.1, rope_base=10_000.0)


def make_stage_1b(max_steps: int) -> StageConfig:
    return StageConfig("1b", max_steps, lr=3e-4, lr_schedule="constant",
                       num_recurrent_steps=4, beta_kl=0.1, rope_base=10_000.0)


def make_stage_2(max_steps: int) -> StageConfig:
    return StageConfig("2", max_steps, lr=3e-5, lr_schedule="cosine",
                       num_recurrent_steps=4, beta_kl=0.05, rope_base=40_000.0)


def _cosine_lr(base_lr: float, step: int, total_steps: int,
               min_ratio: float = 0.1) -> float:
    """Cosine LR schedule with minimum floor at base_lr * min_ratio."""
    min_lr = base_lr * min_ratio
    if step >= total_steps:
        return min_lr
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * step / total_steps))


@dataclass
class TrainerConfig:
    # Optimizer (paper Table 1)
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Training
    max_steps: int = 10_000
    beta_kl: float = 0.1
    # None → use model_config.max_recurrent_steps
    num_recurrent_steps: int | None = None

    # Multi-stage training: if non-empty, max_steps is overridden by sum of stage steps
    stages: list[StageConfig] = field(default_factory=list)

    # Logging
    log_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "looplm"
    wandb_run_name: str | None = None

    # Checkpointing
    save_every: int = 1_000
    checkpoint_dir: str = "checkpoints"

    # Device — "auto" picks cuda > mps > cpu
    device: str = "auto"

    # Periodic evaluation during training (0 = disabled)
    eval_every: int = 0
    eval_tasks: list[str] = field(default_factory=list)
    eval_limit: int | None = None
    tokenizer_id: str = "HuggingFaceTB/SmolLM2-135M"


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Single-GPU training loop for LoopLM (Stage I).

    Expects an iterable of token-id batches: each item is a LongTensor of
    shape (B, S+1). The trainer uses tokens[:, :-1] as inputs and tokens[:, 1:]
    as next-token prediction targets.
    """

    def __init__(self, model_config: LoopLMConfig, trainer_config: TrainerConfig):
        self.model_config = model_config
        self.config = trainer_config
        self.device = _resolve_device(trainer_config.device)

        self.model = LoopLM(model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=trainer_config.lr,
            betas=(trainer_config.beta1, trainer_config.beta2),
            weight_decay=trainer_config.weight_decay,
        )
        self.step = 0

        # Multi-stage state
        self.stage_idx: int = 0
        self._stage_start_step: int = 0
        self._stage_schedule: tuple[str, float, int] | None = None  # (schedule, base_lr, total_steps)

        if trainer_config.stages:
            self.config.max_steps = sum(s.max_steps for s in trainer_config.stages)
            self._apply_stage(trainer_config.stages[0])

        if trainer_config.use_wandb:
            import wandb
            wandb.init(
                project=trainer_config.wandb_project,
                name=trainer_config.wandb_run_name,
                config={
                    "model": dataclasses.asdict(model_config),
                    "trainer": dataclasses.asdict(trainer_config),
                },
            )

    @property
    def num_recurrent_steps(self) -> int:
        return self.config.num_recurrent_steps or self.model_config.max_recurrent_steps

    # ── Stage lifecycle ────────────────────────────────────────────────────────

    def _apply_stage(self, stage: StageConfig, steps_into_stage: int = 0) -> None:
        """Apply a stage's hyperparameters, optionally resuming mid-stage."""
        # Update optimizer LR
        lr = stage.lr
        if stage.lr_schedule == "cosine" and steps_into_stage > 0:
            lr = _cosine_lr(stage.lr, steps_into_stage, stage.max_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        # Mutate config so train_step picks up new values
        self.config.num_recurrent_steps = stage.num_recurrent_steps
        self.config.beta_kl = stage.beta_kl

        # Update RoPE base
        self.model.rope.set_base(stage.rope_base)

        # Store schedule info for _update_lr
        self._stage_schedule = (stage.lr_schedule, stage.lr, stage.max_steps)

        print(
            f"[stage {stage.name}] T={stage.num_recurrent_steps}  β={stage.beta_kl}  "
            f"lr={lr:.2e}  rope_base={stage.rope_base:.0f}  "
            f"schedule={stage.lr_schedule}  steps={stage.max_steps}"
        )

    def _update_lr(self) -> None:
        """Update LR according to current stage schedule (no-op for constant)."""
        if self._stage_schedule is None:
            return
        schedule, base_lr, total_steps = self._stage_schedule
        if schedule != "cosine":
            return
        steps_in_stage = self.step - self._stage_start_step
        lr = _cosine_lr(base_lr, steps_in_stage, total_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _maybe_advance_stage(self) -> None:
        """Advance to the next stage if the current stage's step budget is exhausted."""
        if not self.config.stages:
            return
        current_stage = self.config.stages[self.stage_idx]
        steps_in_stage = self.step - self._stage_start_step
        if steps_in_stage >= current_stage.max_steps:
            next_idx = self.stage_idx + 1
            if next_idx < len(self.config.stages):
                self.stage_idx = next_idx
                self._stage_start_step = self.step
                self._apply_stage(self.config.stages[next_idx])

    # ── Core training step ────────────────────────────────────────────────────

    def train_step(self, token_ids: Tensor) -> dict:
        """Run one gradient update.

        Args:
            token_ids: (B, S+1) — raw token ids; split internally into input/target

        Returns:
            diagnostics dict from compute_looplm_loss
        """
        # TensorDataset wraps batches as a list/tuple — unwrap if needed
        if isinstance(token_ids, (list, tuple)):
            token_ids = token_ids[0]
        token_ids = token_ids.to(self.device)
        x = token_ids[:, :-1]        # (B, S) — inputs
        targets = token_ids[:, 1:]   # (B, S) — next-token targets

        self.model.train()
        out = self.model(x, num_steps=self.num_recurrent_steps)
        loss, diag = compute_looplm_loss(
            out.logits, out.exit_lambdas, targets, beta=self.config.beta_kl
        )

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )
        self.optimizer.step()
        self.step += 1
        self._update_lr()
        self._maybe_advance_stage()

        diag["grad_norm"] = grad_norm.detach()
        return diag

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, dataloader) -> None:
        """Train until max_steps, cycling through the dataloader as needed."""
        data_iter = _infinite(dataloader)
        while self.step < self.config.max_steps:
            batch = next(data_iter)
            diag = self.train_step(batch)

            if self.step % self.config.log_every == 0:
                self._log(diag)

            if self.step % self.config.save_every == 0:
                self.save_checkpoint()

            if self.config.eval_every > 0 and self.step % self.config.eval_every == 0:
                self.eval_checkpoint()

        # Final checkpoint
        self.save_checkpoint()

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self) -> Path:
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"step_{self.step:07d}.pt"
        torch.save(
            {
                "step": self.step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_config": dataclasses.asdict(self.model_config),
                "trainer_config": dataclasses.asdict(self.config),
                "stage_idx": self.stage_idx,
                "stage_start_step": self._stage_start_step,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        """Load checkpoint and return the step it was saved at."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.step = ckpt["step"]
        self.stage_idx = ckpt.get("stage_idx", 0)
        self._stage_start_step = ckpt.get("stage_start_step", 0)
        if self.config.stages:
            steps_into_stage = self.step - self._stage_start_step
            self._apply_stage(
                self.config.stages[self.stage_idx],
                steps_into_stage=steps_into_stage,
            )
        return self.step

    # ── Periodic evaluation ───────────────────────────────────────────────────

    def eval_checkpoint(self) -> dict[str, float]:
        """Evaluate at T=1..max_recurrent_steps; log to wandb; return metrics dict.

        Returns an empty dict if eval_tasks is empty or lm-eval is not installed.
        """
        if not self.config.eval_tasks:
            return {}

        try:
            from src.inference.lm_eval_wrapper import LoopLMLM, run_eval, _extract_acc
            from transformers import AutoTokenizer
        except ImportError:
            return {}

        # Lazy-load and cache tokenizer
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_id)

        tasks = self.config.eval_tasks
        max_T = self.model_config.max_recurrent_steps

        self.model.eval()
        metrics: dict[str, float] = {}

        print(f"\n[eval] step={self.step}  tasks={tasks}  T=1..{max_T}")
        header = f"  {'T':>3} | " + " | ".join(f"{t:^12}" for t in tasks)
        print(header)
        print("  " + "-" * (len(header) - 2))

        for T in range(1, max_T + 1):
            wrapper = LoopLMLM(
                self.model, self._tokenizer, self.device, num_steps=T, batch_size=1
            )
            raw = run_eval(wrapper, tasks, self.config.eval_limit)
            row_parts = []
            for task in tasks:
                acc = _extract_acc(raw, task)
                if acc is not None:
                    metrics[f"eval/{task}/T{T}"] = acc
                    row_parts.append(f"{acc:^12.4f}")
                else:
                    row_parts.append(f"{'N/A':^12}")
            print(f"  {T:>3} | " + " | ".join(row_parts))

        print()
        self.model.train()

        if self.config.use_wandb and metrics:
            import wandb
            wandb.log(metrics, step=self.step)

        return metrics

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, diag: dict) -> None:
        per_step = diag.get("per_step_losses", [])
        step_loss_str = "  ".join(
            f"t{i+1}:{v.item():.3f}" for i, v in enumerate(per_step)
        )
        print(
            f"step {self.step:6d} | "
            f"loss {diag['loss'].item():.4f} | "
            f"task {diag['task_loss'].item():.4f} | "
            f"entropy {diag['entropy'].item():.4f} | "
            f"avg_exit {diag['avg_exit_step'].item():.2f} | "
            f"grad_norm {diag['grad_norm'].item():.3f} | "
            f"{step_loss_str}"
        )

        if self.config.use_wandb:
            import wandb
            log_dict = {
                "loss": diag["loss"].item(),
                "task_loss": diag["task_loss"].item(),
                "entropy": diag["entropy"].item(),
                "avg_exit_step": diag["avg_exit_step"].item(),
                "grad_norm": diag["grad_norm"].item(),
            }
            for i, v in enumerate(per_step):
                log_dict[f"loss_step_{i+1}"] = v.item()
            wandb.log(log_dict, step=self.step)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infinite(dataloader):
    """Cycle through a dataloader indefinitely."""
    while True:
        yield from dataloader
