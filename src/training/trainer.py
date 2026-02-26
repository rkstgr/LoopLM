import dataclasses
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
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        """Load checkpoint and return the step it was saved at."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.step = ckpt["step"]
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
