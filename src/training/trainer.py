import dataclasses
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.training.objectives import compute_adaptive_gate_loss, compute_looplm_loss, compute_q_act_loss


@dataclass
class StageConfig:
    """Configuration for a single training stage.

    Hyperparameters follow Ouro paper (arXiv:2510.25741) Table 1:

    | Stage | LR   | Schedule | T  | β    | RoPE base | Seq len |
    |-------|------|----------|----|------|-----------|---------|
    | 1a    | 3e-4 | Constant | 8  | 0.1  | 10K       | 4K      |
    | 1b    | 3e-4 | Constant | 4  | 0.1  | 10K       | 4K      |
    | 2     | 3e-5 | Cosine   | 4  | 0.05 | 40K       | 16K     |
    | 3     | 3e-5 | Constant | 4  | 0.05 | 1M        | 64K     |
    | 4     | 1e-5 | Cosine   | 4  | 0.05 | 1M        | 32K     |

    ``seq_len`` and ``grad_accum_steps`` are optional (``None`` means keep the
    trainer's global setting at startup, so existing code is unchanged).
    When set, they instruct the trainer to rebuild the DataLoader at the start
    of that stage via the ``dataloader_factory`` callback.
    """
    name: str
    max_steps: int
    lr: float = 3e-4
    lr_schedule: str = "constant"   # "constant" | "cosine"
    num_recurrent_steps: int = 4
    beta_kl: float = 0.1
    rope_base: float = 10_000.0
    # Optional per-stage overrides (None = inherit trainer default at startup)
    seq_len: int | None = None
    grad_accum_steps: int | None = None


def make_stage_1a(max_steps: int) -> StageConfig:
    """Stage 1a — T=8, constant LR 3e-4, beta=0.1, RoPE 10K (Ouro Table 1)."""
    return StageConfig("1a", max_steps, lr=3e-4, lr_schedule="constant",
                       num_recurrent_steps=8, beta_kl=0.1, rope_base=10_000.0)


def make_stage_1b(max_steps: int) -> StageConfig:
    """Stage 1b — T=4, constant LR 3e-4, beta=0.1, RoPE 10K (Ouro Table 1)."""
    return StageConfig("1b", max_steps, lr=3e-4, lr_schedule="constant",
                       num_recurrent_steps=4, beta_kl=0.1, rope_base=10_000.0)


def make_stage_2(max_steps: int) -> StageConfig:
    """Stage 2 — T=4, cosine LR 3e-5, beta=0.05, RoPE 40K (Ouro Table 1).

    Ouro uses this stage at 16K context with 1.4T tokens of reasoning-heavy
    data (OpenCoder-Annealing, MegaMath-HQ).
    """
    return StageConfig("2", max_steps, lr=3e-5, lr_schedule="cosine",
                       num_recurrent_steps=4, beta_kl=0.05, rope_base=40_000.0)


def make_stage_3(max_steps: int) -> StageConfig:
    """Stage 3 — T=4, constant LR 3e-5, beta=0.05, RoPE 1M (Ouro Table 1).

    Designed for long-context annealing at 64K tokens using ~20B tokens of
    long-document data. The 1M RoPE base dramatically extends effective context
    without destabilising the shared-weight block learned in Stages 1 and 2.
    """
    return StageConfig("3", max_steps, lr=3e-5, lr_schedule="constant",
                       num_recurrent_steps=4, beta_kl=0.05, rope_base=1_000_000.0)


def make_stage_4(max_steps: int) -> StageConfig:
    """Stage 4 — T=4, cosine LR 1e-5, beta=0.05, RoPE 1M (Ouro Table 1).

    Supervised fine-tuning (SFT) stage at 32K context with ~300B tokens of
    instruction data (OpenThoughts3, AceReason). Cosine schedule decays LR to
    10% of base over the full step budget.
    """
    return StageConfig("4", max_steps, lr=1e-5, lr_schedule="cosine",
                       num_recurrent_steps=4, beta_kl=0.05, rope_base=1_000_000.0)



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

    # Gradient accumulation
    grad_accum_steps: int = 1

    # Mixed precision (bf16)
    use_amp: bool = True

    # Q-ACT parameters (only used when model has use_q_act=True)
    q_weight: float = 0.1
    q_gamma: float = 0.99

    # Periodic evaluation during training (0 = disabled)
    eval_every: int = 0
    eval_tasks: list[str] = field(default_factory=list)
    eval_limit: int | None = None
    tokenizer_id: str = "HuggingFaceTB/SmolLM2-135M"


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    # In DDP, LOCAL_RANK tells us which GPU this process owns
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_rank0() -> bool:
    return not _is_distributed() or dist.get_rank() == 0


class Trainer:
    """Training loop for LoopLM (Stage I). Supports single-GPU and DDP.

    Expects an iterable of token-id batches: each item is a LongTensor of
    shape (B, S+1). The trainer uses tokens[:, :-1] as inputs and tokens[:, 1:]
    as next-token prediction targets.
    """

    def __init__(self, model_config: LoopLMConfig, trainer_config: TrainerConfig,
                 model_kwargs: dict | None = None,
                 dataloader_factory=None,
                 base_seq_len: int = 512):
        """Initialise the trainer.

        Args:
            model_config:        LoopLMConfig for the model architecture.
            trainer_config:      TrainerConfig with optimizer and stage settings.
            model_kwargs:        Extra kwargs forwarded to LoopLM (e.g. use_q_act).
            dataloader_factory:  Optional callable ``(seq_len, grad_accum_steps) ->
                                 DataLoader``. When provided, stage transitions that
                                 declare a new ``seq_len`` or ``grad_accum_steps``
                                 will call this to rebuild the DataLoader and store
                                 the result in ``self._current_dataloader``.
            base_seq_len:        The initial sequence length. Used to detect when
                                 a stage transition requires a DataLoader rebuild.
        """

        self.model_config = model_config
        self.config = trainer_config
        self.device = _resolve_device(trainer_config.device)
        self.rank0 = _is_rank0()

        model_kwargs = model_kwargs or {}
        self.model = LoopLM(model_config, **model_kwargs).to(self.device)

        # Wrap with DDP if distributed
        if _is_distributed():
            torch.cuda.set_device(self.device)
            self.model = DDP(self.model, device_ids=[self.device])

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=trainer_config.lr,
            betas=(trainer_config.beta1, trainer_config.beta2),
            weight_decay=trainer_config.weight_decay,
        )
        self.step = 0
        self._accum_count = 0

        # DataLoader factory + current-state tracking for stage-driven rebuilds
        self._dataloader_factory = dataloader_factory
        self._base_seq_len = base_seq_len
        self._current_seq_len = base_seq_len
        self._current_accum_steps = trainer_config.grad_accum_steps
        self._current_dataloader = None  # set by external code or stage transitions

        # Multi-stage state
        self.stage_idx: int = 0
        self._stage_start_step: int = 0
        self._stage_schedule: tuple[str, float, int] | None = None  # (schedule, base_lr, total_steps)

        if trainer_config.stages:
            self.config.max_steps = sum(s.max_steps for s in trainer_config.stages)
            self._apply_stage(trainer_config.stages[0])

        if trainer_config.use_wandb and self.rank0:
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
    def raw_model(self) -> LoopLM:
        """Return the unwrapped model (bypassing DDP wrapper)."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    @property
    def num_recurrent_steps(self) -> int:
        return self.config.num_recurrent_steps or self.model_config.max_recurrent_steps

    # ── Stage lifecycle ────────────────────────────────────────────────────────

    def _apply_stage(self, stage: StageConfig, steps_into_stage: int = 0) -> None:
        """Apply a stage's hyperparameters, optionally resuming mid-stage.

        If the stage declares a ``seq_len`` or ``grad_accum_steps`` that
        differs from the current trainer settings *and* a
        ``dataloader_factory`` was provided at construction time, the factory
        is called to build a new DataLoader and store it in
        ``self._current_dataloader``.  The training loop in ``scripts/train.py``
        can poll ``trainer._current_dataloader`` for the latest DataLoader.
        """
        # Update optimizer LR
        lr = stage.lr
        if stage.lr_schedule == "cosine" and steps_into_stage > 0:
            lr = _cosine_lr(stage.lr, steps_into_stage, stage.max_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        # Mutate config so train_step picks up new values
        self.config.num_recurrent_steps = stage.num_recurrent_steps
        self.config.beta_kl = stage.beta_kl

        # Update RoPE base — recomputes frequency caches in-place, zero overhead
        self.raw_model.rope.set_base(stage.rope_base)

        # Apply per-stage grad_accum_steps override
        if stage.grad_accum_steps is not None:
            self.config.grad_accum_steps = stage.grad_accum_steps

        # Rebuild DataLoader if seq_len or grad_accum changed and factory exists
        new_seq_len = stage.seq_len or self._base_seq_len
        new_accum = self.config.grad_accum_steps
        if (
            self._dataloader_factory is not None
            and (new_seq_len != self._current_seq_len
                 or new_accum != self._current_accum_steps)
        ):
            self._current_seq_len = new_seq_len
            self._current_accum_steps = new_accum
            self._current_dataloader = self._dataloader_factory(new_seq_len, new_accum)
            if self.rank0:
                print(
                    f"[stage {stage.name}] Rebuilt DataLoader: "
                    f"seq_len={new_seq_len}  grad_accum={new_accum}"
                )

        # Store schedule info for _update_lr
        self._stage_schedule = (stage.lr_schedule, stage.lr, stage.max_steps)

        if self.rank0:
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
        """Run one forward+backward micro-step with gradient accumulation.

        When grad_accum_steps > 1, gradients accumulate across calls.
        The optimizer steps only every grad_accum_steps calls.

        Returns:
            diagnostics dict; diag["_stepped"] indicates whether an optimizer
            step was taken (True) or gradients are still accumulating (False).
        """
        # TensorDataset wraps batches as a list/tuple — unwrap if needed
        if isinstance(token_ids, (list, tuple)):
            token_ids = token_ids[0]
        token_ids = token_ids.to(self.device)
        x = token_ids[:, :-1]        # (B, S) — inputs
        targets = token_ids[:, 1:]   # (B, S) — next-token targets

        self.model.train()
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if self.config.use_amp and self.device.type == "cuda"
            else torch.amp.autocast(self.device.type, enabled=False)
        )
        with amp_ctx:
            out = self.model(x, num_steps=self.num_recurrent_steps)
            if out.q_values is not None:
                loss, diag = compute_q_act_loss(
                    out.logits, out.q_values, out.exit_lambdas, targets,
                    beta=self.config.beta_kl,
                    q_weight=self.config.q_weight,
                    q_gamma=self.config.q_gamma,
                )
            else:
                loss, diag = compute_looplm_loss(
                    out.logits, out.exit_lambdas, targets, beta=self.config.beta_kl
                )

        scaled_loss = loss / self.config.grad_accum_steps
        scaled_loss.backward()

        self._accum_count += 1
        if self._accum_count >= self.config.grad_accum_steps:
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._accum_count = 0
            self.step += 1
            self._update_lr()
            self._maybe_advance_stage()
            diag["grad_norm"] = grad_norm.detach()
            diag["_stepped"] = True
        else:
            diag["_stepped"] = False

        return diag

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, dataloader) -> None:
        """Train until max_steps, cycling through the dataloader as needed.

        If a stage transition creates a new DataLoader (via dataloader_factory),
        it is automatically picked up mid-run.
        """
        current_dl = dataloader
        data_iter = _infinite(current_dl)
        while self.step < self.config.max_steps:
            if self._current_dataloader is not None and self._current_dataloader is not current_dl:
                current_dl = self._current_dataloader
                data_iter = _infinite(current_dl)
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
        if not self.rank0:
            return Path(self.config.checkpoint_dir) / f"step_{self.step:07d}.pt"
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"step_{self.step:07d}.pt"
        torch.save(
            {
                "step": self.step,
                "model_state_dict": self.raw_model.state_dict(),
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
        self.raw_model.load_state_dict(ckpt["model_state_dict"])
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
        Only runs on rank 0 in distributed training.
        """
        if not self.rank0 or not self.config.eval_tasks:
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
        eval_model = self.raw_model

        eval_model.eval()
        metrics: dict[str, float] = {}

        print(f"\n[eval] step={self.step}  tasks={tasks}  T=1..{max_T}")
        header = f"  {'T':>3} | " + " | ".join(f"{t:^12}" for t in tasks)
        print(header)
        print("  " + "-" * (len(header) - 2))

        for T in range(1, max_T + 1):
            wrapper = LoopLMLM(
                eval_model, self._tokenizer, self.device, num_steps=T, batch_size=1
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
        if not self.rank0:
            return
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


# ── Stage II: Gate fine-tuning ────────────────────────────────────────────────

@dataclass
class GateFinetuneConfig:
    """Config for Stage II: fine-tune the exit gate with LM params frozen."""
    lr: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    max_steps: int = 10_000
    num_recurrent_steps: int = 4
    k: float = 50.0        # sigmoid sharpness for ideal-continuation label
    gamma: float = 0.005   # improvement threshold

    log_every: int = 10
    save_every: int = 1_000
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"

    use_wandb: bool = False
    wandb_project: str = "looplm"
    wandb_run_name: str | None = None


class GateFinetuner:
    """Stage II: fine-tune exit gate with LM parameters frozen.

    Loads a pre-trained LoopLM checkpoint, freezes all LM parameters, then
    trains only the exit gate using compute_adaptive_gate_loss.
    """

    def __init__(self, model_config: LoopLMConfig, config: GateFinetuneConfig):
        self.model_config = model_config
        self.config = config
        self.device = _resolve_device(config.device)

        self.model = LoopLM(model_config).to(self.device)
        self._freeze_lm_params()

        # Optimizer only over exit gate parameters
        self.optimizer = torch.optim.AdamW(
            self.model.exit_gate.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
        self.step = 0

        if config.use_wandb:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config={
                    "model": dataclasses.asdict(model_config),
                    "gate_finetune": dataclasses.asdict(config),
                },
            )

    def _freeze_lm_params(self) -> None:
        """Freeze all parameters except the exit gate."""
        for name, param in self.model.named_parameters():
            if "exit_gate" not in name:
                param.requires_grad_(False)

    # ── Checkpointing ──────────────────────────────────────────────────────────

    def load_checkpoint(self, path: str | Path) -> int:
        """Load model weights from a Stage I (or gate) checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        # Re-freeze after load_state_dict (doesn't touch requires_grad)
        self._freeze_lm_params()
        loaded_step = ckpt.get("step", 0)
        print(f"Loaded checkpoint from step {loaded_step}: {path}")
        return loaded_step

    def save_checkpoint(self) -> Path:
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"gate_step_{self.step:07d}.pt"
        torch.save(
            {
                "step": self.step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_config": dataclasses.asdict(self.model_config),
                "gate_config": dataclasses.asdict(self.config),
            },
            path,
        )
        return path

    # ── Core training step ─────────────────────────────────────────────────────

    def train_step(self, token_ids: Tensor) -> dict:
        if isinstance(token_ids, (list, tuple)):
            token_ids = token_ids[0]
        token_ids = token_ids.to(self.device)
        x = token_ids[:, :-1]
        targets = token_ids[:, 1:]

        self.model.train()
        out = self.model(x, num_steps=self.config.num_recurrent_steps)
        loss, diag = compute_adaptive_gate_loss(
            out.logits, out.exit_lambdas, targets,
            k=self.config.k, gamma=self.config.gamma,
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.exit_gate.parameters(), self.config.grad_clip
        )
        self.optimizer.step()
        self.step += 1
        return diag

    # ── Evaluation ────────────────────────────────────────────────────────────

    def eval_avg_exit_step(
        self,
        dataloader,
        n_batches: int = 50,
        q_thresholds: tuple[float, ...] = (0.5, 0.7, 0.9),
    ) -> dict[str, float]:
        """Compute average exit step at various Q-exit thresholds.

        Returns a dict like {"avg_exit_q0.5": 1.3, "avg_exit_q0.9": 3.1}.
        Lower is better — the gate sends more sequences to earlier exits.
        """
        from src.inference.early_exit import QExit, run_with_early_exit

        self.model.eval()
        results: dict[str, float] = {}

        for q in q_thresholds:
            total_exit_steps = 0.0
            total_sequences = 0

            data_iter = iter(dataloader)
            for _ in range(n_batches):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.device)
                x = batch[:, :-1]
                B = x.shape[0]

                with torch.no_grad():
                    result = run_with_early_exit(
                        self.model, x, QExit(q),
                        max_steps=self.config.num_recurrent_steps,
                    )
                exit_val = result.mean_exit_step if result.mean_exit_step is not None else float(result.exit_step)
                total_exit_steps += exit_val * B
                total_sequences += B

            if total_sequences:
                results[f"avg_exit_q{q}"] = total_exit_steps / total_sequences

        self.model.train()
        return results

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, dataloader, val_dataloader=None) -> None:
        """Fine-tune the gate until max_steps, with optional before/after eval."""
        if val_dataloader is not None:
            print("=== Early exit efficiency BEFORE gate fine-tuning ===")
            before = self.eval_avg_exit_step(val_dataloader)
            for key, val in before.items():
                print(f"  {key}: {val:.3f}")
            print()

        data_iter = _infinite(dataloader)
        while self.step < self.config.max_steps:
            batch = next(data_iter)
            diag = self.train_step(batch)

            if self.step % self.config.log_every == 0:
                self._log(diag)

            if self.step % self.config.save_every == 0:
                self.save_checkpoint()

        self.save_checkpoint()

        if val_dataloader is not None:
            print("\n=== Early exit efficiency AFTER gate fine-tuning ===")
            after = self.eval_avg_exit_step(val_dataloader)
            for key, val in after.items():
                print(f"  {key}: {val:.3f}")
            print()

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, diag: dict) -> None:
        bce_str = "  ".join(
            f"t{i+2}:{v.item():.3f}" for i, v in enumerate(diag.get("per_step_bce", []))
        )
        w_str = "  ".join(
            f"t{i+2}:{v:.3f}" for i, v in enumerate(diag.get("mean_w_per_step", []))
        )
        print(
            f"gate step {self.step:6d} | "
            f"loss {diag['loss'].item():.4f} | "
            f"BCE: {bce_str} | "
            f"w: {w_str}"
        )

        if self.config.use_wandb:
            import wandb
            log_dict = {"gate_loss": diag["loss"].item()}
            for i, v in enumerate(diag.get("per_step_bce", [])):
                log_dict[f"gate_bce_t{i+2}"] = v.item()
            for i, v in enumerate(diag.get("mean_w_per_step", [])):
                log_dict[f"gate_w_t{i+2}"] = v
            wandb.log(log_dict, step=self.step)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infinite(dataloader):
    """Cycle through a dataloader indefinitely."""
    while True:
        yield from dataloader
