#!/usr/bin/env python
"""Entry point for LoopLM pre-training (Stage I).

Quick test run:
    uv run scripts/train.py --max-steps 1000 --batch-size 4 --seq-len 512

Full small-scale run (~100M, wikitext-103):
    uv run scripts/train.py --max-steps 10000 --batch-size 8 --seq-len 1024
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import torch

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.config import LoopLMConfig
from src.training.trainer import (
    Trainer,
    TrainerConfig,
    make_stage_1a,
    make_stage_1b,
    make_stage_2,
)
from src.training.data import make_dataloader


def parse_args():
    p = argparse.ArgumentParser(description="Train a LoopLM model")

    # Model
    p.add_argument("--model-config", default="small",
                   choices=["small", "ouro_1_4b", "ouro_2_6b"])

    # Data
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset-config", default="wikitext-103-v1")
    p.add_argument("--split", default="train")
    p.add_argument("--tokenizer-id", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="Cap number of training chunks (for quick runs)")

    # Training
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta-kl", type=float, default=0.1)
    p.add_argument("--num-recurrent-steps", type=int, default=None)
    p.add_argument("--device", default="auto")

    # Multi-stage training (0 = skip that stage)
    p.add_argument("--stage1a-steps", type=int, default=0,
                   help="Steps for Stage 1a (T=8, constant LR 3e-4, β=0.1). 0=skip.")
    p.add_argument("--stage1b-steps", type=int, default=0,
                   help="Steps for Stage 1b (T=4, constant LR 3e-4, β=0.1). 0=skip.")
    p.add_argument("--stage2-steps", type=int, default=0,
                   help="Steps for Stage 2 (T=4, cosine LR 3e-5, β=0.05, rope 40K). 0=skip.")

    # Logging & checkpointing
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--output-dir", default="runs/train")
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", default="looplm")
    p.add_argument("--run-name", default=None)

    # Periodic evaluation
    p.add_argument("--eval-every", type=int, default=0,
                   help="Evaluate every N steps (0 = disabled).")
    p.add_argument("--eval-tasks", default="",
                   help="Comma-separated lm-eval task names for periodic eval.")
    p.add_argument("--eval-limit", type=int, default=None,
                   help="Max examples per task during periodic eval (default: full).")

    return p.parse_args()


def build_model_config(name: str, seq_len: int) -> LoopLMConfig:
    configs = {
        "small": LoopLMConfig.small(),
        "ouro_1_4b": LoopLMConfig.ouro_1_4b(),
        "ouro_2_6b": LoopLMConfig.ouro_2_6b(),
    }
    cfg = configs[name]
    cfg.max_seq_len = seq_len
    return cfg


def print_summary(history: dict, num_recurrent_steps: int) -> None:
    """Print a summary table comparing first vs last 10% of training."""
    total = len(history["loss"])
    window = max(1, total // 10)

    first = {k: v[:window] for k, v in history.items() if k != "per_step_losses"}
    last  = {k: v[-window:] for k, v in history.items() if k != "per_step_losses"}

    def mean(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    print("\n" + "=" * 60)
    print(f"{'TRAINING SUMMARY':^60}")
    print(f"{'(first ' + str(window) + ' steps vs last ' + str(window) + ' steps)':^60}")
    print("=" * 60)
    print(f"{'Metric':<25} {'First':>10} {'Last':>10} {'Δ':>10}")
    print("-" * 60)
    for key in ("loss", "task_loss", "entropy", "avg_exit_step"):
        if key in history:
            f = mean(first[key])
            l = mean(last[key])
            print(f"  {key:<23} {f:>10.4f} {l:>10.4f} {l-f:>+10.4f}")
    print("-" * 60)

    # Per-step loss table
    first_step_losses = history["per_step_losses"][:window]
    last_step_losses  = history["per_step_losses"][-window:]
    if first_step_losses:
        print(f"\n  Per-recurrent-step losses:")
        print(f"  {'Step':<8} {'First':>10} {'Last':>10} {'Δ':>10}")
        print(f"  {'-'*42}")
        n = num_recurrent_steps
        for t in range(n):
            f_vals = [row[t].item() for row in first_step_losses if len(row) > t]
            l_vals = [row[t].item() for row in last_step_losses  if len(row) > t]
            f, l = mean(f_vals), mean(l_vals)
            print(f"  t={t+1:<6} {f:>10.4f} {l:>10.4f} {l-f:>+10.4f}")

    print("=" * 60)

    # Verification checks
    print("\nVerification:")
    checks = {
        "Total loss decreased": mean(last["loss"]) < mean(first["loss"]),
        "Task loss decreased":  mean(last["task_loss"]) < mean(first["task_loss"]),
        "Entropy not collapsed (> 0.1)": mean(last["entropy"]) > 0.1,
    }
    # Check later steps have lower loss than earlier steps
    if history["per_step_losses"]:
        recent = history["per_step_losses"][-window:]
        t1_vals = [row[0].item() for row in recent if len(row) > 0]
        tn_vals = [row[-1].item() for row in recent]
        checks[f"Step t={num_recurrent_steps} loss < step t=1 loss"] = (
            mean(tn_vals) < mean(t1_vals)
        )

    all_passed = True
    for desc, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  [{status}] {desc}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All checks passed.")
    else:
        print("Some checks failed — review training dynamics.")
    print()


def main():
    args = parse_args()

    model_cfg = build_model_config(args.model_config, args.seq_len)
    eval_tasks = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]

    # Build multi-stage list (each non-zero step value enables that stage)
    stages = []
    if args.stage1a_steps > 0:
        stages.append(make_stage_1a(args.stage1a_steps))
    if args.stage1b_steps > 0:
        stages.append(make_stage_1b(args.stage1b_steps))
    if args.stage2_steps > 0:
        stages.append(make_stage_2(args.stage2_steps))

    # When stages are provided, max_steps is ignored (total = sum of stage steps)
    effective_max_steps = args.max_steps if not stages else sum(s.max_steps for s in stages)

    trainer_cfg = TrainerConfig(
        lr=args.lr,
        max_steps=effective_max_steps,
        beta_kl=args.beta_kl,
        num_recurrent_steps=args.num_recurrent_steps,
        stages=stages,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_dir=str(Path(args.output_dir) / "checkpoints"),
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.run_name,
        device=args.device,
        eval_every=args.eval_every,
        eval_tasks=eval_tasks,
        eval_limit=args.eval_limit,
        tokenizer_id=args.tokenizer_id,
    )

    n_params = model_cfg.num_parameters()
    print(f"Model: {args.model_config}  ({n_params/1e6:.1f}M params)")
    if stages:
        stages_desc = " → ".join(f"{s.name}({s.max_steps})" for s in stages)
        print(f"Device: {trainer_cfg.device}  |  Stages: {stages_desc}  "
              f"|  Batch: {args.batch_size}  |  SeqLen: {args.seq_len}")
    else:
        print(f"Device: {trainer_cfg.device}  |  Steps: {args.max_steps}  "
              f"|  Batch: {args.batch_size}  |  SeqLen: {args.seq_len}")
    print()

    print("Loading dataset...")
    dataloader = make_dataloader(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        tokenizer_id=args.tokenizer_id,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_chunks=args.max_chunks,
    )

    trainer = Trainer(model_cfg, trainer_cfg)
    num_steps = args.num_recurrent_steps or model_cfg.max_recurrent_steps

    # ── Training loop with metric collection ──────────────────────────────────
    history: dict[str, list] = defaultdict(list)

    from src.training.trainer import _infinite
    data_iter = _infinite(dataloader)

    print(f"\nStarting training for {trainer_cfg.max_steps} steps...\n")

    while trainer.step < trainer_cfg.max_steps:
        batch = next(data_iter)
        diag = trainer.train_step(batch)

        history["loss"].append(diag["loss"].item())
        history["task_loss"].append(diag["task_loss"].item())
        history["entropy"].append(diag["entropy"].item())
        history["avg_exit_step"].append(diag["avg_exit_step"].item())
        history["per_step_losses"].append(diag["per_step_losses"])

        if trainer.step % args.log_every == 0:
            trainer._log(diag)

        if trainer.step % args.save_every == 0:
            path = trainer.save_checkpoint()
            print(f"  → Checkpoint saved: {path}")

        if args.eval_every > 0 and trainer.step % args.eval_every == 0:
            eval_results = trainer.eval_checkpoint()
            if eval_results:
                print(f"  → Eval at step {trainer.step}: {eval_results}")

    # Final checkpoint
    path = trainer.save_checkpoint()
    print(f"  → Final checkpoint: {path}")

    print_summary(history, num_steps)


if __name__ == "__main__":
    main()
