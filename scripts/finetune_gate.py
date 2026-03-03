#!/usr/bin/env python
"""Entry point for Stage II gate fine-tuning.

Loads a pre-trained LoopLM checkpoint (Stage I), freezes all LM parameters,
and trains only the exit gate using the adaptive gate loss (paper Eq. 6).

Quick test run:
    uv run scripts/finetune_gate.py \
        --checkpoint runs/train/checkpoints/step_0001000.pt \
        --max-steps 200 --batch-size 4 --seq-len 64 --log-every 10

Full run:
    uv run scripts/finetune_gate.py \
        --checkpoint runs/stage1b/checkpoints/step_final.pt \
        --max-steps 5000 --batch-size 8 --seq-len 512 \
        --use-wandb --run-name gate-finetune-v1
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.config import LoopLMConfig
from src.training.data import make_dataloader
from src.training.trainer import GateFinetuneConfig, GateFinetuner


def parse_args():
    p = argparse.ArgumentParser(description="Stage II: fine-tune the LoopLM exit gate")

    # Required
    p.add_argument("--checkpoint", required=True,
                   help="Path to a Stage I checkpoint to load")

    # Model
    p.add_argument("--model-config", default="small",
                   choices=["small", "ouro_1_4b", "ouro_2_6b"])

    # Data
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset-config", default="wikitext-103-v1")
    p.add_argument("--split", default="train")
    p.add_argument("--val-split", default="validation",
                   help="Dataset split used for before/after efficiency eval")
    p.add_argument("--tokenizer-id", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--max-chunks", type=int, default=None)
    p.add_argument("--val-batches", type=int, default=50,
                   help="Number of batches to use for early-exit efficiency eval")

    # Training
    p.add_argument("--max-steps", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--num-recurrent-steps", type=int, default=4)
    p.add_argument("--k", type=float, default=50.0,
                   help="Sigmoid sharpness for ideal-continuation label")
    p.add_argument("--gamma", type=float, default=0.005,
                   help="Improvement threshold below which gate should exit")
    p.add_argument("--device", default="auto")

    # Logging & checkpointing
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--output-dir", default="runs/gate_finetune")
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", default="looplm")
    p.add_argument("--run-name", default=None)

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


def main():
    args = parse_args()

    model_cfg = build_model_config(args.model_config, args.seq_len)
    gate_cfg = GateFinetuneConfig(
        lr=args.lr,
        max_steps=args.max_steps,
        num_recurrent_steps=args.num_recurrent_steps,
        k=args.k,
        gamma=args.gamma,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_dir=str(Path(args.output_dir) / "checkpoints"),
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.run_name,
        device=args.device,
    )

    n_params = model_cfg.num_parameters()
    print(f"Model: {args.model_config}  ({n_params/1e6:.1f}M params)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {gate_cfg.device}  |  Steps: {args.max_steps}  "
          f"|  Batch: {args.batch_size}  |  SeqLen: {args.seq_len}")
    print()

    print("Loading training data...")
    train_dl = make_dataloader(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        tokenizer_id=args.tokenizer_id,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_chunks=args.max_chunks,
    )

    # Validation dataloader for before/after efficiency eval
    val_dl = None
    try:
        val_dl = make_dataloader(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split=args.val_split,
            tokenizer_id=args.tokenizer_id,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )
        print(f"Validation split: {args.val_split}  ({args.val_batches} batches for eval)")
    except Exception:
        print(f"Note: validation split '{args.val_split}' not available — skipping efficiency eval")

    finetuner = GateFinetuner(model_cfg, gate_cfg)
    finetuner.load_checkpoint(args.checkpoint)

    # Sanity-check: confirm LM params are frozen
    n_trainable = sum(p.numel() for p in finetuner.model.parameters() if p.requires_grad)
    n_gate = sum(p.numel() for p in finetuner.model.exit_gate.parameters())
    print(f"Trainable params: {n_trainable:,}  (gate only: {n_gate:,})")
    assert n_trainable == n_gate, (
        f"Expected only gate params to be trainable ({n_gate}), "
        f"got {n_trainable} total trainable"
    )
    print()

    finetuner.train(train_dl, val_dataloader=val_dl)


if __name__ == "__main__":
    main()
