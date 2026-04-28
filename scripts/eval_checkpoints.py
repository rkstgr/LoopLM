#!/usr/bin/env python
"""Evaluate LoopLM checkpoints on lm-eval benchmarks.

Usage:
    uv run scripts/eval_checkpoints.py \
        --checkpoint /path/to/step_0008000.pt \
        --tasks hellaswag,arc_easy \
        --limit 500
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.inference.lm_eval_wrapper import LoopLMLM, run_eval, _extract_acc
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description="Evaluate LoopLM checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--tasks", default="hellaswag,arc_easy")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tokenizer-id", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model_cfg = LoopLMConfig(**ckpt["model_config"])
    print(f"Model: L={model_cfg.num_layers}, T={model_cfg.max_recurrent_steps}, "
          f"h={model_cfg.hidden_size} ({model_cfg.num_parameters()/1e6:.1f}M params)")

    # Detect if model was trained with Q-ACT by checking for q_head in state dict
    state_dict = ckpt["model_state_dict"]
    use_q_act = any("q_head" in k for k in state_dict)

    model = LoopLM(model_cfg, use_q_act=use_q_act).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    max_T = model_cfg.max_recurrent_steps

    print(f"\nEvaluating tasks={tasks}, T=1..{max_T}, limit={args.limit}")
    print(f"{'T':>3} | " + " | ".join(f"{t:^14}" for t in tasks))
    print("-" * (6 + 17 * len(tasks)))

    all_results = {}
    for T in range(1, max_T + 1):
        wrapper = LoopLMLM(model, tokenizer, device, num_steps=T, batch_size=args.batch_size)
        raw = run_eval(wrapper, tasks, args.limit)
        row_parts = []
        for task in tasks:
            acc = _extract_acc(raw, task)
            if acc is not None:
                all_results[f"{task}/T{T}"] = acc
                row_parts.append(f"{acc:^14.4f}")
            else:
                row_parts.append(f"{'N/A':^14}")
        print(f"{T:>3} | " + " | ".join(row_parts))

    # Save results
    out_path = Path(args.checkpoint).parent.parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
