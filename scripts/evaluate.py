"""scripts/evaluate.py — Evaluate a trained LoopLM checkpoint via lm-eval-harness.

Usage:
    uv run scripts/evaluate.py \\
        --checkpoint checkpoints/step_0010000.pt \\
        --tasks arc_easy,hellaswag \\
        --limit 100 \\
        --device cuda

    # Sprint 3.3 — evaluate at T=1..max_recurrent_steps
    uv run scripts/evaluate.py \\
        --checkpoint checkpoints/step_0010000.pt \\
        --tasks arc_easy \\
        --limit 100 \\
        --eval-all-steps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

# Make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.inference.lm_eval_wrapper import LoopLMLM, run_eval, _extract_acc


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    num_steps_override: int | None = None,
) -> tuple[LoopLM, LoopLMConfig, int]:
    """Load a LoopLM checkpoint.

    Returns:
        (model, config, num_steps_to_use)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = LoopLMConfig(**ckpt["model_config"])
    model = LoopLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    num_steps = num_steps_override if num_steps_override is not None else config.max_recurrent_steps
    return model, config, num_steps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a LoopLM checkpoint with lm-eval-harness.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    p.add_argument(
        "--tasks",
        default="arc_easy,hellaswag",
        help="Comma-separated lm-eval task names (default: arc_easy,hellaswag).",
    )
    p.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of recurrent steps (default: value from checkpoint config).",
    )
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1).")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max examples per task for quick sanity checks (default: full dataset).",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device: auto|cuda|cpu (default: auto).",
    )
    p.add_argument(
        "--output-path",
        default=None,
        help="Optional path to write JSON results.",
    )
    p.add_argument(
        "--eval-all-steps",
        action="store_true",
        help="Evaluate at T=1..max_recurrent_steps and print a depth comparison table.",
    )
    return p.parse_args()


def _print_depth_table(
    table: list[tuple[int, dict[str, float | None]]],
    tasks: list[str],
) -> None:
    """Print a Markdown table of accuracy vs. recurrent depth."""
    header = "| Steps | " + " | ".join(f"{t:^10}" for t in tasks) + " |"
    sep    = "|-------|" + "|".join("-" * 12 for _ in tasks) + "|"
    print()
    print(header)
    print(sep)
    for num_steps, accs in table:
        row_vals = " | ".join(
            f"{accs[t]:^10.4f}" if accs.get(t) is not None else f"{'N/A':^10}"
            for t in tasks
        )
        print(f"| {num_steps:^5d} | {row_vals} |")
    print()


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"Loading checkpoint: {args.checkpoint}")
    model, config, default_steps = load_checkpoint(args.checkpoint, device, args.num_steps)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    if args.eval_all_steps:
        # Sprint 3.3 — evaluate at each recurrent depth
        table: list[tuple[int, dict[str, float | None]]] = []
        for T in range(1, config.max_recurrent_steps + 1):
            print(f"\nEvaluating at T={T} recurrent steps …")
            wrapper = LoopLMLM(model, tokenizer, device, num_steps=T, batch_size=args.batch_size)
            raw = run_eval(wrapper, tasks, args.limit)
            accs = {t: _extract_acc(raw, t) for t in tasks}
            for t, acc in accs.items():
                print(f"  {t}: {acc:.4f}" if acc is not None else f"  {t}: N/A")
            table.append((T, accs))

        print("\n=== Recurrent Depth vs. Accuracy ===")
        _print_depth_table(table, tasks)

        if args.output_path:
            output = {
                "checkpoint": args.checkpoint,
                "tasks": tasks,
                "limit": args.limit,
                "depth_table": [
                    {"steps": T, "accs": accs} for T, accs in table
                ],
            }
            Path(args.output_path).write_text(json.dumps(output, indent=2))
            print(f"Results written to {args.output_path}")

    else:
        # Standard single-depth evaluation
        num_steps = default_steps
        print(f"Running evaluation: tasks={tasks}, steps={num_steps}, limit={args.limit}")
        wrapper = LoopLMLM(model, tokenizer, device, num_steps=num_steps, batch_size=args.batch_size)
        raw = run_eval(wrapper, tasks, args.limit)

        print("\n=== Results ===")
        for task in tasks:
            acc = _extract_acc(raw, task)
            print(f"  {task}: {acc:.4f}" if acc is not None else f"  {task}: N/A")

        if args.output_path:
            Path(args.output_path).write_text(json.dumps(raw, indent=2))
            print(f"Results written to {args.output_path}")


if __name__ == "__main__":
    main()
