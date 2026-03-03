#!/usr/bin/env python
"""Entry point for LoopLM analysis experiments.

Subcommands:
    capo    — Knowledge Capacity (Section 6.1): trains 1M–40M models on
              synthetic biographies and measures bits-of-knowledge per parameter
              for loop=1 vs loop=4.

Usage:
    # Quick smoke test (tiny N, few exposures)
    uv run scripts/analyze.py capo \
        --n-individuals 200 --train-exposures 5 \
        --model-sizes micro --loop-counts 1 4 \
        --batch-size 8 --seq-len 64

    # Small-scale paper replication
    uv run scripts/analyze.py capo \
        --n-individuals 20000 --train-exposures 100 \
        --model-sizes micro mini --loop-counts 1 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Capo subcommand ───────────────────────────────────────────────────────────

def run_capo(args) -> None:
    from src.analysis.capo import CapoConfig, run_capo_experiment, print_capo_results

    config = CapoConfig(
        n_individuals=args.n_individuals,
        train_exposures=args.train_exposures,
        model_sizes=args.model_sizes,
        loop_counts=args.loop_counts,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup_steps=args.warmup_steps,
        tokenizer_id=args.tokenizer_id,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f"Capo experiment")
    print(f"  N individuals : {args.n_individuals:,}")
    print(f"  Exposures     : {args.train_exposures}")
    print(f"  Model sizes   : {args.model_sizes}")
    print(f"  Loop counts   : {args.loop_counts}")
    print(f"  Device        : {args.device}")
    print()

    results = run_capo_experiment(config)
    print_capo_results(results)

    # Optionally save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "capo_results.csv"
    with open(results_path, "w") as f:
        f.write("model_size,n_params,loop_count,n_individuals,bits_per_param,p1,p2\n")
        for r in results:
            f.write(
                f"{r.model_size},{r.n_params},{r.loop_count},{r.n_individuals},"
                f"{r.bits_per_param:.6f},{r.name_loss_nats:.6f},{r.attr_loss_nats:.6f}\n"
            )
    print(f"Results saved to {results_path}")


# ── Argument parsing ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoopLM analysis experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── capo ──────────────────────────────────────────────────────────────────
    capo = sub.add_parser("capo", help="Knowledge capacity experiment (Section 6.1)")

    capo.add_argument("--n-individuals", type=int, default=10_000,
                      help="Number of synthetic individuals in the biography dataset")
    capo.add_argument("--train-exposures", type=int, default=100,
                      help="Times each biography is seen during training (paper: 1000)")
    capo.add_argument("--model-sizes", nargs="+", default=["micro", "mini"],
                      choices=["micro", "mini", "small", "medium"],
                      help="Model size presets to benchmark")
    capo.add_argument("--loop-counts", nargs="+", type=int, default=[1, 4],
                      help="Recurrent step counts to compare")

    capo.add_argument("--lr", type=float, default=1e-3)
    capo.add_argument("--batch-size", type=int, default=192)
    capo.add_argument("--seq-len", type=int, default=512)
    capo.add_argument("--warmup-steps", type=int, default=1_000)
    capo.add_argument("--tokenizer-id", default="HuggingFaceTB/SmolLM2-135M")
    capo.add_argument("--device", default="auto")
    capo.add_argument("--seed", type=int, default=42)
    capo.add_argument("--output-dir", default="runs/capo")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "capo":
        run_capo(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
