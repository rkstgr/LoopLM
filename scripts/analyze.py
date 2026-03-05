#!/usr/bin/env python
"""Entry point for LoopLM analysis experiments.

Subcommands:
    capo    — Knowledge Capacity (Section 6.1): trains 1M–40M models on
              synthetic biographies and measures bits-of-knowledge per parameter
              for loop=1 vs loop=4.

Usage:
    # Quick smoke test (tiny N, few exposures)
    uv run scripts/analyze.py capo \
        --n-individuals 200 --train-exposures 50 \
        --model-sizes micro --loop-counts 1 4 \
        --batch-size 8 --seq-len 64

    # Paper-scale replication (N=20K, 1000 exposures, 3 seeds)
    uv run scripts/analyze.py capo \
        --n-individuals 20000 --train-exposures 1000 \
        --model-sizes micro mini small --loop-counts 1 4 \
        --num-seeds 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Capo subcommand ───────────────────────────────────────────────────────────


def run_capo(args) -> None:
    import dataclasses
    import statistics

    from src.analysis.capo import (
        CapoConfig,
        CapoResult,
        print_capo_results,
        run_capo_experiment,
    )

    print(f"Capo experiment")
    print(f"  N individuals : {args.n_individuals:,}")
    print(f"  Exposures     : {args.train_exposures}")
    print(f"  Model sizes   : {args.model_sizes}")
    print(f"  Loop counts   : {args.loop_counts}")
    print(f"  Seeds         : {args.num_seeds}")
    print(f"  Device        : {args.device}")
    print()

    all_results: list[list[CapoResult]] = []
    for seed_idx in range(args.num_seeds):
        seed = args.seed + seed_idx
        print(f"--- Seed {seed_idx + 1}/{args.num_seeds} (seed={seed}) ---")
        config = CapoConfig(
            n_individuals=args.n_individuals,
            train_exposures=args.train_exposures,
            model_sizes=args.model_sizes,
            loop_counts=args.loop_counts,
            lr=args.lr,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            warmup_steps=args.warmup_steps,
            log_every=args.log_every,
            tokenizer_id=args.tokenizer_id,
            device=args.device,
            seed=seed,
            output_dir=args.output_dir,
        )
        results = run_capo_experiment(config)
        all_results.append(results)

    # Aggregate across seeds
    if args.num_seeds == 1:
        final_results = all_results[0]
        print_capo_results(final_results)
    else:
        print("\n" + "=" * 75)
        print(f"{'CAPO RESULTS — Mean ± Std across seeds':^75}")
        print("=" * 75)
        # Group by (model_size, loop_count)
        n_runs = len(all_results[0])
        print(
            f"  {'Size':<8} {'Params':>8} {'Loop':>5} {'N':>8} {'bits/param':>16} {'p1':>10} {'p2':>10}"
        )
        print("  " + "-" * 67)
        for i in range(n_runs):
            r0 = all_results[0][i]
            bpp_vals = [all_results[s][i].bits_per_param for s in range(args.num_seeds)]
            p1_vals = [all_results[s][i].name_loss_nats for s in range(args.num_seeds)]
            p2_vals = [all_results[s][i].attr_loss_nats for s in range(args.num_seeds)]
            bpp_mean = statistics.mean(bpp_vals)
            bpp_std = statistics.stdev(bpp_vals) if args.num_seeds > 1 else 0.0
            p1_mean = statistics.mean(p1_vals)
            p2_mean = statistics.mean(p2_vals)
            print(
                f"  {r0.model_size:<8} {r0.n_params / 1e6:>6.1f}M {r0.loop_count:>5} "
                f"{r0.n_individuals:>8,} {bpp_mean:>8.4f}±{bpp_std:.4f} "
                f"{p1_mean:>10.3f} {p2_mean:>10.3f}"
            )
        print("=" * 75)
        print("Expected: bits/param ≈ 2.0 for both loop=1 and loop=4")
        print()
        final_results = all_results[0]  # use first seed for CSV

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "capo_results.csv"
    with open(results_path, "w") as f:
        f.write("model_size,n_params,loop_count,n_individuals,bits_per_param,p1,p2\n")
        for r in final_results:
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

    capo.add_argument(
        "--n-individuals",
        type=int,
        default=20_000,
        help="Number of synthetic individuals (paper: 20K–500K)",
    )
    capo.add_argument(
        "--train-exposures",
        type=int,
        default=1_000,
        help="Times each biography is seen during training (paper: 1000)",
    )
    capo.add_argument(
        "--model-sizes",
        nargs="+",
        default=["micro"],
        choices=["micro", "mini", "small", "medium"],
        help="Model size presets to benchmark",
    )
    capo.add_argument(
        "--loop-counts",
        nargs="+",
        type=int,
        default=[1, 4],
        help="Recurrent step counts to compare",
    )
    capo.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of random seeds to average over (paper uses ≥2 for robustness)",
    )

    capo.add_argument("--lr", type=float, default=1e-3)
    capo.add_argument("--batch-size", type=int, default=192)
    capo.add_argument("--seq-len", type=int, default=512)
    capo.add_argument("--warmup-steps", type=int, default=1_000)
    capo.add_argument("--log-every", type=int, default=100, help="Print progress every N steps")
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
