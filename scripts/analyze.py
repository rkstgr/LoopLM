#!/usr/bin/env python
"""Entry point for LoopLM analysis experiments.

Subcommands:
    capo         — Knowledge Capacity (Section 6.1): trains 1M–40M models on
                   synthetic biographies and measures bits-of-knowledge per parameter
                   for loop=1 vs loop=4.
    mano         — Knowledge Manipulation (Section 6.2): trains models on modular
                   arithmetic tree expressions (mod 23) and compares base vs. looped
                   models at iso-FLOP budgets.
    mano-collect — Aggregate Mano results from a SLURM job array. Reads per-task
                   CSVs and prints a combined table with mean ± std across seeds.

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

    # Mano quick test
    uv run scripts/analyze.py mano \
        --max-ops 5 --n-train 10000 --train-steps 500 \
        --model-preset tiny --model-configs 4:1 2:2 1:4 \
        --batch-size 16 --seq-len 64
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
    print(f"  Accumulation  : {args.accumulation_steps}")
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
            accumulation_steps=args.accumulation_steps,
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
    capo.add_argument("--batch-size", type=int, default=192, help="Micro-batch size per forward pass")
    capo.add_argument("--accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients")
    capo.add_argument("--seq-len", type=int, default=512)
    capo.add_argument("--warmup-steps", type=int, default=1_000)
    capo.add_argument("--log-every", type=int, default=100, help="Print progress every N steps")
    capo.add_argument("--tokenizer-id", default="HuggingFaceTB/SmolLM2-135M")
    capo.add_argument("--device", default="auto")
    capo.add_argument("--seed", type=int, default=42)
    capo.add_argument("--output-dir", default="runs/capo")

    # ── mano ──────────────────────────────────────────────────────────────────
    mano = sub.add_parser("mano", help="Knowledge manipulation experiment (Section 6.2)")

    mano.add_argument(
        "--max-ops",
        type=int,
        default=10,
        help="Maximum expression tree depth (paper: 10, 16, 24)",
    )
    mano.add_argument(
        "--n-train",
        type=int,
        default=500_000,
        help="Number of training examples to generate",
    )
    mano.add_argument(
        "--n-eval",
        type=int,
        default=1_000,
        help="Number of evaluation examples (hardest difficulty)",
    )
    mano.add_argument(
        "--model-configs",
        nargs="+",
        default=["4:1", "2:2", "1:4"],
        help="Model configs as layers:loops pairs (e.g., 4:1 2:2 1:4)",
    )
    mano.add_argument(
        "--model-preset",
        default="small",
        choices=["tiny", "small", "medium", "paper"],
        help="Model size preset (hidden dim / num heads)",
    )
    mano.add_argument("--num-seeds", type=int, default=1)

    mano.add_argument("--lr", type=float, default=2e-4)
    mano.add_argument("--weight-decay", type=float, default=0.1)
    mano.add_argument("--batch-size", type=int, default=128)
    mano.add_argument("--accumulation-steps", type=int, default=1)
    mano.add_argument("--seq-len", type=int, default=1024)
    mano.add_argument("--warmup-steps", type=int, default=1_000)
    mano.add_argument("--train-steps", type=int, default=80_000)
    mano.add_argument("--beta-kl", type=float, default=0.1)
    mano.add_argument("--log-every", type=int, default=500)
    mano.add_argument("--device", default="auto")
    mano.add_argument("--seed", type=int, default=42)
    mano.add_argument("--output-dir", default="runs/mano")
    mano.add_argument("--use-wandb", action="store_true", help="Log to wandb")
    mano.add_argument("--wandb-project", default="looplm")
    mano.add_argument("--wandb-run-name", default=None, help="wandb run name (auto-generated if omitted)")

    # ── mano-collect ─────────────────────────────────────────────────────────
    collect = sub.add_parser(
        "mano-collect",
        help="Aggregate Mano results from job array output directory",
    )
    collect.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing task_*/mano_results.csv files",
    )

    return parser


# ── Mano subcommand ──────────────────────────────────────────────────────────


def _parse_model_configs(config_strs: list[str]) -> list[tuple[int, int]]:
    """Parse 'layers:loops' strings into (num_layers, loop_count) tuples."""
    configs = []
    for s in config_strs:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid model config '{s}', expected 'layers:loops'")
        configs.append((int(parts[0]), int(parts[1])))
    return configs


def run_mano(args) -> None:
    import statistics

    from src.analysis.mano import (
        ManoConfig,
        ManoResult,
        print_mano_results,
        run_mano_experiment,
    )

    model_configs = _parse_model_configs(args.model_configs)

    print(f"Mano experiment")
    print(f"  Max ops       : {args.max_ops}")
    print(f"  Train examples: {args.n_train:,}")
    print(f"  Eval examples : {args.n_eval:,}")
    print(f"  Model configs : {model_configs}")
    print(f"  Model preset  : {args.model_preset}")
    print(f"  Accumulation  : {args.accumulation_steps}")
    print(f"  Train steps   : {args.train_steps:,}")
    print(f"  Seeds         : {args.num_seeds}")
    print(f"  Device        : {args.device}")
    print()

    all_results: list[list[ManoResult]] = []
    for seed_idx in range(args.num_seeds):
        seed = args.seed + seed_idx
        print(f"--- Seed {seed_idx + 1}/{args.num_seeds} (seed={seed}) ---")
        config = ManoConfig(
            max_ops=args.max_ops,
            n_train_examples=args.n_train,
            n_eval_examples=args.n_eval,
            model_configs=model_configs,
            model_preset=args.model_preset,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            seq_len=args.seq_len,
            warmup_steps=args.warmup_steps,
            train_steps=args.train_steps,
            beta_kl=args.beta_kl,
            log_every=args.log_every,
            device=args.device,
            seed=seed,
            output_dir=args.output_dir,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )
        results = run_mano_experiment(config)
        all_results.append(results)

    # Aggregate across seeds
    if args.num_seeds == 1:
        final_results = all_results[0]
        print_mano_results(final_results)
    else:
        print("\n" + "=" * 80)
        print(f"{'MANO RESULTS — Mean ± Std across seeds':^80}")
        print("=" * 80)
        n_runs = len(all_results[0])
        print(
            f"  {'Layers':>6} {'Loop':>5} {'Depth':>6} {'Params':>10} "
            f"{'max_ops':>8} {'Accuracy':>16} {'Loss':>8}"
        )
        print("  " + "-" * 76)
        for i in range(n_runs):
            r0 = all_results[0][i]
            acc_vals = [all_results[s][i].accuracy for s in range(args.num_seeds)]
            acc_mean = statistics.mean(acc_vals)
            acc_std = statistics.stdev(acc_vals) if args.num_seeds > 1 else 0.0
            loss_mean = statistics.mean(
                [all_results[s][i].final_loss for s in range(args.num_seeds)]
            )
            print(
                f"  {r0.num_layers:>6} {r0.loop_count:>5} {r0.total_depth:>6} "
                f"{r0.n_params / 1e6:>8.2f}M {r0.max_ops:>8} "
                f"{acc_mean:>8.4f}±{acc_std:.4f} {loss_mean:>8.4f}"
            )
        print("=" * 80)
        print("Expected: looped models outperform non-looped at same total depth")
        print()
        final_results = all_results[0]

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "mano_results.csv"
    with open(results_path, "w") as f:
        f.write("num_layers,loop_count,total_depth,n_params,max_ops,accuracy,final_loss\n")
        for r in final_results:
            f.write(
                f"{r.num_layers},{r.loop_count},{r.total_depth},{r.n_params},"
                f"{r.max_ops},{r.accuracy:.6f},{r.final_loss:.6f}\n"
            )
    print(f"Results saved to {results_path}")


def run_mano_collect(args) -> None:
    import csv
    import statistics
    from collections import defaultdict

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    # Collect all mano_results.csv files from subdirectories
    csv_files = sorted(input_dir.glob("*/mano_results.csv"))
    if not csv_files:
        # Also try flat layout
        csv_files = sorted(input_dir.glob("mano_result_*.csv"))
    if not csv_files:
        print(f"Error: no mano result CSVs found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} result file(s) in {input_dir}")

    # Read all rows
    rows: list[dict] = []
    for csv_file in csv_files:
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    if not rows:
        print("Error: no data rows found")
        sys.exit(1)

    # Group by (num_layers, loop_count, total_depth)
    groups: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (int(row["num_layers"]), int(row["loop_count"]), int(row["total_depth"]))
        groups[key] = groups.get(key, [])
        groups[key].append(row)

    # Print aggregated table
    print()
    print("=" * 80)
    print(f"{'MANO RESULTS — Aggregated across seeds':^80}")
    print("=" * 80)
    print(
        f"  {'Layers':>6} {'Loop':>5} {'Depth':>6} {'Params':>10} "
        f"{'max_ops':>8} {'N':>4} {'Accuracy':>16} {'Loss':>14}"
    )
    print("  " + "-" * 76)

    combined_rows: list[dict] = []
    for key in sorted(groups.keys()):
        entries = groups[key]
        n = len(entries)
        acc_vals = [float(e["accuracy"]) for e in entries]
        loss_vals = [float(e["final_loss"]) for e in entries]
        acc_mean = statistics.mean(acc_vals)
        acc_std = statistics.stdev(acc_vals) if n > 1 else 0.0
        loss_mean = statistics.mean(loss_vals)
        loss_std = statistics.stdev(loss_vals) if n > 1 else 0.0

        num_layers, loop_count, total_depth = key
        n_params = int(entries[0]["n_params"])
        max_ops = int(entries[0]["max_ops"])

        print(
            f"  {num_layers:>6} {loop_count:>5} {total_depth:>6} "
            f"{n_params / 1e6:>8.2f}M {max_ops:>8} {n:>4} "
            f"{acc_mean:>8.4f}±{acc_std:.4f} {loss_mean:>7.4f}±{loss_std:.4f}"
        )

        combined_rows.append({
            "num_layers": num_layers,
            "loop_count": loop_count,
            "total_depth": total_depth,
            "n_params": n_params,
            "max_ops": max_ops,
            "n_seeds": n,
            "accuracy_mean": f"{acc_mean:.6f}",
            "accuracy_std": f"{acc_std:.6f}",
            "loss_mean": f"{loss_mean:.6f}",
            "loss_std": f"{loss_std:.6f}",
        })

    print("=" * 80)
    print("Expected: looped models outperform non-looped at same total depth")
    print()

    # Write combined CSV
    output_path = input_dir / "mano_results_combined.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "num_layers", "loop_count", "total_depth", "n_params", "max_ops",
            "n_seeds", "accuracy_mean", "accuracy_std", "loss_mean", "loss_std",
        ])
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"Combined results saved to {output_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "capo":
        run_capo(args)
    elif args.command == "mano":
        run_mano(args)
    elif args.command == "mano-collect":
        run_mano_collect(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
