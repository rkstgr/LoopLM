#!/usr/bin/env python
"""Pre-tokenize SmolLM-Corpus into .npy shards for offline training.

Run on the login node (needs internet). Streams from HuggingFace so RAM usage
stays bounded. Saves uint16 .npy shards (~100M tokens each).

Usage:
    # Pilot run (10B tokens):
    uv run scripts/pretokenize_smollm.py --max-tokens 10_000_000_000

    # Full corpus (~252B tokens):
    uv run scripts/pretokenize_smollm.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Pre-tokenize SmolLM-Corpus")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to save shards (default: $SCRATCH/looplm/data/smollm_pretokenized)",
    )
    p.add_argument(
        "--tokenizer-id",
        default="HuggingFaceTB/SmolLM2-135M",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=100_000_000,
        help="Tokens per shard file (default: 100M)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Stop after this many tokens (default: process everything)",
    )
    p.add_argument(
        "--subsets",
        nargs="+",
        default=["fineweb-edu-dedup", "cosmopedia-v2", "python-edu"],
        help="SmolLM-Corpus subsets to process",
    )
    return p.parse_args()


def pretokenize_subset(
    subset: str,
    tokenizer,
    output_dir: Path,
    shard_size: int,
    max_tokens: int | None,
    global_token_count: int,
):
    """Stream and tokenize one subset, writing .npy shards.

    Returns the updated global token count.
    """
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print(f"Processing subset: {subset}")
    print(f"{'='*60}")

    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus", subset, split="train", streaming=True
    )

    eos = tokenizer.eos_token_id or 0
    buffer = []
    shard_idx = 0
    doc_count = 0
    t0 = time.time()

    # Check for existing shards to resume
    existing = sorted(output_dir.glob(f"{subset}_shard_*.npy"))
    if existing:
        last_shard = existing[-1].stem
        shard_idx = int(last_shard.split("_")[-1]) + 1
        existing_tokens = sum(np.load(p, mmap_mode="r").shape[0] for p in existing)
        global_token_count += existing_tokens
        print(
            f"  Resuming: found {len(existing)} existing shards "
            f"({existing_tokens/1e9:.2f}B tokens), starting at shard {shard_idx}"
        )
        # Skip documents proportional to existing tokens (approximate)
        # This is imperfect but avoids re-processing most data
        docs_to_skip = int(existing_tokens / 500)  # ~500 tokens per doc average
        print(f"  Skipping ~{docs_to_skip:,} documents...")
        ds = ds.skip(docs_to_skip)

    for doc in ds:
        text = doc.get("text", "")
        if not text or not text.strip():
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue

        buffer.extend(ids)
        buffer.append(eos)
        doc_count += 1

        # Write shard when buffer is full
        while len(buffer) >= shard_size:
            arr = np.array(buffer[:shard_size], dtype=np.uint16)
            shard_path = output_dir / f"{subset}_shard_{shard_idx:04d}.npy"
            np.save(shard_path, arr)
            global_token_count += shard_size
            shard_idx += 1
            buffer = buffer[shard_size:]

            elapsed = time.time() - t0
            rate = global_token_count / elapsed if elapsed > 0 else 0
            print(
                f"  Shard {shard_idx - 1:4d} | "
                f"{global_token_count/1e9:.2f}B tokens | "
                f"{doc_count:,} docs | "
                f"{rate/1e6:.1f}M tok/s | "
                f"{elapsed/3600:.1f}h elapsed"
            )

            if max_tokens and global_token_count >= max_tokens:
                print(f"  Reached --max-tokens limit ({max_tokens/1e9:.1f}B)")
                return global_token_count

    # Write remaining buffer
    if buffer:
        arr = np.array(buffer, dtype=np.uint16)
        shard_path = output_dir / f"{subset}_shard_{shard_idx:04d}.npy"
        np.save(shard_path, arr)
        global_token_count += len(buffer)
        print(f"  Final shard {shard_idx} ({len(buffer):,} tokens)")

    elapsed = time.time() - t0
    print(
        f"  Done: {subset} | {doc_count:,} docs | "
        f"{global_token_count/1e9:.2f}B total tokens | {elapsed/3600:.1f}h"
    )
    return global_token_count


def main():
    args = parse_args()

    output_dir = Path(
        args.output_dir
        or os.path.join(
            os.environ.get("SCRATCH", "/tmp"),
            "looplm/data/smollm_pretokenized",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Shard size: {args.shard_size/1e6:.0f}M tokens")
    if args.max_tokens:
        print(f"Max tokens: {args.max_tokens/1e9:.1f}B")
    print(f"Subsets: {args.subsets}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    print(f"Tokenizer: {args.tokenizer_id} (vocab_size={tokenizer.vocab_size})")
    assert tokenizer.vocab_size <= 65535, "vocab_size must fit in uint16"

    global_token_count = 0
    t_start = time.time()

    for subset in args.subsets:
        global_token_count = pretokenize_subset(
            subset=subset,
            tokenizer=tokenizer,
            output_dir=output_dir,
            shard_size=args.shard_size,
            max_tokens=args.max_tokens,
            global_token_count=global_token_count,
        )
        if args.max_tokens and global_token_count >= args.max_tokens:
            break

    total_time = time.time() - t_start
    total_gb = sum(f.stat().st_size for f in output_dir.glob("*.npy")) / 1e9
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"  Total tokens: {global_token_count/1e9:.2f}B")
    print(f"  Total size: {total_gb:.1f} GB")
    print(f"  Total time: {total_time/3600:.1f}h")
    print(f"  Shards: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
