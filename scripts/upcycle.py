#!/usr/bin/env python
"""scripts/upcycle.py — CLI to upcycle a LoopLM checkpoint (e.g., 1.4B 24 layers -> 2.6B 48 layers).

Usage:
    uv run scripts/upcycle.py --input checkpoints/ouro_1_4b_stage1a.pt --output checkpoints/ouro_2_6b_stage1b.pt
    uv run scripts/upcycle.py --input checkpoint.pt --output upcycled.pt --target-layers 48
"""

import argparse
import sys
from pathlib import Path

# Allow importing from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.upcycle import upcycle_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Upcycle a LoopLM checkpoint via layer duplication.")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to source LoopLM checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to save upcycled checkpoint (.pt file)",
    )
    parser.add_argument(
        "--target-layers", "-l",
        type=int,
        default=None,
        help="Target number of layers (defaults to 2x source layers)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading checkpoint from: {args.input}")
    output_path = upcycle_checkpoint(
        source_checkpoint_path=args.input,
        target_checkpoint_path=args.output,
        target_layers=args.target_layers,
    )
    print(f"Successfully upcycled checkpoint saved to: {output_path}")


if __name__ == "__main__":
    main()
