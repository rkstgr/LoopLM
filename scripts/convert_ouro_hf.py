#!/usr/bin/env python
"""scripts/convert_ouro_hf.py — Convert weights between official ByteDance Ouro (Hugging Face) and LoopLM.

Examples:
    # 1. Download & convert official ByteDance Ouro-1.4B from HuggingFace to LoopLM checkpoint:
    uv run scripts/convert_ouro_hf.py --hf-model ByteDance/Ouro-1.4B --output checkpoints/ouro_1_4b.pt

    # 2. Convert a local HuggingFace folder to LoopLM:
    uv run scripts/convert_ouro_hf.py --hf-model ./hf_ouro_model --output checkpoints/ouro_model.pt

    # 3. Export a trained LoopLM checkpoint to Hugging Face format (config.json + model.safetensors):
    uv run scripts/convert_ouro_hf.py --input-checkpoint checkpoints/step_0010000.pt --output-dir ./hf_export --to-hf
"""

import argparse
import sys
from pathlib import Path

# Allow importing from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.model.hf_converter import load_from_hf, save_to_hf
from src.utils import load_checkpoint_safe


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert model weights between ByteDance Ouro (Hugging Face) and LoopLM."
    )
    # HF -> LoopLM direction
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="HuggingFace repo ID (e.g. ByteDance/Ouro-1.4B) or local directory containing config.json and weights",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Target LoopLM .pt checkpoint path",
    )
    # LoopLM -> HF direction
    parser.add_argument(
        "--input-checkpoint",
        type=str,
        default=None,
        help="Source LoopLM .pt checkpoint path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target directory to write HuggingFace config.json and model.safetensors",
    )
    parser.add_argument(
        "--to-hf",
        action="store_true",
        help="Export LoopLM checkpoint to Hugging Face format",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Precision for converted weights",
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        default=False,
        help="Allow unpickling legacy arbitrary code if safe loading fails",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    if args.to_hf:
        if not args.input_checkpoint or not args.output_dir:
            print("Error: --to-hf requires --input-checkpoint and --output-dir", file=sys.stderr)
            sys.exit(1)

        print(f"Loading LoopLM checkpoint from {args.input_checkpoint}...")
        ckpt = load_checkpoint_safe(args.input_checkpoint, map_location="cpu", trust=args.trust_checkpoint)

        if "model_config" in ckpt:
            cfg = LoopLMConfig(**ckpt["model_config"])
        else:
            cfg = LoopLMConfig.ouro_1_4b()

        model = LoopLM(cfg).to(dtype=dtype)
        model.load_state_dict(ckpt["model_state_dict"])

        print(f"Exporting to Hugging Face format at {args.output_dir}...")
        out_dir = save_to_hf(model, args.output_dir, safe_serialization=True)
        print(f"Successfully exported Hugging Face model to {out_dir}")

    else:
        if not args.hf_model or not args.output:
            print("Error: HF->LoopLM conversion requires --hf-model and --output", file=sys.stderr)
            sys.exit(1)

        print(f"Loading Hugging Face model from {args.hf_model}...")
        model = load_from_hf(args.hf_model, device="cpu", dtype=dtype)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        import dataclasses
        ckpt = {
            "step": 0,
            "model_state_dict": model.state_dict(),
            "model_config": dataclasses.asdict(model.config),
            "stage_idx": 0,
            "stage_start_step": 0,
        }
        torch.save(ckpt, out_path)
        print(f"Successfully converted and saved LoopLM checkpoint to {out_path}")


if __name__ == "__main__":
    main()
