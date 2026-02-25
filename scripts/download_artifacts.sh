#!/usr/bin/env bash
# Run on login node (has internet). Downloads tokenizer + dataset into
# $PROJECT/hf_cache so compute nodes can read them offline.
set -euo pipefail

HF_CACHE="${PROJECT}/hf_cache"
export HF_HOME="$HF_CACHE"

echo "==> Downloading to $HF_CACHE"
mkdir -p "$HF_CACHE"

uv run python - <<'EOF'
import os
from transformers import AutoTokenizer
from datasets import load_dataset

print("Downloading tokenizer...")
AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

print("Downloading wikitext-103-v1 (train split)...")
load_dataset("wikitext", "wikitext-103-v1", split="train")

print("Done. All artifacts cached.")
EOF

echo "==> Cache location: $HF_CACHE"
du -sh "$HF_CACHE"
