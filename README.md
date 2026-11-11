# LoopLM

Reimplementation of [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) (Ouro, arXiv:2510.25741v4).

A decoder-only transformer where a shared-weight layer stack is applied recurrently, enabling adaptive computation depth at inference time.

## Architecture

- **Shared-weight recurrence** — a stack of N layers is applied T times, reusing the same weights each pass
- **Sandwich normalization** — RMSNorm before both attention and FFN (critical for recurrent stability)
- **Exit gate** — at each step, a learned gate predicts whether to stop early
- **Multi-step loss** — all T steps contribute to training via a weighted sum, where weights come from the learned exit distribution
- **SwiGLU FFN**, **RoPE** positional encoding, 49,152 token vocabulary (SmolLM2 tokenizer)

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd looplm
uv sync
```

## Training

### Local

```bash
uv run scripts/train.py \
    --model-config small \
    --dataset wikitext \
    --dataset-config wikitext-103-v1 \
    --max-steps 10000 \
    --batch-size 8 \
    --seq-len 512 \
    --lr 3e-4 \
    --output-dir ./checkpoints
```

## Project Structure

```
LoopLM/
├── src/
│   ├── model/
│   │   ├── config.py            # LoopLMConfig (small ~100M, ouro_1_4b, ouro_2_6b)
│   │   ├── looplm.py            # LoopLM: recurrent forward pass, exit gate, generation
│   │   ├── transformer.py       # TransformerBlock with sandwich norm, SwiGLU, RoPE MHA
│   │   ├── rope.py              # Rotary position embeddings with dynamic base annealing
│   │   ├── upcycle.py           # Model upcycling (layer duplication 24L -> 48L)
│   │   ├── hf_converter.py      # Bidirectional ByteDance Ouro HF converter (safetensors)
│   │   └── moe_layer.py         # Mixture-of-Experts recurrent layer ablation
│   ├── training/
│   │   ├── trainer.py           # Multi-stage Trainer (1a-4, RoPE annealing, DDP, AMP)
│   │   ├── objectives.py        # Stage I joint loss & Stage II adaptive gate loss
│   │   ├── data.py              # HuggingFace datasets & pre-tokenized DataLoader
│   │   └── adam_atan2.py        # Adam-atan2 optimizer
│   ├── inference/
│   │   ├── kv_cache.py          # KV cache sharing (last_step, full, averaged, first_step)
│   │   ├── early_exit.py        # Token-level Q-exit, static exit, hidden state diff
│   │   └── lm_eval_wrapper.py   # LM Evaluation Harness integration (LoopLMLM adapter)
│   ├── analysis/
│   │   ├── capo.py              # Section 6.1: Knowledge Capacity (memorization bits/param)
│   │   ├── mano.py              # Section 6.2: Knowledge Manipulation (tree arithmetic mod 23)
│   │   ├── arithmetic.py        # Prefix-notation modular arithmetic generator
│   │   └── gen_name.py          # Entity & synthetic graph generator
│   └── utils/
│       └── checkpoint.py        # Safe checkpoint loader (weights_only & allowlist)
├── scripts/
│   ├── train.py                 # Stage I continuous pre-training (CLI, multi-stage)
│   ├── finetune_gate.py         # Stage II adaptive gate fine-tuning (frozen LM)
│   ├── evaluate.py              # Standalone LM-Eval benchmark runner
│   ├── eval_checkpoints.py      # Multi-checkpoint evaluation across recurrent steps
│   ├── analyze.py               # Capo and Mano experiment suite & SLURM collector
│   ├── upcycle.py               # CLI for model upcycling (24 -> 48 layers)
│   ├── convert_ouro_hf.py       # CLI for ByteDance Ouro Hugging Face weight conversion
│   ├── verify_artifacts.py      # SHA-256 checksum verification for cached tokenizers
│   ├── check_secrets.py         # Pre-commit secret & API key scanner
│   ├── pretokenize_smollm.py    # Offline dataset pre-tokenization
│   ├── download_artifacts.sh    # Pre-stage HF artifacts for offline HPC nodes
│   └── *.slurm                  # HPC SLURM job array scripts (JUWELS H100)
├── tests/                       # Test suites (260+ tests passing)
├── paper/                       # Ouro, Capo/Mano, and HRM research papers
├── .pre-commit-config.yaml      # Pre-commit hooks for secret scanning & code hygiene
├── CONTRIBUTORS.md              # Project authors and AI contributors
├── LICENSE                      # Apache License 2.0
├── pyproject.toml
└── README.md
```

## Model Configs

| Config | Parameters | Layers | Hidden ($d_{\text{model}}$) | Heads | Intermediate | Recurrent Steps ($T$) | RoPE Base |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `small()` | ~100M | 6 | 768 | 12 | 2048 | 4 | 10K |
| `ouro_1_4b()` | 1.4B | 24 | 2048 | 16 | 5632 | 4 | 10K $\to$ 40K $\to$ 1M |
| `ouro_2_6b()` | 2.6B | 48 | 2048 | 16 | 5632 | 4 | 10K $\to$ 40K $\to$ 1M |

## Training

### Stage I Pre-Training (Local & Distributed)

```bash
# Single GPU quick run
uv run scripts/train.py \
    --model-config small \
    --dataset wikitext \
    --dataset-config wikitext-103-v1 \
    --max-steps 10000 \
    --batch-size 8 \
    --seq-len 512 \
    --lr 3e-4 \
    --output-dir ./checkpoints
```

### Stage II Gate Fine-Tuning

In Stage II, all language model parameters are frozen, and only the exit gate is trained using the detached performance improvement signal ($I_i^{(t)} = \max(0, \mathcal{L}^{(t-1)} - \mathcal{L}^{(t)})$):

```bash
uv run scripts/finetune_gate.py \
    --checkpoint checkpoints/step_0010000.pt \
    --max-steps 5000 \
    --batch-size 8 \
    --seq-len 512 \
    --k 50.0 --gamma 0.005
```

### JUWELS (westai / H100 Cluster Execution)

**One-time setup on login node:**
```bash
uv sync
bash scripts/download_artifacts.sh   # caches tokenizer + dataset into $PROJECT/hf_cache
```

**Submit pre-training:**
```bash
sbatch scripts/train.slurm
```

## Research & Analysis (Capo & Mano)

LoopLM provides reproduction suites for the empirical experiments in Section 6 of the paper:

### Knowledge Manipulation (Mano Task — Section 6.2)
Trains models on modular arithmetic tree expressions (mod 23) to evaluate whether recurrent depth enhances multi-step manipulation under iso-FLOP budgets:

```bash
# Paper reproduction with learning rate search across seeds
uv run scripts/analyze.py mano \
    --max-ops 10 \
    --model-preset paper \
    --lr-search \
    --num-seeds 3 \
    --report-best

# Submit 9 parallel SLURM array jobs on cluster
sbatch scripts/mano.slurm

# Aggregate results across all seed tasks into mano_results_combined.csv
uv run scripts/analyze.py mano-collect --input-dir ${SCRATCH}/looplm/mano
```

### Knowledge Capacity (Capo Task — Section 6.1)
Measures memorization capacity (bits of knowledge per parameter) across model sizes (`micro`, `mini`, `small`) comparing $T=1$ vs. $T=4$:

```bash
uv run scripts/analyze.py capo \
    --n-individuals 20000 \
    --train-exposures 1000 \
    --model-sizes micro mini small \
    --loop-counts 1 4 \
    --num-seeds 3

# Aggregate Capo SLURM array jobs
uv run scripts/analyze.py capo-collect --input-dir ${SCRATCH}/looplm/capo
```

## Evaluation Harness Integration

Evaluate LoopLM checkpoints against standard benchmarks (ARC-Easy, ARC-Challenge, HellaSwag, MMLU, GSM8K) across different recurrent depths $T=1..T_{\text{max}}$:

```bash
# Evaluate across all recurrent steps T=1..4
uv run scripts/evaluate.py \
    --checkpoint checkpoints/step_0010000.pt \
    --tasks arc_easy,hellaswag \
    --eval-all-steps \
    --limit 100

# Batch evaluate checkpoints in a directory
uv run scripts/eval_checkpoints.py \
    --checkpoint-dir checkpoints/ \
    --tasks arc_easy \
    --eval-all-steps
```

## Running Tests

The test suite contains **262 tests across 15 test files**, verifying architecture, numerical stability, KV cache, early exit, multi-stage scheduling, and weight conversion:

```bash
# Run all tests
uv run pytest

# Verbose output
uv run pytest -v

# Stop at first failure
uv run pytest -x
```

## Implementation Status

- [x] Config system (`LoopLMConfig`)
- [x] RoPE with configurable base frequency
- [x] Transformer block (sandwich norm, SwiGLU, RoPE attention)
- [x] LoopLM model (recurrence, exit gate, per-step logits)
- [x] Stage I training objective (entropy-regularized, uniform prior)
- [x] Stage II adaptive gate loss (detached improvement signal)
- [x] Training loop (AdamW, grad clip, checkpointing, wandb)
- [x] Data pipeline (HuggingFace datasets, tokenization)
- [x] Training entry point (`scripts/train.py`)
- [x] SLURM job script for JUWELS westai (H100)
- [x] Early exit inference strategies (token-level Q-exit matching Ouro, hidden state diff)
- [x] KV cache sharing for efficient decode (last_step, full, averaged, first_step)
- [x] Evaluation harness integration (lm-eval wrapper)
- [x] Multi-stage training (stage transitions, RoPE base annealing, Stages 1a-4)
- [x] Scale to 1.4B / upcycling to 2.6B (layer duplication, HF weight converter)

## Multi-Stage Curriculum Training

LoopLM implements the complete 4-stage training curriculum from Table 1 of the Ouro paper:

- **Stage 1a** ($T=8$, constant LR $3\times 10^{-4}$, $\beta=0.1$, RoPE base $10\text{K}$, 4K context) — initial deep recurrence setup.
- **Stage 1b** ($T=4$, constant LR $3\times 10^{-4}$, $\beta=0.1$, RoPE base $10\text{K}$, 4K context) — stabilization of gradient dynamics.
- **Stage 2** ($T=4$, cosine LR $3\times 10^{-5}$, $\beta=0.05$, RoPE base **$40\text{K}$**, 16K context) — reasoning data annealing and context expansion.
- **Stage 3** ($T=4$, constant LR $3\times 10^{-5}$, $\beta=0.05$, RoPE base **$1\text{M}$**, 64K context) — ultra-long context annealing.
- **Stage 4** ($T=4$, cosine LR $1\times 10^{-5}$, $\beta=0.05$, RoPE base $1\text{M}$, 32K context) — reasoning SFT fine-tuning.

Stage transitions happen automatically during training, updating learning rate schedules, recurrent steps $T$, entropy penalty $\beta$, RoPE base frequency via `model.rope.set_base(...)`, gradient accumulation steps, and rebuilding DataLoaders for expanded context windows.

```bash
# Example: 5-stage continuous curriculum
uv run scripts/train.py \
    --stage1a-steps 2000 \
    --stage1b-steps 2000 \
    --stage2-steps 1500 \
    --stage3-steps 500 \
    --stage4-steps 1000 \
    --batch-size 4 --seq-len 512
```

## Model Upcycling (1.4B → 2.6B)

Following Section 4.3.1 of the Ouro paper, LoopLM supports **Stability-Driven Upcycling**:
In Stage 1b, the 24-layer 1.4B model is scaled to 48 layers (2.6B parameters) via layer duplication. Because layers are already trained recurrently, layer 23's outputs reside on the exact manifold layer 0 expects, eliminating representation shock:

```bash
# Upcycle a 24-layer checkpoint to a 48-layer checkpoint
uv run scripts/upcycle.py \
    --input checkpoints/ouro_1_4b_stage1a.pt \
    --output checkpoints/ouro_2_6b_stage1b.pt \
    --target-layers 48
```

## ByteDance Ouro Hugging Face Weight Interoperability

LoopLM provides bidirectional weight conversion with official pre-trained ByteDance checkpoints (`ByteDance/Ouro-1.4B`, `ByteDance/Ouro-2.6B`) using `safetensors`:

```bash
# 1. Download & convert official ByteDance/Ouro-1.4B from Hugging Face into a LoopLM checkpoint:
uv run scripts/convert_ouro_hf.py \
    --hf-model ByteDance/Ouro-1.4B \
    --output checkpoints/ouro_1_4b.pt

# 2. Export a trained LoopLM checkpoint to Hugging Face format (config.json + model.safetensors):
uv run scripts/convert_ouro_hf.py \
    --input-checkpoint checkpoints/step_0010000.pt \
    --output-dir ./hf_export \
    --to-hf
```

Programmatic loading in Python:
```python
from src.model import load_from_hf

# Load directly from HuggingFace Hub:
model = load_from_hf("ByteDance/Ouro-1.4B", device="cuda")
res = model.generate(input_ids, strategy="last_step")
```

## Inference & Generation

LoopLM supports efficient autoregressive decoding with **KV-cache sharing** across recurrent steps (Section 5.4.2 & Table 14 of the Ouro paper):

- **`last_step`** (recommended): All decode steps reuse the final recurrent step's KV cache, yielding a **4× memory reduction** with negligible accuracy difference compared to uncompressed KV caches.
- **`full`**: Uncompressed baseline maintaining separate KV caches for all $T \times L$ layers.
- **`averaged`**: Maintains a running average of KV representations across recurrent loops.
- **`first_step`**: Reuses step 1 representations only (ablation baseline).

```python
import torch
from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM

model = LoopLM(LoopLMConfig.small()).eval()
input_ids = torch.tensor([[1, 42, 108]])

# Autoregressive generation with 4x memory savings via last-step KV sharing:
res = model.generate(
    input_ids,
    max_new_tokens=64,
    temperature=0.0,      # greedy decode
    strategy="last_step", # options: "last_step", "full", "averaged", "first_step"
)
print("Generated token IDs:", res.sequences)
```

## Contributors & Attribution

This repository is a downstream fork of [`rkstgr/LoopLM`](https://github.com/rkstgr/LoopLM).
- **Erik Steiger** ([@rkstgr](https://github.com/rkstgr), `rkstgr@gmail.com`) — Original author and creator of the upstream codebase (commits 1–58 up to snapshot `8d79e31`), including core recurrence loop, sandwich normalization, Q-ACT, gate finetuning, MANO analysis, training harness, and HPC scripts.
- **supus4562** ([@supus4562](https://github.com/supus4562)) — Fork maintainer; architecture extensions, multi-stage curriculum training integration, upcycling infrastructure, KV-cache sharing, checkpoint safety, and tooling.
- **Claude** (Anthropic) — AI contributor: KV-cache sharing, token-level early exit matching Ouro, multi-stage training pipeline with RoPE annealing, model upcycling (1.4B → 2.6B), official ByteDance Hugging Face weight converter, and checkpoint safety wrappers.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for full attribution details.

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.


## Future Ideas


### Detached recurrence (truncated BPTT)
Only backprop through the last recurrent step, detaching hidden states between iterations. [HRM](https://arxiv.org/) does this. Eliminates gradient explosion entirely but loses cross-step gradient coordination. Each step still gets a direct loss signal, and since weights are shared, improving one step tends to improve all.

### Diffusion-style step sampling
Instead of computing all T steps and weighting by the exit distribution, **sample a single recurrent step per batch** and importance-weight the loss. Direct parallel to diffusion training where you sample a random timestep t rather than denoising the full chain.

Relevant diffusion insights that transfer:
- [Min-SNR](https://arxiv.org/abs/2303.09556) — weight timesteps by clamped signal-to-noise ratio to resolve conflicting gradients between steps (3.4x faster convergence)
- CLTS — curriculum from uniform to focused Gaussian sampling (easy steps first, hard steps later)
- [A Closer Look at Time Steps](https://arxiv.org/html/2405.17403v1) — non-uniform timestep sampling for 3x speedup

Looped transformer work already exploring this direction:
- [LoopFormer](https://loopformer.github.io/) — conditions on internal time t and step size, allowing step-skipping (analogous to DDIM)
- [Looped Transformers are Better at Learning Learning Algorithms](https://arxiv.org/html/2311.12424v2) — randomly samples iteration count per training batch
- [Retrofitted Recurrence](https://arxiv.org/html/2511.07384v1) — schedules max backprop depth over training

A curriculum schedule (early recurrence steps first, shift to later steps as training progresses) could both stabilize training and improve sample efficiency.

## Reference

- Paper: [arXiv:2510.25741v4](https://arxiv.org/abs/2510.25741)
- Project page: https://ouro-llm.github.io

