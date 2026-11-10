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

### Mano task

`scripts/mano.slurm` — Rewritten as a SLURM job array:
  - --array=0-8 runs 9 tasks (3 configs × 3 seeds), each with 1 GPU
  - Maps SLURM_ARRAY_TASK_ID → (config, seed) using bash arrays
  - Each task writes to $OUT_DIR/task_<ID>/mano_results.csv
  - Logs use %A_%a pattern (array job ID + task ID)

  `scripts/analyze.py` — Added mano-collect subcommand:
  - Reads all */mano_results.csv files from the input directory
  - Groups by (num_layers, loop_count, total_depth), computes mean ± std
  - Prints a formatted table and writes mano_results_combined.csv

  Usage:
  # Submit all 9 jobs in parallel
  ```bash
  sbatch scripts/mano.slurm
  ```

  # After all complete, aggregate results
  ```bash
  uv run scripts/analyze.py mano-collect --input-dir ${SCRATCH}/looplm/mano
  ```

### JUWELS (westai / H100)

**One-time setup on login node** (needs internet):

```bash
uv sync
bash scripts/download_artifacts.sh   # caches tokenizer + dataset into $PROJECT/hf_cache
```

**Submit job:**

```bash
# Set --account in scripts/train.slurm first, then:
sbatch scripts/train.slurm
```

**Monitor:**

```bash
squeue --me
tail -f $(ls -t ${SCRATCH}/looplm/logs/*.out | head -1)
```

**After completion — sync wandb from login node:**

```bash
wandb sync ${SCRATCH}/looplm/wandb/offline-run-*/
```

Compute nodes have no internet access. The SLURM script sets `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `WANDB_MODE=offline` automatically.

## Running Tests

```bash
uv run --with pytest pytest tests/ -v
```

Stop at first failure:

```bash
uv run --with pytest pytest tests/ -x -v
```

## Project Structure

```
src/
  model/
    config.py        # LoopLMConfig dataclass (small ~100M, ouro_1_4b, ouro_2_6b)
    rope.py          # Rotary position embeddings with configurable base frequency
    transformer.py   # TransformerBlock with sandwich norm, SwiGLU, RoPE
    looplm.py        # LoopLM: recurrent forward pass, exit gate, per-step logits
  training/
    objectives.py    # Stage I entropy-regularized loss; Stage II adaptive gate loss
    trainer.py       # Training loop: AdamW, grad clip, checkpointing, wandb logging
    data.py          # HuggingFace dataset loading and tokenization
scripts/
  train.py                  # Training entry point (argparse CLI)
  train.slurm               # SLURM job script for JUWELS westai partition
  download_artifacts.sh     # Pre-stage HF artifacts on login node
configs/
  small.yaml                # ~100M params (6 layers, hidden 768, 12 heads)
  medium.yaml               # ~400M params
  ouro_1_4b.yaml            # 1.4B, 24 layers, hidden 2048
  ouro_2_6b.yaml            # 2.6B, 48 layers, hidden 2048
tests/                      # 74 tests covering all modules above
```

## Model Configs

| Config      | Params | Layers | Hidden | Heads | Recurrent steps |
|-------------|--------|--------|--------|-------|-----------------|
| `small()`   | ~100M  | 6      | 768    | 12    | 4               |
| `ouro_1_4b` | 1.4B   | 24     | 2048   | 16    | 4               |
| `ouro_2_6b` | 2.6B   | 48     | 2048   | 16    | 4               |

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

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the full list of project contributors.
- **supus4562** — Core architecture, training framework, evaluation harness.
- **Claude** (Anthropic) — KV-cache sharing, token-level early exit matching Ouro, multi-stage training pipeline, model upcycling (1.4B → 2.6B), and official ByteDance Hugging Face weight converter.


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

