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
- [ ] Early exit inference strategies (Q-exit, hidden state diff)
- [ ] KV cache sharing for efficient decode
- [ ] Evaluation harness integration (lm-eval)
- [ ] Multi-stage training (stage transitions, RoPE base annealing)
- [ ] Scale to 1.4B / upcycling to 2.6B

## Reference

- Paper: [arXiv:2510.25741v4](https://arxiv.org/abs/2510.25741)
- Project page: https://ouro-llm.github.io
