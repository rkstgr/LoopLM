# LoopLM

Reimplementation of [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) (Ouro, arXiv:2510.25741v4).

A decoder-only transformer where a shared-weight layer stack is applied recurrently, enabling adaptive computation depth at inference time.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd looplm
uv sync
```

## Running Tests

```bash
uv run --with pytest pytest tests/ -v
```

Run and stop at first failure:

```bash
uv run --with pytest pytest tests/ -x -v
```

Run a specific test file:

```bash
uv run --with pytest pytest tests/test_model.py -v
```

## Project Structure

```
src/model/       # Model architecture (config, transformer, RoPE, LoopLM)
src/training/    # Training objectives and trainer
src/inference/   # Early exit strategies and KV cache
src/analysis/    # Mechanistic analysis experiments
configs/         # YAML configs for different model sizes
scripts/         # Entry points for training, evaluation, analysis
tests/           # Tests mirroring src/ structure
```

## Reference

- Paper: arXiv:2510.25741v4
- Project page: https://ouro-llm.github.io
