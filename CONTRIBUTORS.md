# Contributors

This project is an open research implementation of Looped Language Models based on [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) (Ouro, NeurIPS 2026).

## Maintainers & Authors
- **supus4562** — Architecture, core recurrence implementation, training pipeline, and evaluation harness.

## AI Contributors
- **Claude** (Anthropic) — Key architectural contributions:
  - KV-cache sharing and efficient autoregressive decoding (`last_step`, `full`, `averaged`, `first_step`).
  - Pre-allocated zero-copy dynamic `KVCache` buffer eliminating $O(S^2)$ memory fragmentation.
  - Token-level and batch-heterogeneous early exit with exact PDF/CDF distributions matching ByteDance Ouro-1.4B reference dynamics.
  - Multi-stage curriculum training with automatic stage transitions (1a, 1b, 2, 3, 4), dynamic RoPE base frequency annealing ($10\text{K} \to 40\text{K} \to 1\text{M}$), per-stage gradient accumulation scaling, and dynamic DataLoader context expansion.
  - Model upcycling (1.4B 24 layers → 2.6B 48 layers) via layer duplication.
  - Bidirectional ByteDance Ouro Hugging Face weight converter supporting official checkpoints (`ByteDance/Ouro-1.4B`, `ByteDance/Ouro-2.6B`) with `safetensors`.
