# Contributors & Attribution

This project is an open research implementation of Looped Language Models based on [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) (Ouro, NeurIPS 2026).

This repository is a downstream fork of the upstream project [`rkstgr/LoopLM`](https://github.com/rkstgr/LoopLM).

## Original Author & Upstream Creator
- **Erik Steiger** ([@rkstgr](https://github.com/rkstgr), `rkstgr@gmail.com`) — Original author and creator of LoopLM. Designed, developed, and authored all work in commits 1 through 58 (up to snapshot `8d79e31`), including:
  - Foundational looped transformer architecture with sandwich normalization (`TransformerBlock`, `RMSNorm`, `SwiGLU`, `RoPE`).
  - Core recurrent forward pass and multi-step exit gate loss.
  - Initial training loop, Adam-atan2 optimizer, and JURECA supercomputing / SLURM infrastructure.
  - Gate finetuning (`GateFinetuner`) and Q-ACT adaptive computation time heads.
  - MANO representation geometry and dynamical systems trajectory analysis.

## Fork Maintainer & Enhancements
- **supus4562** ([@supus4562](https://github.com/supus4562)) — Fork maintainer; oversaw extension to Ouro-1.4B / 2.6B parity, multi-stage curriculum orchestration, upcycling infrastructure, KV cache sharing benchmarks, security hardening, and ongoing development.

## AI Contributors
- **Claude** (Anthropic) — Key architectural contributions:
  - KV-cache sharing and efficient autoregressive decoding (`last_step`, `full`, `averaged`, `first_step`).
  - Pre-allocated zero-copy dynamic `KVCache` buffer eliminating $O(S^2)$ memory fragmentation.
  - Token-level and batch-heterogeneous early exit with exact PDF/CDF distributions matching ByteDance Ouro-1.4B reference dynamics.
  - Multi-stage curriculum training with automatic stage transitions (1a, 1b, 2, 3, 4), dynamic RoPE base frequency annealing ($10\text{K} \to 40\text{K} \to 1\text{M}$), per-stage gradient accumulation scaling, and dynamic DataLoader context expansion.
  - Model upcycling (1.4B 24 layers → 2.6B 48 layers) via layer duplication.
  - Bidirectional ByteDance Ouro Hugging Face weight converter supporting official checkpoints (`ByteDance/Ouro-1.4B`, `ByteDance/Ouro-2.6B`) with `safetensors`.
  - Checkpoint safety serialization (`load_checkpoint_safe`) and artifact checksum verification.
