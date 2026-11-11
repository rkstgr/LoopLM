# Contributors

This project is an open research implementation of Looped Language Models based on [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) (Ouro, NeurIPS 2026).

## Active Maintainers & Contributors

| Contributor | Role & Contributions |
|:---|:---|
| [![supus4562](https://github.com/supus4562.png?size=64)](https://github.com/supus4562)<br>**[supus4562](https://github.com/supus4562)** | **Maintainer & Lead Contributor**<br>• Core architecture extensions & Ouro-1.4B/2.6B scaling<br>• KV-cache sharing and autoregressive decode engine (`last_step`, `full`, `averaged`)<br>• Multi-stage curriculum training pipeline (Stages 1a–4) with dynamic RoPE base frequency annealing<br>• Stability-driven model upcycling via layer duplication (24L → 48L)<br>• Official ByteDance Hugging Face bidirectional weight converter (`safetensors`)<br>• Serverless cloud GPU training on Modal.com & universal multi-GPU DDP launcher<br>• Checkpoint security defenses (`load_checkpoint_safe`) and artifact verification |
| [![rkstgr](https://github.com/rkstgr.png?size=64)](https://github.com/rkstgr)<br>**[Erik Steiger (@rkstgr)](https://github.com/rkstgr)** | **Original Upstream Author (Commits 1–58)**<br>• Initial recurrent transformer architecture with sandwich normalization (`TransformerBlock`, `RMSNorm`, `SwiGLU`, `RoPE`)<br>• Core recurrent forward pass and multi-step exit gate loss<br>• Initial training loop, Adam-atan2 optimizer, and JURECA supercomputing infrastructure<br>• Gate finetuning (`GateFinetuner`) and Q-ACT adaptive computation time heads<br>• MANO representation geometry and Capo knowledge capacity analysis suites |

---

## AI Contributors

| Contributor | Role & Contributions |
|:---|:---|
| **[Claude](https://anthropic.com)** (Anthropic) | **AI Research & Engineering Contributor**<br>• Co-authoring and implementation assistance across KV-cache sharing, token-level early exit matching Ouro reference dynamics, multi-stage training pipeline, layer duplication upcycling, and cloud runners |

---

## Attribution & Repository Lineage

This repository is maintained and developed by **[@supus4562](https://github.com/supus4562)** as an enhanced downstream fork of [`rkstgr/LoopLM`](https://github.com/rkstgr/LoopLM). All foundational architecture from commits 1 through 58 is authored by original creator **Erik Steiger** (`@rkstgr`), and all subsequent enhancements, Ouro scaling milestones, cloud runners, and new capabilities are authored by **@supus4562**.

---

## Community Contributions Encouraged

We warmly welcome and actively encourage contributions from the broader research and open-source community! 

Whether you are interested in:
- Proposing novel recurrent architectures, routing mechanisms, or step-skipping schemes
- Adding benchmark evaluation suites (e.g., GSM8K, MATH, HumanEval, ARC)
- Writing optimized Triton or CUDA attention kernels
- Performing mechanistic interpretability on recurrent hidden state geometry
- Fixing bugs, writing documentation, or expanding unit tests

All contributions, feedback, and pull requests are welcomed. Feel free to open an issue or submit a pull request!

