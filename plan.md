# Reimplementation Plan: Scaling Latent Reasoning via Looped Language Models (Ouro)

## Overview

This plan covers reimplementing the core LoopLM architecture, training pipeline, and evaluation from the paper "Scaling Latent Reasoning via Looped Language Models" (Zhu et al., 2025). The plan is structured in phases, starting with the architecture core and scaling up incrementally.

---

## Phase 1: Core LoopLM Architecture

**Goal:** Implement the basic looped transformer that can run R recurrent steps with shared weights.

### 1.1 Base Transformer Block
- Decoder-only transformer with:
  - Multi-Head Attention (MHA) with RoPE positional embeddings
  - SwiGLU feed-forward network
  - **Sandwich normalization**: RMSNorm before both attention and FFN sub-layers (critical for training stability with recurrence)
- Vocabulary: use any standard tokenizer to start (the paper uses SmolLM2's 49,152-token vocab)

### 1.2 Looping Mechanism
- Given N transformer layers parameterized by θ, define the layer stack `M_L`
- The looped forward pass applies `M_L` iteratively T times:
  ```
  F^(t)(x) = lmhead ∘ M_L ∘ M_L ∘ ... ∘ M_L (t times) ∘ emb(x)
  ```
- At each recurrent step t, produce an LM head output and compute cross-entropy loss L^(t)
- Key detail: hidden states from the output of step t become the input to step t+1 (no explicit feedback token — this is the "implicit" variant)

### 1.3 Exit Gate
- A linear layer + sigmoid applied to the final-layer hidden state at each step t:
  ```
  λ_t(x) = σ(Linear_φ(h^(t)))
  ```
- Compute survival probability: `S_t(x) = ∏_{j=1}^{t} (1 - λ_j(x))`
- Exit distribution: `p_φ(t|x) = λ_t · S_{t-1}` for t < T_max, and `p_φ(T_max|x) = S_{T_max-1}`
- This gate runs in parallel with the LM head at each step

### 1.4 Model Configurations (from Table 2)

| Model     | Params | Layers | Hidden Size | Attention | FFN    |
|-----------|--------|--------|-------------|-----------|--------|
| Ouro 1.4B | 1.4B   | 24     | 2048        | MHA       | SwiGLU |
| Ouro 2.6B | 2.6B   | 48     | 2048        | MHA       | SwiGLU |

**Recommendation for prototyping:** Start with a small model (e.g., 50-150M params, 6-12 layers, hidden 512-1024) to validate the architecture before scaling.

---

## Phase 2: Training Objectives

### 2.1 Stage I — Entropy-Regularized Objective (Pre-training)

The combined loss is:

```
L = Σ_{t=1}^{T_max} p_φ(t|x) · L^(t) - β · H(p_φ(·|x))
```

Where:
- `L^(t)` is the standard cross-entropy next-token prediction loss at step t
- `H(p_φ)` is the entropy of the exit distribution
- β controls exploration vs. exploitation (start with β=0.1)

**Interpretation:** This is equivalent to an ELBO with a uniform prior over exit steps. The entropy term prevents collapse to always using T_max.

**Implementation notes:**
- Compute L^(t) at every recurrent step (each step gets a loss signal)
- The exit gate probabilities p_φ(t|x) weight the per-step losses
- Gradients flow through both the LM parameters and the gate parameters jointly

### 2.2 Stage II — Focused Adaptive Gate Training

After pre-training, freeze the LM and train only the exit gate φ:

1. Compute detached per-step loss `L^(t)_{i,stop}` (no grad through LM)
2. Loss improvement: `I^(t)_i = max(0, L^(t-1)_{i,stop} - L^(t)_{i,stop})`
3. Ideal continuation label: `w^(t)_i = σ(k · (I^(t)_i - γ))` with k=50.0, γ=0.005
4. Binary cross-entropy between predicted continuation `(1 - λ^(t)_i)` and target `w^(t)_i`
5. Average across steps: `L_adaptive = (1/T_max) Σ_{t=2}^{T_max} L^(t)_adaptive`

This teaches the gate to exit when marginal improvement from another loop is negligible.

---

## Phase 3: Inference

### 3.1 Early Exit via CDF Threshold (Q-Exit)

At inference time:
1. Run recurrent steps sequentially
2. At each step t, compute `CDF(t|x) = Σ_{i=1}^{t} p_φ(i|x)`
3. Exit at the first step where `CDF(t|x) ≥ q` (threshold q ∈ [0,1])
4. Smaller q = earlier exit (less compute), larger q = deeper computation

### 3.2 KV Cache Strategy

- **Prefill phase:** Each recurrent step needs its own KV cache (no sharing)
- **Decoding phase:** Can reuse only the **last step's** KV cache with minimal performance loss (~0.07 drop on GSM8K) for 4× memory reduction
- Do NOT reuse first-step cache (catastrophic collapse from 78.92 → 18.73 on GSM8K)

### 3.3 Static Inference (Simpler Baseline)

For initial testing, just always run all T_max steps and use the final step's output. Add early exit later.

---

## Phase 4: Data Pipeline

### 4.1 Pre-training Data (Stages 1-4)

The paper uses entirely open-source data totaling 7.7T tokens:

| Stage | Data | Tokens | Seq Length |
|-------|------|--------|------------|
| 1a/1b (Pre-train) | Nemotron-CC, MAP-CC, OpenCoder, MegaMath | 6T dataset, 3T+3T used | 4K |
| 2 (CT Annealing) | HQ Nemotron-CC, Math, Code, SFT | 1.4T | 16K |
| 3 (LongCT) | ProLong-64K | 20B | 64K |
| 4 (Mid-training) | 20+ SFT datasets in ChatML | 300B effective | 32K |

**For a scaled-down replication:**
- Use FineWeb-Edu (1.3T tokens) or a subset for Stage 1
- Use OpenCoder-Annealing + MegaMath-HQ for Stage 2
- Skip Stage 3 initially
- Use a mix of open SFT data for Stage 4

### 4.2 SFT Data (for Ouro-Thinking)

~8.3M examples:
- Math: OpenThoughts3, AceReason-1.1-SFT (3.5M)
- Code: AceReason, OpenCodeReasoning, Llama-Nemotron (3.2M)
- Science: OpenThoughts3, Llama-Nemotron (808K)
- Chat: OO1-Chat-747K, DeepWriting-20K (767K)

---

## Phase 5: Training Schedule

### 5.1 Hyperparameters (from Table 1)

| Parameter | Stage 1a | Stage 1b | Stage 2 | Stage 3 | Stage 4 |
|-----------|----------|----------|---------|---------|---------|
| LR (final) | 3e-4 | 3e-4 | 3e-5 | 3e-5 | 1e-5 |
| LR schedule | Constant | Constant | Cosine | Constant | Cosine |
| Weight decay | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| Grad clip | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Optimizer | AdamW (β1=0.9, β2=0.95) | | | | |
| Batch (tokens) | 4M→8M | 8M | 8M | 8M | 8M |
| Recurrent steps | 8 | 4 | 4 | 4 | 4 |
| β (KL) | 0.1 | 0.1 | 0.05 | 0.05 | 0.05 |
| RoPE base | 10K | 10K | 40K | 1M | 1M |

### 5.2 Key Stability Insights

1. **Reduce recurrent steps from 8→4** early on if training becomes unstable (loss spikes, gradient oscillations)
2. **Progressive batch size scaling** (4M → 8M tokens) for more stable gradients
3. **Reduce β from 0.1 → 0.05** in later stages to allow the model freedom to learn depth patterns
4. **LoopLMs need smaller learning rates** than standard transformers of the same parameter count
5. Use **sandwich normalization** (RMSNorm before attention AND FFN) — critical for recurrent stability

### 5.3 Upcycling (1.4B → 2.6B)

After Stage 1a, the 2.6B model is created by duplicating the 24 layers to 48 layers (layer duplication). The recurrent/shared-weight nature makes this smoother than standard transformer upcycling.

### 5.4 SFT Configuration

- 2 epochs, max sequence length 32K
- Adam optimizer, LR 2e-5, β=(0.9, 0.95), cosine decay
- Framework: LlamaFactory

---

## Phase 6: Evaluation

### 6.1 Base Model Benchmarks

Use lm-eval-harness and evalplus:

| Benchmark | Setting |
|-----------|---------|
| MMLU | logprobs, 5-shot |
| MMLU-Pro | strict match, 5-shot CoT |
| BBH | strict match, 3-shot CoT |
| ARC-C | logprobs, 25-shot |
| HellaSwag | logprobs, 10-shot |
| Winogrande | logprobs, 5-shot |
| GSM8K | strict match, 3-shot CoT |
| MATH500 | strict match, 5-shot CoT |
| HumanEval/+ | pass@1 |
| MBPP/+ | pass@1 |

**Key targets (1.4B R4):** MMLU 67.35, BBH 71.02, GSM8K 78.92, MATH500 82.40

### 6.2 Reasoning Model Benchmarks

AIME 2024/2025, OlympiadBench, GPQA, SuperGPQA, BeyondAIME, HLE — all with temp=1.0, top_p=0.7, LLM-as-judge protocol.

### 6.3 Recurrent Depth Analysis

Evaluate at each step T=1 through T=8 to verify:
- Performance improves up to trained depth (T=4)
- Graceful degradation at extrapolated depths (T=5-8)
- Safety improves monotonically even beyond training depth

### 6.4 Early Exit Comparison

Compare on MMLU:
1. Static exit (fixed step)
2. Hidden state difference threshold
3. Untrained ponder gate (from pre-training)
4. Trained ponder gate (after Stage II adaptive training)

Plot accuracy vs. average exit round.

---

## Phase 7: Mechanistic Analysis (Understanding LoopLM Gains)

### 7.1 Knowledge Capacity (Capo Task)

- Generate synthetic biographies (bioS(N)) with N=20K-500K individuals
- Train small GPT-2-style models (1M-40M params) with and without looping
- Measure bits of knowledge stored
- **Expected result:** ~2 bits/parameter for both looped and non-looped (looping does NOT increase knowledge capacity)

### 7.2 Knowledge Manipulation (Mano Task)

- Modular arithmetic on tree structures (prefix notation, mod 23)
- Compare base (k⊗1) vs. loop (k⊗12/k) models
- **Expected result:** Looped models consistently outperform iso-parameter non-looped models

### 7.3 Multi-hop QA

- Synthetic facts with 3-hop questions following Yao et al. (2025)
- Compare sample efficiency: how many QA pairs needed for 100% accuracy
- **Expected result:** Looped models learn with fewer samples

### 7.4 Faithfulness Analysis

- Train linear probes on hidden states at each layer to predict per-step answers
- Use Quora Question Pairs dataset (ambiguous examples)
- Compute step-by-step agreement matrix
- **Expected result:** Adjacent steps show meaningful disagreement (unlike CoT models where the answer is decided before reasoning)

---

## Phase 8: Safety Evaluation

- Evaluate on HEx-PHI (330 examples, 11 prohibited categories)
- Use GPT-4o as judge, harmfulness score 1-5
- PCA on hidden representations to visualize benign vs. harmful prompt separation
- **Expected result:** Safety improves with more recurrent steps, even when extrapolating beyond training depth

---

## Implementation Priorities & Suggested Timeline

### Milestone 1: Proof of Concept (2-3 weeks)
- [ ] Implement LoopLM architecture (small scale: ~100M params, 6 layers, 4 recurrent steps)
- [ ] Implement entropy-regularized training objective
- [ ] Train on FineWeb-Edu subset (~20B tokens)
- [ ] Verify that performance improves with recurrent depth on a few benchmarks (ARC, HellaSwag)

### Milestone 2: Training Pipeline (2-3 weeks)
- [ ] Implement multi-stage training (at minimum Stages 1+2)
- [ ] Implement progressive batch size scaling
- [ ] Implement RoPE base frequency adjustment across stages
- [ ] Implement β annealing (0.1 → 0.05)
- [ ] Add KV cache sharing for efficient inference

### Milestone 3: Scale Up (3-4 weeks)
- [ ] Scale to 1.4B parameters, 24 layers
- [ ] Implement upcycling to 2.6B (48 layers via layer duplication)
- [ ] Full multi-stage training on larger data budget
- [ ] Implement Stage II adaptive gate training

### Milestone 4: SFT & Evaluation (2-3 weeks)
- [ ] Reasoning SFT with OpenThoughts3 + AceReason data
- [ ] Full benchmark evaluation suite
- [ ] Early exit analysis and comparison

### Milestone 5: Analysis & Ablations (2-3 weeks)
- [ ] Capo task (knowledge capacity)
- [ ] Mano task (knowledge manipulation)
- [ ] Multi-hop QA (sample efficiency)
- [ ] Faithfulness probing
- [ ] Safety evaluation

---

## Key Libraries & Infrastructure

| Component | Recommended Tool |
|-----------|-----------------|
| Pre-training framework | torchtitan or flame (paper uses flame built on torchtitan) |
| Model implementation | PyTorch (custom transformer with looping) |
| Optimizer | AdamW from PyTorch |
| SFT | LlamaFactory |
| Evaluation | lm-eval-harness, evalplus |
| Inference | vLLM (needs custom modifications for variable-depth computation) |
| Distributed training | FSDP or tensor parallelism via torchtitan |

---

## Critical Implementation Details (Easy to Miss)

1. **Sandwich normalization** — RMSNorm before BOTH attention and FFN. Not just pre-norm. This is essential for recurrent stability.

2. **The exit gate runs at every step during training** — you compute all T_max losses and weight them by p_φ(t|x). You don't actually "exit early" during training.

3. **Entropy regularization uses a uniform prior** — the paper shows this outperforms geometric priors (Appendix A). Don't use PonderNet's geometric prior.

4. **Gradients compound through loops** — this is why stability is harder. The paper explicitly notes reducing recurrent steps from 8→4 to stabilize training.

5. **KV cache during prefill vs. decode is different** — you need per-step KV caches during prefill but can share during decode.

6. **The 2.6B model shares the same hidden dimension (2048) as the 1.4B** — it just has 48 layers vs. 24. Both use the same recurrent step count.

7. **Stage 1a uses 8 recurrent steps, Stage 1b onwards uses 4** — this transition is driven by stability, not performance.

8. **RL did not work well** due to vLLM/SGLang incompatibility with variable-depth computation. The released models are SFT-only.

---

## Resources

- **Paper:** arXiv:2510.25741v4
- **Project page & models:** http://ouro-llm.github.io
- **Key references for implementation:**
  - Universal Transformer (Dehghani et al., 2018) — original looped transformer
  - PonderNet (Banino et al., 2021) — ELBO for dynamic halting
  - Geiping et al. (2025) — recurrent depth with sandwich normalization
