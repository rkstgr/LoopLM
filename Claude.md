# CLAUDE.md — LoopLM Implementation Guide

## Project Context
We are reimplementing the paper "Scaling Latent Reasoning via Looped Language Models" (Ouro, arXiv:2510.25741v4).
The paper PDF is at `paper/looplm.pdf`. Reference it when you need architectural details.

---

## Repository Structure

```
looplm/
├── CLAUDE.md                    # This file
├── paper/
│   └── looplm.pdf               # The paper for reference
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── config.py            # Model configs (small, 1.4B, 2.6B)
│   │   ├── transformer.py       # Base transformer block (RoPE, SwiGLU, sandwich norm)
│   │   ├── looplm.py            # LoopLM wrapper (recurrence, exit gate, multi-step loss)
│   │   └── rope.py              # RoPE implementation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── objectives.py        # Entropy-regularized loss, adaptive gate loss
│   │   ├── trainer.py           # Training loop with multi-stage support
│   │   └── data.py              # Data loading and tokenization
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── early_exit.py        # Q-exit, hidden state diff, static exit
│   │   └── kv_cache.py          # KV cache sharing strategies
│   └── analysis/
│       ├── __init__.py
│       ├── capo.py              # Knowledge capacity experiments
│       ├── mano.py              # Knowledge manipulation experiments
│       └── faithfulness.py      # Linear probing, agreement matrices
├── configs/
│   ├── small.yaml               # ~100M param config for prototyping
│   ├── medium.yaml              # ~400M param config
│   ├── ouro_1_4b.yaml           # Full 1.4B config
│   └── ouro_2_6b.yaml           # Full 2.6B config
├── scripts/
│   ├── train.py                 # Entry point for training
│   ├── evaluate.py              # Entry point for evaluation
│   └── analyze.py               # Entry point for analysis experiments
├── tests/
│   ├── test_model.py            # Unit tests for architecture
│   ├── test_objectives.py       # Unit tests for loss functions
│   ├── test_gate.py             # Unit tests for exit gate logic
│   └── test_inference.py        # Tests for early exit, KV cache
├── pyproject.toml
└── README.md
```

---

## Core Architecture Rules
- This is a decoder-only transformer with **shared-weight recurrence** (LoopLM)
- A stack of N layers is applied T times (recurrent steps), reusing the same weights
- Each recurrent step produces an LM head output and a loss
- An exit gate at each step predicts the probability of stopping
- **Sandwich normalization**: RMSNorm before BOTH attention AND FFN — this is critical for recurrent stability
- Use SwiGLU for FFN, RoPE for positional encoding
- Vocabulary: 49,152 tokens (SmolLM2 tokenizer)

## Implementation Principles
1. **Test-driven**: Write tests before or alongside every module. Run `pytest tests/ -x` frequently.
2. **Small first**: Default config is ~100M params (6 layers, hidden 768, 12 heads). Only scale up after correctness is verified.
3. **Modular**: Keep the loop mechanism, exit gate, and loss computation cleanly separated from the base transformer.
4. **Numerical checks**: After implementing any component, verify shapes and gradient flow with a small forward+backward pass.

## File Conventions
- Model code in `src/model/`
- Training logic in `src/training/`
- Inference logic in `src/inference/`
- Analysis experiments in `src/analysis/`
- Configs in `configs/` as YAML
- All entry points in `scripts/`
- Tests in `tests/` mirroring `src/` structure

---

## Key Implementation Details (from paper)

### Architecture (Section 3.1, Figure 3)
- The looped forward pass: `F^(t)(x) = lmhead ∘ M_L^t ∘ emb(x)` where M_L is the full layer stack applied t times
- The exit gate: `λ_t = sigmoid(Linear(h^(t)))` where h^(t) is the final-layer hidden state at step t
- Survival probability: `S_t = ∏_{j=1}^{t} (1 - λ_j)`
- Exit distribution: `p(t|x) = λ_t · S_{t-1}` for t < T_max, `p(T_max|x) = S_{T_max-1}`
- The gate runs in parallel with the LM head at each step
- **The exit gate runs at every step during training** — all T_max losses are computed and weighted by p(t|x). No actual early exit happens during training.
- Hidden states from the output of step t become the input to step t+1 (implicit recurrence — no feedback token)

### Training Objective — Stage I (Section 3.3, Equation 4)
```
L = Σ_{t=1}^{T_max} p(t|x) · L^(t) - β · H(p(·|x))
```
- L^(t) is standard cross-entropy at step t
- Use **uniform prior** (NOT geometric) — see Appendix A for empirical validation
- β starts at 0.1, reduces to 0.05 in later stages
- This is equivalent to an ELBO with uniform prior over exit steps
- Gradients flow through both LM parameters and gate parameters jointly

### Training Objective — Stage II (Section 3.4, Equation 6)
- Freeze LM parameters, train only exit gate
- Detached improvement: `I^(t) = max(0, L^(t-1)_detached - L^(t)_detached)`
- Ideal continuation: `w^(t) = sigmoid(50 * (I^(t) - 0.005))`
- BCE between predicted continuation `(1 - λ^(t))` and target `w^(t)`
- Average across steps t=2..T_max

### Inference (Section 3.2)
- Q-exit: accumulate CDF of exit probs, stop when CDF ≥ threshold q
- KV cache: separate per step during prefill, reuse last-step only during decode (4x savings)
- For initial testing, just always run all T_max steps (static inference) — add early exit later

### Model Configs (Table 2)

| Model     | Params | Layers | Hidden Size | Attention | FFN    |
|-----------|--------|--------|-------------|-----------|--------|
| Ouro 1.4B | 1.4B   | 24     | 2048        | MHA       | SwiGLU |
| Ouro 2.6B | 2.6B   | 48     | 2048        | MHA       | SwiGLU |

Both use 4 recurrent steps at inference. The 2.6B shares the same hidden dimension (2048) — it just has 48 layers vs. 24.

**Prototyping:** Start with ~100M params (6 layers, hidden 768, 12 heads, 4 recurrent steps).

### Training Schedule (Table 1)

| Parameter  | Stage 1a         | Stage 1b | Stage 2         | Stage 3   | Stage 4         |
|------------|------------------|----------|-----------------|-----------|-----------------|
| LR         | 3e-4             | 3e-4     | 3e-5            | 3e-5      | 1e-5            |
| LR schedule| Constant         | Constant | Cosine          | Constant  | Cosine          |
| Batch      | 4M→8M tokens     | 8M       | 8M              | 8M        | 8M              |
| Recurrent steps | 8           | 4        | 4               | 4         | 4               |
| β (KL)     | 0.1              | 0.1      | 0.05            | 0.05      | 0.05            |
| RoPE base  | 10K              | 10K      | 40K             | 1M        | 1M              |
| Seq length | 4K               | 4K       | 16K             | 64K       | 32K             |
| Tokens     | 3T               | 3T       | 1.4T            | 20B       | 300B            |

- Optimizer: AdamW, β1=0.9, β2=0.95, weight_decay=0.1, grad_clip=1.0
- Stage 1a uses 8 recurrent steps, Stage 1b onwards uses 4 — this transition is driven by stability

**Stability insights:**
1. Reduce recurrent steps from 8→4 early if loss spikes or gradient oscillations occur
2. Progressive batch size scaling (4M → 8M) for more stable gradients
3. Reduce β from 0.1 → 0.05 in later stages
4. LoopLMs need smaller LRs than standard transformers of the same parameter count

**Upcycling (1.4B → 2.6B):** After Stage 1a, duplicate the 24 layers to 48 layers. The recurrent/shared-weight nature makes this smoother than standard transformer upcycling.

**SFT config:** 2 epochs, max seq len 32K, Adam LR 2e-5, β=(0.9, 0.95), cosine decay. Framework: LlamaFactory.

**RL note:** RL did not work well due to vLLM/SGLang incompatibility with variable-depth computation. Released models are SFT-only.

---

## Data Pipeline

### Pre-training Data (Stages 1-4)

| Stage | Data | Tokens | Seq Length |
|-------|------|--------|------------|
| 1a/1b | Nemotron-CC, MAP-CC, OpenCoder, MegaMath | 6T dataset, 3T+3T used | 4K |
| 2 | HQ Nemotron-CC, Math, Code, SFT | 1.4T | 16K |
| 3 | ProLong-64K | 20B | 64K |
| 4 | 20+ SFT datasets in ChatML | 300B effective | 32K |

**For a scaled-down replication:**
- Use FineWeb-Edu (1.3T tokens) or a subset for Stage 1
- Use OpenCoder-Annealing + MegaMath-HQ for Stage 2
- Skip Stage 3 initially
- Use a mix of open SFT data for Stage 4

### SFT Data (~8.3M examples)
- Math: OpenThoughts3, AceReason-1.1-SFT (3.5M)
- Code: AceReason, OpenCodeReasoning, Llama-Nemotron (3.2M)
- Science: OpenThoughts3, Llama-Nemotron (808K)
- Chat: OO1-Chat-747K, DeepWriting-20K (767K)

---

## Evaluation

### Base Model Benchmarks (lm-eval-harness + evalplus)

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

### Reasoning Model Benchmarks
AIME 2024/2025, OlympiadBench, GPQA, SuperGPQA, BeyondAIME, HLE — all with temp=1.0, top_p=0.7, LLM-as-judge.

### Recurrent Depth Analysis
Evaluate at each step T=1 through T=8 to verify:
- Performance improves up to trained depth (T=4)
- Graceful degradation at extrapolated depths (T=5-8)
- Safety improves monotonically even beyond training depth

### Early Exit Comparison
Compare on MMLU: (1) static exit, (2) hidden state diff threshold, (3) untrained ponder gate, (4) trained ponder gate. Plot accuracy vs. average exit round.

---

## Mechanistic Analysis

### Capo Task (Knowledge Capacity)
- Synthetic biographies (bioS(N)) with N=20K–500K individuals
- Train 1M–40M param models with loop=1 and loop=4
- **Expected:** ~2 bits/parameter for both — looping does NOT increase knowledge capacity

### Mano Task (Knowledge Manipulation)
- Modular arithmetic on tree structures (prefix notation, mod 23)
- Compare base (k⊗1) vs. loop (k⊗12/k) models
- **Expected:** Looped models consistently outperform iso-parameter non-looped models

### Multi-hop QA
- Synthetic facts with 3-hop questions (Yao et al., 2025)
- **Expected:** Looped models learn with fewer samples

### Faithfulness Analysis
- Linear probes on hidden states at each layer to predict per-step answers
- Use Quora Question Pairs (ambiguous examples), compute step-by-step agreement matrix
- **Expected:** Adjacent steps show meaningful disagreement (unlike CoT where answer is decided before reasoning)

### Safety Evaluation
- HEx-PHI (330 examples, 11 prohibited categories), GPT-4o as judge (harmfulness 1-5)
- PCA on hidden representations for benign vs. harmful prompt separation
- **Expected:** Safety improves with more recurrent steps, even beyond training depth

---

## Common Pitfalls to Avoid
1. DON'T use pre-norm only — must be **sandwich norm** (RMSNorm before attention AND before FFN)
2. DON'T let the exit gate collapse to always using T_max — monitor entropy of p(t|x)
3. DON'T share KV cache during prefill — only during decode phase
4. DON'T forget to detach losses in Stage II gate training
5. DON'T use a geometric prior for the KL term — use uniform
6. DON'T start with 8 recurrent steps — use 4 for prototyping
7. DON'T use the first-step KV cache for decode reuse — use last-step (first-step collapses performance: 78.92 → 18.73 on GSM8K)
8. Gradients compound through loops — this is why stability is harder

## When Making Changes
- Always run existing tests after changes: `pytest tests/ -x`
- For any new module, add a corresponding test file
- Use type hints throughout
- Keep functions focused — prefer many small functions over monolithic ones
- Log per-step losses, entropy, and average exit step during training (key diagnostics)

---

## Dependencies & Infrastructure

### Python Dependencies
```bash
pip install torch torchvision torchaudio  # PyTorch 2.x
pip install transformers                   # Tokenizer, comparison models
pip install datasets                       # HuggingFace datasets
pip install wandb                          # Experiment tracking
pip install lm-eval                        # lm-eval-harness for benchmarks
pip install evalplus                       # Code evaluation
pip install pyyaml pytest
pip install flash-attn                     # Optional, for speed
```

### Hardware Considerations

| Phase | Minimum GPU | Recommended |
|-------|------------|-------------|
| Prototyping (~100M, 20B tokens) | 1x A100 40GB | 1x A100 80GB |
| Medium scale (~400M) | 2-4x A100 | 4x A100 80GB |
| Full 1.4B | 8x A100 80GB | 8x H100 |
| Full 2.6B | 16x A100 80GB | 16x H100 |

The looping adds compute but NOT memory (weights are shared).

### Key Libraries

| Component | Tool |
|-----------|------|
| Pre-training framework | torchtitan or flame |
| Optimizer | AdamW from PyTorch |
| SFT | LlamaFactory |
| Evaluation | lm-eval-harness, evalplus |
| Inference | vLLM (needs custom mods for variable-depth) |
| Distributed training | FSDP or tensor parallelism via torchtitan |

---

## Implementation Milestones

### Milestone 1: Proof of Concept
- [ ] LoopLM architecture (~100M params, 6 layers, 4 recurrent steps)
- [ ] Entropy-regularized training objective
- [ ] Train on FineWeb-Edu subset (~20B tokens)
- [ ] Verify performance improves with recurrent depth on ARC, HellaSwag

### Milestone 2: Training Pipeline
- [ ] Multi-stage training (at minimum Stages 1+2)
- [ ] Progressive batch size scaling
- [ ] RoPE base frequency adjustment across stages
- [ ] β annealing (0.1 → 0.05)
- [ ] KV cache sharing for efficient inference

### Milestone 3: Scale Up
- [ ] Scale to 1.4B parameters, 24 layers
- [ ] Upcycling to 2.6B (48 layers via layer duplication)
- [ ] Full multi-stage training on larger data budget
- [ ] Stage II adaptive gate training

### Milestone 4: SFT & Evaluation
- [ ] Reasoning SFT with OpenThoughts3 + AceReason data
- [ ] Full benchmark evaluation suite
- [ ] Early exit analysis and comparison

### Milestone 5: Analysis & Ablations
- [ ] Capo task, Mano task, Multi-hop QA, Faithfulness probing, Safety evaluation

---

## Sprint Implementation Guide

Use these prompts in sequence. Each builds on the previous.

### Sprint 1: Foundation

**1.1 — Config system**
```
Create src/model/config.py with a LoopLMConfig dataclass. Include fields for:
vocab_size, hidden_size, num_layers, num_heads, intermediate_size (for SwiGLU),
max_seq_len, max_recurrent_steps, rope_base, dropout, beta_kl.
Add a class method `small()` that returns a ~100M param config
(6 layers, hidden 768, 12 heads, intermediate 2048, 4 recurrent steps).
Add a test in tests/test_model.py that instantiates the config and checks parameter
count estimation.
```

**1.2 — RoPE**
```
Implement RoPE (Rotary Position Embeddings) in src/model/rope.py.
Support configurable base frequency (default 10000, but needs to change to 40K/1M
in later training stages). Include precomputed cos/sin cache.
Add a test that verifies the rotation is applied correctly to Q and K tensors.
```

**1.3 — Transformer block**
```
Implement the base transformer block in src/model/transformer.py.
Requirements:
- Multi-head attention with RoPE (from rope.py)
- SwiGLU FFN: gate = silu(W_gate · x), out = W_down · (gate * (W_up · x))
- Sandwich normalization: RMSNorm before attention AND before FFN
- Residual connections around both attention and FFN
Add a test that does a forward pass with random input and checks output shape matches
input shape, and that gradients flow through all parameters.
```

**1.4 — LoopLM model**
```
Implement the LoopLM model in src/model/looplm.py using config.py and transformer.py.
Requirements:
- Token embedding + stack of N TransformerBlock layers
- Looping: apply the layer stack T times (T from 1 to max_recurrent_steps)
- LM head (linear projection to vocab_size) applied after each recurrent step
- Exit gate: Linear + sigmoid on final hidden state at each step
- Forward returns: list of logits per step, list of exit probabilities per step
Test: verify that with T=1 it behaves like a standard transformer.
Test: verify that all T steps produce valid logits of shape (batch, seq_len, vocab_size).
Test: verify exit probabilities sum to 1 across steps.
```

### Sprint 2: Training Objective

**2.1 — Entropy-regularized loss**
```
Implement the Stage I training objective in src/training/objectives.py.
Function signature: compute_looplm_loss(logits_per_step, exit_probs_per_step, targets, beta=0.1)
- Compute cross-entropy loss L^(t) at each step t
- Compute exit distribution p(t|x) from the exit probabilities
- Expected task loss = Σ p(t|x) · L^(t)
- Total loss = expected task loss - β * entropy
- Return total loss and a dict of diagnostics (per-step losses, entropy, avg exit step)
Add tests: verify entropy term pushes distribution toward uniform, verify gradients flow
to both LM params and gate params.
```

**2.2 — Adaptive gate loss (Stage II)**
```
Implement the Stage II adaptive gate training loss in src/training/objectives.py.
Function: compute_adaptive_gate_loss(logits_per_step, exit_lambdas, targets, k=50.0, gamma=0.005)
- Compute detached per-step losses (no gradient through LM)
- Improvement signal I^(t) = max(0, L^(t-1)_detached - L^(t)_detached)
- Ideal continuation label w^(t) = sigmoid(k * (I^(t) - gamma))
- BCE loss between (1 - lambda^(t)) and w^(t), averaged across steps t=2..T_max
Test: when step t improves a lot, gate should learn to continue.
Test: when no improvement, gate should learn to exit.
```

**2.3 — Basic training loop**
```
Implement a basic training loop in src/training/trainer.py.
- Takes a LoopLMConfig, creates model
- AdamW with β1=0.9, β2=0.95, weight_decay=0.1, grad_clip=1.0
- Loads data from HuggingFace datasets
- Logs loss, per-step losses, entropy, avg exit step to wandb
- Saves checkpoints, single GPU for now
Add a smoke test that trains for 10 steps on random data and verifies loss decreases.
```

### Sprint 3: Validation

**3.1 — Small-scale training run**
```
Create scripts/train.py. Run with small config (~100M params) on a subset of
FineWeb-Edu or wikitext-103 for 1000 steps. After training, verify:
1. Total loss decreased
2. Per-step losses all decreased
3. Entropy of exit distribution stayed reasonable (not collapsed)
4. Later recurrent steps have lower loss than earlier ones
Print a summary table of step-wise losses at the end.
```

**3.2 — Evaluation harness integration**
```
Create scripts/evaluate.py that loads a trained LoopLM checkpoint and evaluates
using lm-eval-harness. Create a wrapper class that runs full T_max recurrent steps
and returns logits from the final step. Start with ARC-Easy and HellaSwag.
Verify that a model with T=4 scores higher than T=1 on these benchmarks.
```

**3.3 — Recurrent depth analysis**
```
Add to scripts/evaluate.py the ability to evaluate at each recurrent step T=1..T_max.
Output a table showing score vs. recurrent depth.
Expected: monotonically increasing up to T_max, with diminishing returns.
```

### Sprint 4: Inference & Efficiency

**4.1 — Early exit strategies**
```
Implement three early exit strategies in src/inference/early_exit.py:
1. StaticExit(step): always exit at a fixed step
2. HiddenStateDiffExit(threshold): exit when ||h_t - h_{t-1}|| < epsilon
3. QExit(q_threshold): exit when CDF of exit probs exceeds q
Test: QExit with q=0 should always exit at step 1. QExit with q=1 should use all steps.
```

**4.2 — KV cache implementation**
```
Implement KV cache management in src/inference/kv_cache.py.
- During prefill: maintain separate KV caches for each recurrent step
- During decode: option to reuse only the last step's KV cache (4x memory reduction)
- Support "full cache" and "last-step reuse" modes
Test: verify last-step reuse produces nearly identical outputs to full cache.
```

### Sprint 5: Multi-Stage Training

**5.1 — Multi-stage trainer**
```
Extend src/training/trainer.py to support the multi-stage pipeline:
- Stage 1a: constant LR, 8 recurrent steps, β=0.1
- Stage 1b: constant LR, 4 recurrent steps, β=0.1
- Stage 2: cosine LR, 4 steps, β=0.05, RoPE base 40K
- Support resuming from any stage checkpoint, changing recurrent steps and RoPE base mid-training.
```

Manual test:
```bash
uv run scripts/train.py --stage1a-steps 8 --stage1b-steps 8 --batch-size 2 --seq-len 64 --log-every 1
```

**5.2 — Stage II gate fine-tuning**
```
Add a gate_finetune mode to the trainer that:
- Loads a pre-trained checkpoint, freezes all LM parameters
- Trains only exit gate parameters using compute_adaptive_gate_loss
- Evaluates early exit efficiency before and after on a validation set
```

### Sprint 6: Analysis Experiments

**6.1 — Capo task** — See `src/analysis/capo.py`: synthetic biographies, 1M-40M param models, bits of knowledge stored.

**6.2 — Mano task** — See `src/analysis/mano.py`: modular arithmetic in prefix notation, mod 23, looped vs. non-looped accuracy.

### Sprint 7: Scale Up

**7.1 — 1.4B config**
```
Set up 24 layers, hidden 2048, 4 recurrent steps. Data pipeline for Stage 1 using
Nemotron-CC or FineWeb-Edu streamed from HuggingFace. Configure FSDP.
Batch 4M tokens, sequence length 4K.
```

**7.2 — Upcycling 1.4B → 2.6B**
```
Take a trained 24-layer 1.4B checkpoint and create a 48-layer 2.6B model by
duplicating layers. Verify the upcycled model produces reasonable outputs before
continuing training.
```

---

## Debugging Tips

**Gate collapse (entropy → 0):**
```
The model is using step 4 for 98% of tokens. What changes to β or training setup
could fix this? Refer to Section 3.3 of the paper.
```

**Loss not decreasing:**
```
Training loss not decreasing after 500 steps. Per-step losses are flat.
Entropy is 0.01 (collapsed). Diagnose the issue.
```

**Gradient flow check:**
```
Run a gradient flow check: for each named parameter, print the mean absolute
gradient after one training step. Identify parameters with zero or exploding gradients.
```

**Performance profiling:**
```
Profile the training loop for one step. Identify the bottleneck. Verify embeddings
are not being recomputed at each recurrent step.
```

**Core comparison:**
```
Train two models with identical configs: one with 4 recurrent steps (LoopLM) and one
with 4x the layers and no looping (standard transformer, same FLOP budget).
Compare on ARC-C and GSM8K after 20B tokens. This reproduces the core claim of the paper.
```
