# LoopLM Implementation: Setup & Claude Code Workflow

## Part 1: Project Setup

### 1.1 Repository Structure

```
looplm/
├── CLAUDE.md                    # Instructions for Claude Code (see Part 2)
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

### 1.2 Environment Setup

```bash
# Create project
mkdir looplm && cd looplm
git init

# Python environment
python -m venv .venv
source .venv/bin/activate

# Core dependencies
pip install torch torchvision torchaudio  # PyTorch 2.x
pip install transformers                   # For tokenizer, comparison models
pip install datasets                       # HuggingFace datasets
pip install wandb                          # Experiment tracking
pip install lm-eval                        # lm-eval-harness for benchmarks
pip install evalplus                       # Code evaluation
pip install pyyaml                         # Config files
pip install pytest                         # Testing

# Optional but recommended
pip install flash-attn                     # Flash Attention 2
pip install torchtitan                     # Distributed training (later)
```

### 1.3 Hardware Considerations

| Phase | Minimum GPU | Recommended |
|-------|------------|-------------|
| Prototyping (~100M, 20B tokens) | 1x A100 40GB | 1x A100 80GB |
| Medium scale (~400M) | 2-4x A100 | 4x A100 80GB |
| Full 1.4B | 8x A100 80GB | 8x H100 |
| Full 2.6B | 16x A100 80GB | 16x H100 |

For prototyping, a single high-end GPU is sufficient. The looping adds compute but NOT memory (weights are shared).

---

## Part 2: CLAUDE.md — Instructions for Claude Code

Place this file at the root of your repo. Claude Code reads it automatically.

```markdown
# CLAUDE.md — LoopLM Implementation Guide

## Project Context
We are reimplementing the paper "Scaling Latent Reasoning via Looped Language Models" (Ouro).
The paper PDF is at `paper/looplm.pdf`. Reference it when you need architectural details.

## Core Architecture Rules
- This is a decoder-only transformer with **shared-weight recurrence** (LoopLM)
- A stack of N layers is applied T times (recurrent steps), reusing the same weights
- Each recurrent step produces an LM head output and a loss
- An exit gate at each step predicts the probability of stopping
- **Sandwich normalization**: RMSNorm before BOTH attention AND FFN — this is critical
- Use SwiGLU for FFN, RoPE for positional encoding

## Implementation Principles
1. **Test-driven**: Write tests before or alongside every module. Run `pytest tests/` frequently.
2. **Small first**: Default config is ~100M params (6 layers, hidden 768, 4 heads). Only scale up after correctness is verified.
3. **Modular**: Keep the loop mechanism, exit gate, and loss computation cleanly separated from the base transformer.
4. **Numerical checks**: After implementing any component, verify shapes and gradient flow with a small forward+backward pass.

## File Conventions
- Model code in `src/model/`
- Training logic in `src/training/`
- Inference logic in `src/inference/`
- Configs in `configs/` as YAML
- All entry points in `scripts/`
- Tests in `tests/`

## Key Implementation Details (from paper)

### Architecture
- The exit gate is: `λ_t = sigmoid(Linear(h^(t)))` where h^(t) is the final-layer hidden state at step t
- Exit distribution: `p(t|x) = λ_t · S_{t-1}` where `S_t = ∏(1 - λ_j)` for j=1..t
- Remaining mass goes to T_max: `p(T_max|x) = S_{T_max - 1}`

### Training Objective (Stage I)
```
L = Σ_t p(t|x) · L^(t) - β · H(p(·|x))
```
- Use **uniform prior** (NOT geometric). β starts at 0.1, reduces to 0.05 in later stages.
- L^(t) is standard cross-entropy at step t.

### Training Objective (Stage II — gate only)
- Freeze LM, train only gate parameters
- Improvement signal: `I^(t) = max(0, L^(t-1)_detached - L^(t)_detached)`
- Target: `w^(t) = sigmoid(50 * (I^(t) - 0.005))`
- Loss: binary cross-entropy between `(1 - λ^(t))` and `w^(t)`

### Inference
- Q-exit: accumulate CDF of exit probabilities, stop when CDF ≥ threshold q
- KV cache: separate caches per step during prefill, reuse last-step cache during decode

## When Making Changes
- Always run existing tests after changes: `pytest tests/ -x`
- For any new module, add a corresponding test file
- Use type hints throughout
- Keep functions focused — prefer many small functions over monolithic ones

## Common Pitfalls to Avoid
1. DON'T use pre-norm only — must be sandwich norm (RMSNorm before attention AND before FFN)
2. DON'T let the exit gate collapse — verify entropy of p(t|x) stays reasonable during training
3. DON'T share KV cache during prefill — only during decode
4. DON'T forget to detach losses in Stage II gate training
5. DON'T use a geometric prior for the KL term — use uniform
```

---

## Part 3: Iterative Implementation Prompts for Claude Code

Use these prompts in sequence. Each builds on the previous. Copy-paste them one at a time.

### Sprint 1: Foundation (Day 1-2)

**Prompt 1.1 — Config system**
```
Create src/model/config.py with a LoopLMConfig dataclass. Include fields for:
vocab_size, hidden_size, num_layers, num_heads, intermediate_size (for SwiGLU),
max_seq_len, max_recurrent_steps, rope_base, dropout, beta_kl.
Add a class method `small()` that returns a ~100M param config
(6 layers, hidden 768, 12 heads, intermediate 2048, 4 recurrent steps).
Add a test in tests/test_model.py that instantiates the config and checks parameter
count estimation.
```

**Prompt 1.2 — RoPE**
```
Implement RoPE (Rotary Position Embeddings) in src/model/rope.py.
Support configurable base frequency (default 10000, but needs to change to 40K/1M
in later training stages). Include precomputed cos/sin cache.
Add a test that verifies the rotation is applied correctly to Q and K tensors.
```

**Prompt 1.3 — Transformer block**
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

**Prompt 1.4 — LoopLM model**
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

### Sprint 2: Training Objective (Day 3-4)

**Prompt 2.1 — Entropy-regularized loss**
```
Implement the Stage I training objective in src/training/objectives.py.
Function signature: compute_looplm_loss(logits_per_step, exit_probs_per_step,
targets, beta=0.1)
Requirements:
- Compute cross-entropy loss L^(t) at each step t
- Compute exit distribution p(t|x) from the exit probabilities
- Expected task loss = Σ p(t|x) · L^(t)
- Entropy regularization = -β · H(p(·|x))
- Total loss = expected task loss - β * entropy
- Return total loss and a dict of diagnostics (per-step losses, entropy, avg exit step)
Add comprehensive tests:
- Verify loss decreases with random logits vs. logits matching targets
- Verify entropy term pushes distribution toward uniform
- Verify gradient flows to both LM params and gate params
```

**Prompt 2.2 — Adaptive gate loss (Stage II)**
```
Implement the Stage II adaptive gate training loss in src/training/objectives.py.
Function: compute_adaptive_gate_loss(logits_per_step, exit_lambdas, targets, k=50.0, gamma=0.005)
Requirements:
- Compute detached per-step losses (no gradient through LM)
- Improvement signal I^(t) = max(0, L^(t-1)_detached - L^(t)_detached)
- Ideal continuation label w^(t) = sigmoid(k * (I^(t) - gamma))
- BCE loss between (1 - lambda^(t)) and w^(t)
- Average across steps t=2..T_max
Test: when step t improves a lot over t-1, the gate should learn to continue.
Test: when no improvement, the gate should learn to exit.
```

**Prompt 2.3 — Basic training loop**
```
Implement a basic training loop in src/training/trainer.py.
Use a simple approach first — no distributed, no multi-stage.
Requirements:
- Takes a LoopLMConfig, creates model
- AdamW optimizer with β1=0.9, β2=0.95, weight_decay=0.1, grad_clip=1.0
- Supports loading data from HuggingFace datasets (start with a small text dataset)
- Logs loss, per-step losses, entropy, avg exit step to wandb
- Saves checkpoints
- Single GPU for now
Add a smoke test that trains for 10 steps on random data and verifies loss decreases.
```

### Sprint 3: Validation (Day 5-7)

**Prompt 3.1 — Small-scale training run**
```
Create scripts/train.py as the entry point.
Set up a training run with the small config (~100M params) on a subset of
FineWeb-Edu (or wikitext-103 for quick testing). Train for 1000 steps.
Log to wandb. After training, verify:
1. Total loss decreased
2. Per-step losses all decreased
3. Entropy of exit distribution stayed reasonable (not collapsed)
4. Later recurrent steps have lower loss than earlier ones
Print a summary table of step-wise losses at the end.
```

**Prompt 3.2 — Evaluation harness integration**
```
Create scripts/evaluate.py that loads a trained LoopLM checkpoint and evaluates
it using lm-eval-harness.
Key challenge: lm-eval-harness expects a standard HF model interface.
Create a wrapper class that:
- Runs the full T_max recurrent steps
- Returns logits from the final step
- Properly handles the generate() interface for CoT tasks
Start by evaluating on ARC-Easy and HellaSwag (fast benchmarks).
Verify that a model with T=4 scores higher than T=1 on these benchmarks.
```

**Prompt 3.3 — Recurrent depth analysis**
```
Add to scripts/evaluate.py the ability to evaluate at each recurrent step T=1..T_max.
For each step, run the benchmark and record the score.
Output a table showing score vs. recurrent depth.
Expected: monotonically increasing up to T_max, with diminishing returns.
This is our key validation that the architecture is working correctly.
```

### Sprint 4: Inference & Efficiency (Day 8-10)

**Prompt 4.1 — Early exit strategies**
```
Implement three early exit strategies in src/inference/early_exit.py:
1. StaticExit(step): always exit at a fixed step
2. HiddenStateDiffExit(threshold): exit when ||h_t - h_{t-1}|| < epsilon
3. QExit(q_threshold): exit when CDF of exit probs exceeds q
Each should wrap a LoopLM model and provide a generate() method.
Test: QExit with q=0 should always exit at step 1.
Test: QExit with q=1 should always use all steps.
Test: HiddenStateDiffExit should use fewer steps on "easy" inputs.
```

**Prompt 4.2 — KV cache implementation**
```
Implement KV cache management in src/inference/kv_cache.py for LoopLM.
Requirements:
- During prefill: maintain separate KV caches for each recurrent step
- During decode: option to reuse only the last step's KV cache (4x memory reduction)
- Support both "full cache" and "last-step reuse" modes
- Integrate with the generate() function
Test: verify that last-step reuse produces nearly identical outputs to full cache
on a short generation task.
```

### Sprint 5: Multi-Stage Training (Day 11-14)

**Prompt 5.1 — Multi-stage trainer**
```
Extend src/training/trainer.py to support the multi-stage training pipeline:
- Stage 1a: constant LR, 8 recurrent steps, β=0.1
- Stage 1b: constant LR, 4 recurrent steps, β=0.1 (reduce from 8)
- Stage 2: cosine decay LR, 4 steps, β=0.05, increase RoPE base to 40K
- Stage 3: constant LR, increase seq len to 64K, RoPE base 1M
- Stage 4: cosine decay, high-quality SFT data
Support resuming from any stage checkpoint.
Support changing recurrent steps mid-training (8→4 transition).
Support changing RoPE base frequency (requires cache recomputation).
```

**Prompt 5.2 — Stage II gate fine-tuning**
```
Add a gate_finetune mode to the trainer that:
- Loads a pre-trained LoopLM checkpoint
- Freezes all LM parameters (embedding, transformer blocks, LM head)
- Only trains the exit gate parameters
- Uses the adaptive gate loss from objectives.py
- Runs for a configurable number of steps
- Evaluates early exit efficiency before and after on a validation set
```

### Sprint 6: Analysis Experiments (Day 15-18)

**Prompt 6.1 — Capo task (knowledge capacity)**
```
Implement the Capo knowledge capacity experiment in src/analysis/capo.py.
Requirements:
- Generate synthetic biographies bioS(N) with configurable N (20K-500K people)
- Each person has: name, gender, birth date, university, major, employer
- Train small GPT-2-style models (1M-40M params) with loop=1 and loop=4
- Measure bits of knowledge stored (cross-entropy on attribute tokens)
- Plot bits vs. parameters for both looped and non-looped
Expected result: ~2 bits/parameter regardless of looping.
```

**Prompt 6.2 — Mano task (knowledge manipulation)**
```
Implement the Mano knowledge manipulation task in src/analysis/mano.py.
Requirements:
- Generate modular arithmetic expressions in prefix notation, mod 23
- Variable expression lengths L=10, 16, 24
- Train base (k×1) and loop (k×12/k) models for k=2,3,6
- Also train a 12×1 iso-FLOP baseline
- Report test accuracy on maximum length expressions
Expected result: looped models outperform iso-parameter baselines at all sizes.
```

### Sprint 7: Scale Up (Day 19+)

**Prompt 7.1 — 1.4B config and data pipeline**
```
Set up the full 1.4B parameter configuration (24 layers, hidden 2048, MHA, SwiGLU)
with 4 recurrent steps. Set up the data pipeline for Stage 1 using
Nemotron-CC or FineWeb-Edu streamed from HuggingFace.
Configure distributed training with FSDP.
Set batch size to 4M tokens, sequence length 4K.
```

**Prompt 7.2 — Upcycling 1.4B → 2.6B**
```
Implement model upcycling: take a trained 24-layer 1.4B checkpoint and create a
48-layer 2.6B model by duplicating layers. Since weights are shared across recurrent
steps, this is straightforward layer duplication.
Verify the upcycled model produces reasonable outputs before continuing training.
```

---

## Part 4: Tips for Working with Claude Code on This Project

### General Workflow

1. **One prompt per logical unit.** Don't ask for the whole model at once. Ask for config, then RoPE, then transformer block, etc.

2. **Always ask for tests.** Include "add tests" in every prompt. Claude Code will write them and you can run them immediately.

3. **Review before moving on.** After each prompt, review the code, run tests, and fix issues before the next prompt.

4. **Use follow-up prompts for fixes:**
   ```
   The test_exit_gate test is failing because the probabilities don't sum to 1
   when T_max=1. Fix the edge case in looplm.py.
   ```

5. **Reference the paper when needed:**
   ```
   Looking at Equation 4 in the paper, I think the entropy term sign is wrong
   in our implementation. The paper subtracts β·H but we're adding it.
   Please check and fix.
   ```

### Debugging Prompts

```
The training loss is not decreasing after 500 steps. The per-step losses are:
step 1: 10.2, step 2: 10.1, step 3: 10.1, step 4: 10.1.
The entropy is 0.01 (collapsed). Diagnose the issue.
```

```
Run a gradient flow check: for each named parameter, print the mean absolute
gradient after one training step. Identify any parameters with zero or
exploding gradients.
```

```
The model is using step 4 for 98% of tokens. The exit distribution has
collapsed. What changes to the β parameter or training setup could fix this?
Refer to Section 3.3 of the paper.
```

### Performance Prompts

```
Profile the training loop for one step. Identify the bottleneck.
The looping should be the main cost — verify that we're not doing unnecessary
recomputations (e.g., recomputing embeddings at each step).
```

```
Implement gradient checkpointing for the recurrent steps to reduce memory usage.
Each recurrent step should be a checkpointed segment.
```

### Comparison Prompts

```
Train two models with identical configs except: one with 4 recurrent steps (LoopLM)
and one with 4x the layers and no looping (standard transformer, same FLOP budget).
Compare their performance on ARC-C and GSM8K after 20B tokens.
This reproduces the core claim of the paper.
```
