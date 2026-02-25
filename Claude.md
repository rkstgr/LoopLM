# CLAUDE.md — LoopLM Implementation Guide

## Project Context
We are reimplementing the paper "Scaling Latent Reasoning via Looped Language Models" (Ouro, arXiv:2510.25741v4).
The paper PDF is at `paper/looplm.pdf`. Reference it when you need architectural details.

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

## Key Implementation Details (from paper)

### Architecture (Section 3.1, Figure 3)
- The looped forward pass: `F^(t)(x) = lmhead ∘ M_L^t ∘ emb(x)` where M_L is the full layer stack applied t times
- The exit gate: `λ_t = sigmoid(Linear(h^(t)))` where h^(t) is the final-layer hidden state at step t
- Survival probability: `S_t = ∏_{j=1}^{t} (1 - λ_j)`
- Exit distribution: `p(t|x) = λ_t · S_{t-1}` for t < T_max, `p(T_max|x) = S_{T_max-1}`
- The gate runs in parallel with the LM head at each step

### Training Objective — Stage I (Section 3.3, Equation 4)
```
L = Σ_{t=1}^{T_max} p(t|x) · L^(t) - β · H(p(·|x))
```
- L^(t) is standard cross-entropy at step t
- Use **uniform prior** (NOT geometric) — see Appendix A for empirical validation
- β starts at 0.1, reduces to 0.05 in later stages
- This is equivalent to an ELBO with uniform prior over exit steps

### Training Objective — Stage II (Section 3.4, Equation 6)
- Freeze LM parameters, train only exit gate
- Detached improvement: `I^(t) = max(0, L^(t-1)_detached - L^(t)_detached)`
- Ideal continuation: `w^(t) = sigmoid(50 * (I^(t) - 0.005))`
- BCE between predicted continuation `(1 - λ^(t))` and target `w^(t)`
- Average across steps t=2..T_max

### Inference (Section 3.2)
- Q-exit: accumulate CDF of exit probs, stop when CDF ≥ threshold q
- KV cache: separate per step during prefill, reuse last-step only during decode (4x savings)

### Model Configs (Table 2)
- Ouro 1.4B: 24 layers, hidden 2048, MHA, SwiGLU, RoPE, vocab 49152
- Ouro 2.6B: 48 layers, hidden 2048, MHA, SwiGLU, RoPE, vocab 49152
- Both use 4 recurrent steps at inference

### Training Schedule (Table 1)
- Stage 1a: LR 3e-4 constant, 8 recurrent steps, β=0.1, seq 4K, 3T tokens
- Stage 1b: LR 3e-4 constant, 4 recurrent steps, β=0.1, seq 4K, 3T tokens
- Stage 2: LR 3e-5 cosine, 4 steps, β=0.05, seq 16K, RoPE base 40K, 1.4T tokens
- Stage 3: LR 3e-5 constant, 4 steps, seq 64K, RoPE base 1M, 20B tokens
- Stage 4: LR 1e-5 cosine, 4 steps, seq 32K, RoPE base 1M, 300B tokens
- Optimizer: AdamW, β1=0.9, β2=0.95, weight_decay=0.1, grad_clip=1.0

## Common Pitfalls to Avoid
1. DON'T use pre-norm only — must be **sandwich norm** (RMSNorm before attention AND before FFN)
2. DON'T let the exit gate collapse to always using T_max — monitor entropy of p(t|x)
3. DON'T share KV cache during prefill — only during decode phase
4. DON'T forget to detach losses in Stage II gate training
5. DON'T use a geometric prior for the KL term — use uniform
6. DON'T start with 8 recurrent steps — the paper found this unstable, use 4 for prototyping
7. DON'T use the first-step KV cache for decode reuse — use last-step (first-step collapses performance)

## When Making Changes
- Always run existing tests after changes: `pytest tests/ -x`
- For any new module, add a corresponding test file
- Use type hints throughout
- Keep functions focused — prefer many small functions over monolithic ones
- Log per-step losses, entropy, and average exit step during training (key diagnostics)

## Dependencies
- PyTorch 2.x
- transformers (tokenizer)
- datasets (data loading)
- wandb (logging)
- lm-eval (evaluation)
- evalplus (code evaluation)
- flash-attn (optional, for speed)
