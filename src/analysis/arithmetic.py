"""src/analysis/arithmetic.py — Multi-step arithmetic for exit gate debugging.

Trains a small LoopLM on chains of additions/subtractions with varying
difficulty (1-4 operations). Tests whether the exit gate learns to allocate
more loops to harder problems.

Task format:
    <bos> <len_2> 8 + 3 - 5 = 6 <eos>

Difficulty axis:
    1 op  → should exit early  (avg_exit ~ 1)
    4 ops → should use more loops (avg_exit ~ 3-4)

Key metric: correlation between difficulty and avg_exit_step.

Usage:
    from src.analysis.arithmetic import ArithConfig, run_arith_experiment

    config = ArithConfig(max_ops=4, train_steps=5000, beta_kl=0.5)
    run_arith_experiment(config)
"""

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM, compute_exit_distribution


# ── Tokenizer ────────────────────────────────────────────────────────────────


class ArithTokenizer:
    """Minimal tokenizer for arithmetic chains.

    Vocabulary:
      - Digits: "0".."9"           (10 tokens)
      - Operators: "+", "-", "*"   (2-3 tokens)
      - Punctuation: " ", "="      (2 tokens)
      - Special: <bos>, <eos>, <pad>  (3 tokens)
      - Length markers: <len_1>..<len_L>  (max_ops tokens)
    """

    def __init__(self, max_ops: int, use_mul: bool = False):
        self.max_ops = max_ops
        self.use_mul = use_mul
        tokens: list[str] = []

        # Digits
        for i in range(10):
            tokens.append(str(i))

        # Operators and punctuation
        tokens.extend(["+", "-"])
        if use_mul:
            tokens.append("*")
        tokens.extend([" ", "="])

        # Special tokens
        tokens.extend(["<bos>", "<eos>", "<pad>"])

        # Length markers
        for i in range(1, max_ops + 1):
            tokens.append(f"<len_{i}>")

        self._token_to_id = {t: i for i, t in enumerate(tokens)}
        self._id_to_token = {i: t for i, t in enumerate(tokens)}
        self._tokens = tokens

        self.pad_token_id: int = self._token_to_id["<pad>"]
        self.eos_token_id: int = self._token_to_id["<eos>"]
        self.bos_token_id: int = self._token_to_id["<bos>"]
        self.eq_token_id: int = self._token_to_id["="]

    @property
    def vocab_size(self) -> int:
        return len(self._tokens)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self._token_to_id[t] for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self._id_to_token[i] for i in ids]


# ── Expression generation ────────────────────────────────────────────────────


def generate_arith_example(
    num_ops: int,
    rng: random.Random,
    use_mul: bool = False,
    two_digit: bool = False,
) -> tuple[list[str], int]:
    """Generate an arithmetic chain with `num_ops` operations.

    Format: <bos> <len_k> d op d op d ... = answer <eos>

    Args:
        num_ops: number of binary operations.
        rng: random number generator.
        use_mul: include multiplication as a possible operator.
        two_digit: use two-digit operands (10-99) instead of single-digit (1-9).

    Returns:
        (tokens, answer) where tokens is the full sequence.
    """
    lo, hi = (10, 99) if two_digit else (1, 9)
    op_choices = ["+", "-", "*"] if use_mul else ["+", "-"]

    # First operand
    operands = [rng.randint(lo, hi)]
    operators: list[str] = []

    result = operands[0]
    for _ in range(num_ops):
        op = rng.choice(op_choices)

        if op == "*":
            # Keep multipliers small to avoid huge results
            val = rng.randint(2, 9)
            # Bail to addition if result would overflow
            if abs(result * val) > 9999:
                op = "+"
                val = rng.randint(lo, hi)
            else:
                operators.append(op)
                operands.append(val)
                result *= val
                continue

        # For +/-, keep result in a reasonable range
        if op == "+":
            max_val = min(hi, 9999 - abs(result))
            if max_val < lo:
                op = "-"
        if op == "-":
            max_val = min(hi, abs(result))
            if max_val < lo:
                op = "+"
                max_val = min(hi, 9999 - abs(result))

        val = rng.randint(lo, max(lo, max_val))
        operators.append(op)
        operands.append(val)

        if op == "+":
            result += val
        else:
            result -= val

    # Build token sequence — each operand is digit-by-digit
    tokens = ["<bos>", f"<len_{num_ops}>"]
    for digit in str(operands[0]):
        tokens.append(digit)
    for i, op in enumerate(operators):
        tokens.extend([" ", op, " "])
        for digit in str(operands[i + 1]):
            tokens.append(digit)
    tokens.extend([" ", "=", " "])

    # Handle negative results
    ans_str = str(result)
    if result < 0:
        tokens.append("-")
        ans_str = ans_str[1:]  # skip the minus sign, we already added "-" token
    for digit in ans_str:
        tokens.append(digit)
    tokens.append("<eos>")

    return tokens, result


# ── Dataset ──────────────────────────────────────────────────────────────────


class ArithOnTheFlyDataset(Dataset):
    """Generates fresh arithmetic examples packed into fixed-length windows.

    Each __getitem__ packs multiple problems (separated by <eos>) into a
    (seq_len + 1)-token window with problem IDs for block-causal masking.
    """

    def __init__(
        self,
        tokenizer: ArithTokenizer,
        max_ops: int,
        seq_len: int = 128,
        seed: int = 42,
        length: int = 10_000,
        use_mul: bool = False,
        two_digit: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_ops = max_ops
        self.seq_len = seq_len
        self.chunk_len = seq_len + 1
        self.seed = seed
        self._length = length
        self._call_count = 0
        self.use_mul = use_mul
        self.two_digit = two_digit

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        self._call_count += 1
        rng = random.Random((self.seed, idx, self._call_count))

        ids: list[int] = []
        prob_ids: list[int] = []
        prob_idx = 0

        while len(ids) < self.chunk_len:
            num_ops = rng.randint(1, self.max_ops)
            tokens, _ = generate_arith_example(
                num_ops, rng, use_mul=self.use_mul, two_digit=self.two_digit
            )
            token_ids = self.tokenizer.encode(tokens)
            ids.extend(token_ids)
            prob_ids.extend([prob_idx] * len(token_ids))
            prob_idx += 1

        return (
            torch.tensor(ids[: self.chunk_len], dtype=torch.long),
            torch.tensor(prob_ids[: self.chunk_len], dtype=torch.long),
        )


# ── Block-causal mask (reused from MANO) ────────────────────────────────────


def build_block_causal_mask(problem_ids: Tensor) -> Tensor:
    """Build additive attention mask: 0.0 = attend, -inf = blocked."""
    B, S = problem_ids.shape
    device = problem_ids.device
    same_prob = problem_ids.unsqueeze(2) == problem_ids.unsqueeze(1)
    causal = torch.ones(S, S, dtype=torch.bool, device=device).tril()
    allow = same_prob & causal.unsqueeze(0)
    mask = torch.zeros(B, 1, S, S, device=device)
    mask.masked_fill_(~allow.unsqueeze(1), torch.finfo(mask.dtype).min)
    return mask


# ── Evaluation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_accuracy(
    model: LoopLM,
    tokenizer: ArithTokenizer,
    num_steps: int,
    max_ops: int,
    n_eval: int,
    device: torch.device,
    seed: int = 99999,
    use_mul: bool = False,
    two_digit: bool = False,
) -> dict[int, float]:
    """Evaluate exact-match accuracy per difficulty level.

    Returns:
        Dict mapping num_ops -> accuracy.
    """
    model.eval()
    rng = random.Random(seed)

    correct_by_diff: dict[int, int] = {d: 0 for d in range(1, max_ops + 1)}
    total_by_diff: dict[int, int] = {d: 0 for d in range(1, max_ops + 1)}

    for _ in range(n_eval):
        num_ops = rng.randint(1, max_ops)
        tokens, answer = generate_arith_example(num_ops, rng, use_mul=use_mul, two_digit=two_digit)
        ids = tokenizer.encode(tokens)

        # Find where "=" is — predict tokens after "="
        eq_pos = tokens.index("=")
        # Input: everything up to and including the space after "="
        # We predict the answer digits
        input_end = eq_pos + 2  # "=" + " " after it
        input_ids = tokenizer.encode(tokens[:input_end])

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_tensor, num_steps=num_steps)
            logits = out.logits[-1]  # final recurrent step

        # Predict digit by digit
        answer_str = str(answer)
        pred_digits = []
        for i in range(len(answer_str)):
            pos = -1 if i == 0 else -1  # always predict from last position
            pred_id = logits[0, -1].argmax().item()
            pred_token = tokenizer.decode([pred_id])[0]
            pred_digits.append(pred_token)

            if i < len(answer_str) - 1:
                # Extend input with predicted token for next digit
                input_ids.append(pred_id)
                input_tensor = torch.tensor(
                    [input_ids], dtype=torch.long, device=device
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(input_tensor, num_steps=num_steps)
                    logits = out.logits[-1]

        pred_str = "".join(pred_digits)
        total_by_diff[num_ops] += 1
        if pred_str == answer_str:
            correct_by_diff[num_ops] += 1

    return {
        d: correct_by_diff[d] / max(1, total_by_diff[d])
        for d in range(1, max_ops + 1)
    }


@torch.no_grad()
def evaluate_exit_times(
    model: LoopLM,
    tokenizer: ArithTokenizer,
    num_steps: int,
    max_ops: int,
    n_eval: int,
    device: torch.device,
    seed: int = 99999,
    use_mul: bool = False,
    two_digit: bool = False,
) -> dict[int, float]:
    """Measure average exit step per difficulty level.

    Uses the exit distribution p(t|x) to compute E[t] for the answer tokens.

    Returns:
        Dict mapping num_ops -> avg_exit_step (over answer token positions).
    """
    model.eval()
    rng = random.Random(seed)

    exit_sums: dict[int, float] = {d: 0.0 for d in range(1, max_ops + 1)}
    exit_counts: dict[int, int] = {d: 0 for d in range(1, max_ops + 1)}

    for _ in range(n_eval):
        num_ops = rng.randint(1, max_ops)
        tokens, answer = generate_arith_example(num_ops, rng, use_mul=use_mul, two_digit=two_digit)
        ids = tokenizer.encode(tokens)

        input_tensor = torch.tensor([ids[:-1]], dtype=torch.long, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_tensor, num_steps=num_steps)

        # Compute exit distribution
        exit_probs = compute_exit_distribution(out.exit_lambdas)  # (T, 1, S)
        T = exit_probs.shape[0]
        step_indices = torch.arange(1, T + 1, dtype=torch.float32, device=device)
        avg_exit = (exit_probs * step_indices.view(T, 1, 1)).sum(dim=0)  # (1, S)

        # Find answer token positions (after "=")
        eq_pos = tokens.index("=")
        # Answer starts at eq_pos + 2 (after "= ")
        ans_start = eq_pos + 2
        ans_end = len(tokens) - 1  # exclude <eos>

        for pos in range(ans_start, ans_end):
            if pos < avg_exit.shape[1]:
                exit_sums[num_ops] += avg_exit[0, pos].item()
                exit_counts[num_ops] += 1

    return {
        d: exit_sums[d] / max(1, exit_counts[d]) for d in range(1, max_ops + 1)
    }


# ── Experiment config ────────────────────────────────────────────────────────


@dataclass
class ArithConfig:
    """Configuration for the arithmetic exit gate experiment."""

    max_ops: int = 4  # Maximum number of operations (= max difficulty)
    num_recurrent_steps: int = 4  # T for LoopLM
    use_mul: bool = False  # Include multiplication
    two_digit: bool = False  # Use two-digit operands (10-99)

    # Model (tiny — ~3M params)
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 512

    # Training
    train_steps: int = 5_000
    batch_size: int = 128
    seq_len: int = 128
    lr: float = 1e-3
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta2: float = 0.98
    eps: float = 1e-6
    grad_clip: float = 1.0
    beta_kl: float = 0.1  # Entropy regularization weight
    detach_between_steps: bool = False  # 1-step gradient approx (HRM-style)
    deep_supervision: bool = False  # HRM-style: M segments, each with own optimizer step
    num_segments: int = 1  # M: number of deep supervision segments per training step
    use_postnorm: bool = False  # Post-norm instead of sandwich norm
    use_lecun_init: bool = False  # Truncated LeCun normal init
    use_adam_atan2: bool = False  # Scale-invariant Adam variant

    # Evaluation
    n_eval: int = 500
    eval_every: int = 1_000
    extrap_eval_steps: list[int] | None = None  # Test at these T values after training

    # Logging
    log_every: int = 100
    device: str = "auto"
    seed: int = 42
    output_dir: str = "runs/arithmetic"

    use_wandb: bool = False
    wandb_project: str = "looplm"
    wandb_run_name: str | None = None


# ── Training ─────────────────────────────────────────────────────────────────


def _deep_supervision_step(
    model: LoopLM,
    x: Tensor,
    tgt: Tensor,
    attn_mask: Tensor,
    config: "ArithConfig",
    optimizer,
    compute_loss_fn,
) -> tuple[float, dict]:
    """HRM-style deep supervision: M segments, each with own optimizer step.

    Each segment runs T recurrent steps. Between segments, the hidden state is
    detached. Each segment gets its own loss computation and optimizer step.
    """
    B, S = x.shape
    cos, sin = model.rope.get_cos_sin(S, x.device)

    h = model.embed(x)
    last_diags: dict = {}
    total_grad_norm = 0.0

    for seg in range(config.num_segments):
        if seg > 0:
            h = h.detach()

        logits_per_step: list[Tensor] = []
        exit_lambdas: list[Tensor] = []
        for _ in range(config.num_recurrent_steps):
            for layer in model.layers:
                h = layer(h, cos, sin, attn_mask)
            logits = model.lm_head(model.final_norm(h))
            lam = torch.sigmoid(model.exit_gate(h)).squeeze(-1)
            logits_per_step.append(logits)
            exit_lambdas.append(lam)

        loss, diags = compute_loss_fn(
            logits_per_step=logits_per_step,
            exit_lambdas=exit_lambdas,
            targets=tgt,
            beta=config.beta_kl,
        )

        optimizer.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        total_grad_norm += gn.item()
        last_diags = diags

    return total_grad_norm / config.num_segments, last_diags


def run_arith_experiment(config: ArithConfig) -> None:
    """Run the full arithmetic experiment: train + evaluate exit times."""
    device = _resolve_device(config.device)
    tokenizer = ArithTokenizer(config.max_ops, use_mul=config.use_mul)

    model_config = LoopLMConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        max_seq_len=config.seq_len,
        max_recurrent_steps=config.num_recurrent_steps,
    )

    torch.manual_seed(config.seed)
    model = LoopLM(model_config, use_postnorm=config.use_postnorm, use_lecun_init=config.use_lecun_init).to(device)
    P = sum(p.numel() for p in model.parameters())

    print(f"Arithmetic exit gate experiment")
    variant = []
    if config.use_mul:
        variant.append("mul")
    if config.two_digit:
        variant.append("2digit")
    variant_str = f"  variant={'|'.join(variant)}" if variant else ""
    flags = []
    if config.detach_between_steps:
        flags.append("detach")
    if config.deep_supervision:
        flags.append(f"deepsup(M={config.num_segments})")
    if config.use_postnorm:
        flags.append("postnorm")
    if config.use_lecun_init:
        flags.append("lecun")
    if config.use_adam_atan2:
        flags.append("adam_atan2")
    flags_str = f"  [{', '.join(flags)}]" if flags else ""
    print(f"  max_ops={config.max_ops}  T={config.num_recurrent_steps}  beta_kl={config.beta_kl}{variant_str}{flags_str}")
    print(f"  model: h={config.hidden_size}  L={config.num_layers}  params={P:,}")
    print(f"  training: {config.train_steps} steps  bs={config.batch_size}  lr={config.lr}")
    if config.deep_supervision:
        total_T = config.num_segments * config.num_recurrent_steps
        print(f"  deep supervision: M={config.num_segments} segments x T={config.num_recurrent_steps} steps = {total_T} total")
    print(f"  device: {device}")
    print()

    if config.use_wandb:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"arith_T{config.num_recurrent_steps}_beta{config.beta_kl}",
            config={
                "experiment": "arithmetic",
                "max_ops": config.max_ops,
                "num_recurrent_steps": config.num_recurrent_steps,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "n_params": P,
                "train_steps": config.train_steps,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "beta_kl": config.beta_kl,
                "use_mul": config.use_mul,
                "two_digit": config.two_digit,
                "detach_between_steps": config.detach_between_steps,
                "deep_supervision": config.deep_supervision,
                "num_segments": config.num_segments,
                "use_postnorm": config.use_postnorm,
                "use_lecun_init": config.use_lecun_init,
                "use_adam_atan2": config.use_adam_atan2,
                "seed": config.seed,
            },
        )

    dataset = ArithOnTheFlyDataset(
        tokenizer,
        max_ops=config.max_ops,
        seq_len=config.seq_len,
        seed=config.seed,
        use_mul=config.use_mul,
        two_digit=config.two_digit,
    )

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )

    if config.use_adam_atan2:
        from src.training.adam_atan2 import AdamAtan2
        optimizer = AdamAtan2(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, config.beta2),
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    warmup = min(config.warmup_steps, config.train_steps // 10)

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, config.train_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    from src.training.objectives import compute_looplm_loss

    # ── Training loop ────────────────────────────────────────────────────────
    model.train()
    data_iter = _infinite_iter(dataloader)
    t_start = time.monotonic()
    step = 0

    while step < config.train_steps:
        tokens, prob_ids = next(data_iter)
        tokens = tokens.to(device)
        prob_ids = prob_ids.to(device)

        x, tgt = tokens[:, :-1], tokens[:, 1:]
        prob_ids_x = prob_ids[:, :-1]
        attn_mask = build_block_causal_mask(prob_ids_x)

        if config.deep_supervision:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                grad_norm, diags = _deep_supervision_step(
                    model, x, tgt, attn_mask, config, optimizer, compute_looplm_loss,
                )
            scheduler.step()
            step += 1
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x, num_steps=config.num_recurrent_steps, attention_mask=attn_mask, detach_between_steps=config.detach_between_steps)
                loss, diags = compute_looplm_loss(
                    logits_per_step=out.logits,
                    exit_lambdas=out.exit_lambdas,
                    targets=tgt,
                    beta=config.beta_kl,
                )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
            optimizer.step()
            scheduler.step()
            step += 1

        # Logging
        if step % config.log_every == 0:
            elapsed = time.monotonic() - t_start
            eta = elapsed / step * (config.train_steps - step)

            per_step = diags.get("per_step_losses", [])
            step_str = "  ".join(f"t{i+1}:{v.item():.3f}" for i, v in enumerate(per_step))

            gn = grad_norm if isinstance(grad_norm, float) else grad_norm

            print(
                f"  step {step:5d}/{config.train_steps}  "
                f"loss={diags['loss'].item():.4f}  "
                f"task={diags['task_loss'].item():.4f}  "
                f"ent={diags['entropy'].item():.4f}  "
                f"exit={diags['avg_exit_step'].item():.2f}  "
                f"gnorm={gn:.3f}  "
                f"{step_str}  "
                f"[{_fmt_duration(elapsed)}/{_fmt_duration(elapsed + eta)}]"
            )

            if config.use_wandb:
                import wandb

                log_dict = {
                    "loss": diags["loss"].item(),
                    "task_loss": diags["task_loss"].item(),
                    "entropy": diags["entropy"].item(),
                    "avg_exit_step": diags["avg_exit_step"].item(),
                    "grad_norm": gn,
                    "lr": scheduler.get_last_lr()[0],
                }
                for i, v in enumerate(per_step):
                    log_dict[f"loss_step_{i+1}"] = v.item()
                wandb.log(log_dict, step=step)

        # Periodic evaluation
        if config.eval_every > 0 and step % config.eval_every == 0:
            _run_eval(model, tokenizer, config, device, step)
            model.train()

    # ── Final evaluation ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'FINAL EVALUATION':^70}")
    print("=" * 70)
    _run_eval(model, tokenizer, config, device, step)

    # ── Extrapolation evaluation ─────────────────────────────────────────────
    if config.extrap_eval_steps:
        print("\n" + "=" * 70)
        print(f"{'EXTRAPOLATION EVALUATION':^70}")
        print("=" * 70)
        print(f"  Trained with T={config.num_recurrent_steps}, evaluating at T={config.extrap_eval_steps}")
        _run_extrap_eval(model, tokenizer, config, device, step)

    # Save model
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "vocab_size": tokenizer.vocab_size,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "intermediate_size": config.intermediate_size,
                "max_seq_len": config.seq_len,
                "max_recurrent_steps": config.num_recurrent_steps,
            },
            "config": {
                "max_ops": config.max_ops,
                "beta_kl": config.beta_kl,
                "train_steps": config.train_steps,
            },
        },
        model_path,
    )
    print(f"\nModel saved to {model_path}")

    if config.use_wandb:
        import wandb

        wandb.finish()


def _run_eval(
    model: LoopLM,
    tokenizer: ArithTokenizer,
    config: ArithConfig,
    device: torch.device,
    step: int,
) -> None:
    """Run evaluation and print results."""
    acc_by_diff = evaluate_accuracy(
        model, tokenizer, config.num_recurrent_steps,
        config.max_ops, config.n_eval, device,
        use_mul=config.use_mul, two_digit=config.two_digit,
    )
    exit_by_diff = evaluate_exit_times(
        model, tokenizer, config.num_recurrent_steps,
        config.max_ops, config.n_eval, device,
        use_mul=config.use_mul, two_digit=config.two_digit,
    )

    print(f"\n  [eval @ step {step}]")
    print(f"  {'Difficulty':>10}  {'Accuracy':>10}  {'Avg Exit':>10}")
    print(f"  {'-'*34}")
    for d in range(1, config.max_ops + 1):
        print(f"  {d:>10}  {acc_by_diff[d]:>10.4f}  {exit_by_diff[d]:>10.2f}")

    overall_acc = sum(acc_by_diff.values()) / len(acc_by_diff)
    overall_exit = sum(exit_by_diff.values()) / len(exit_by_diff)
    print(f"  {'overall':>10}  {overall_acc:>10.4f}  {overall_exit:>10.2f}")

    # Exit correlation check
    diffs = list(range(1, config.max_ops + 1))
    exits = [exit_by_diff[d] for d in diffs]
    monotonic = all(exits[i] <= exits[i + 1] for i in range(len(exits) - 1))
    spread = exits[-1] - exits[0]
    print(f"\n  Exit monotonic: {'YES' if monotonic else 'NO'}  |  Spread: {spread:.2f}")
    print()

    if config.use_wandb:
        import wandb

        log_dict = {}
        for d in range(1, config.max_ops + 1):
            log_dict[f"eval/accuracy_ops{d}"] = acc_by_diff[d]
            log_dict[f"eval/avg_exit_ops{d}"] = exit_by_diff[d]
        log_dict["eval/accuracy_overall"] = overall_acc
        log_dict["eval/exit_spread"] = spread
        log_dict["eval/exit_monotonic"] = float(monotonic)
        wandb.log(log_dict, step=step)


def _run_extrap_eval(
    model: LoopLM,
    tokenizer: ArithTokenizer,
    config: ArithConfig,
    device: torch.device,
    step: int,
) -> None:
    """Evaluate accuracy at multiple T values (extrapolation test)."""
    assert config.extrap_eval_steps is not None

    # Header
    t_values = config.extrap_eval_steps
    header = f"  {'Diff':>6}"
    for t in t_values:
        marker = " *" if t == config.num_recurrent_steps else ""
        header += f"  {'T=' + str(t) + marker:>8}"
    print(header)
    print(f"  {'-' * (8 + 10 * len(t_values))}")

    # Evaluate each T
    all_accs: dict[int, dict[int, float]] = {}
    for t in t_values:
        all_accs[t] = evaluate_accuracy(
            model, tokenizer, t,
            config.max_ops, config.n_eval, device,
            use_mul=config.use_mul, two_digit=config.two_digit,
        )

    # Print per-difficulty rows
    for d in range(1, config.max_ops + 1):
        row = f"  {d:>6}"
        for t in t_values:
            row += f"  {all_accs[t][d]:>8.1%}"
        print(row)

    # Overall row
    row = f"  {'all':>6}"
    for t in t_values:
        overall = sum(all_accs[t].values()) / len(all_accs[t])
        row += f"  {overall:>8.1%}"
    print(row)
    print(f"\n  * = training T")
    print()

    if config.use_wandb:
        import wandb
        for t in t_values:
            overall = sum(all_accs[t].values()) / len(all_accs[t])
            wandb.log({f"extrap/accuracy_T{t}": overall}, step=step)
            for d in range(1, config.max_ops + 1):
                wandb.log({f"extrap/accuracy_T{t}_ops{d}": all_accs[t][d]}, step=step)


# ── Utilities ────────────────────────────────────────────────────────────────


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _infinite_iter(dataloader):
    while True:
        yield from dataloader
