"""src/analysis/mano.py — Knowledge Manipulation experiment (Mano task).

Replicates Section 6.2 / Appendix B.2 of the LoopLM paper: train models on
modular arithmetic over tree structures in prefix notation (mod 23) and compare
base (k layers, loop=1) vs. looped (k/c layers, loop=c) models.

Expected result: looped models consistently outperform iso-parameter non-looped
models at knowledge manipulation, confirming that looping adds manipulation
capability beyond raw capacity.

Usage:
    from src.analysis.mano import ManoConfig, run_mano_experiment, print_mano_results

    # Paper reproduction (hidden=1024, all 7 configs, LR search)
    config = ManoConfig(
        max_ops=10,
        lr_candidates=PAPER_LR_CANDIDATES,
    )
    results = run_mano_experiment(config)
    print_mano_results(results)

    # Quick prototyping (small models)
    config = ManoConfig(
        max_ops=10,
        model_configs=[(4, 1), (2, 2), (1, 4)],
        model_preset="small",
        train_steps=5000,
    )
    results = run_mano_experiment(config)
    print_mano_results(results)

Implementation notes:
  - Expressions use prefix notation with operators {+, -, *} on F_23.
  - A custom tokenizer handles the tiny vocabulary (~55 tokens).
  - Training uses packing with block-causal masking (same concept as Capo).
  - Evaluation: exact-match accuracy on the answer token for hardest
    difficulty (ℓ = max_ops) only.
"""

import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM


# ── Modular arithmetic constants ─────────────────────────────────────────────

MODULUS: int = 23
OPS: list[str] = ["+", "-", "*"]


# ── Custom tokenizer ─────────────────────────────────────────────────────────


class ManoTokenizer:
    """Minimal tokenizer for Mano arithmetic expressions.

    Vocabulary:
      - Numbers: "0", "1", ..., "22"          (23 tokens)
      - Operators: "+", "-", "*"              (3 tokens)
      - Special: <bos>, <ans>, <eos>, <pad>   (4 tokens)
      - Length markers: <len_0> .. <len_L>    (max_ops + 1 tokens)

    Total vocab size = 23 + 3 + 4 + (max_ops + 1) = 31 + max_ops.
    """

    def __init__(self, max_ops: int):
        self.max_ops = max_ops
        tokens: list[str] = []

        # Numbers 0..22
        for i in range(MODULUS):
            tokens.append(str(i))

        # Operators
        tokens.extend(OPS)

        # Special tokens
        tokens.extend(["<bos>", "<ans>", "<eos>", "<pad>"])

        # Length markers
        for i in range(max_ops + 1):
            tokens.append(f"<len_{i}>")

        self._token_to_id = {t: i for i, t in enumerate(tokens)}
        self._id_to_token = {i: t for i, t in enumerate(tokens)}
        self._tokens = tokens

        self.pad_token_id: int = self._token_to_id["<pad>"]
        self.eos_token_id: int = self._token_to_id["<eos>"]
        self.bos_token_id: int = self._token_to_id["<bos>"]
        self.ans_token_id: int = self._token_to_id["<ans>"]

    @property
    def vocab_size(self) -> int:
        return len(self._tokens)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self._token_to_id[t] for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self._id_to_token[i] for i in ids]


# ── Expression tree generation ───────────────────────────────────────────────


def generate_expression(
    num_ops: int, rng: random.Random
) -> tuple[list[str], int]:
    """Generate a random prefix-notation expression with `num_ops` operators.

    Operators are drawn from {+, -, *}, operands from {0, ..., 22}.
    Computation is performed mod 23.

    Args:
        num_ops: number of binary operators (internal nodes) in the tree.
                 0 means a single leaf (number).
        rng: random number generator.

    Returns:
        (tokens, answer) where tokens is the list of prefix-notation tokens
        and answer is the integer result mod 23.
    """
    if num_ops == 0:
        val = rng.randint(0, MODULUS - 1)
        return [str(val)], val

    # Split remaining ops between left and right subtrees
    left_ops = rng.randint(0, num_ops - 1)
    right_ops = num_ops - 1 - left_ops

    op = rng.choice(OPS)
    left_tokens, left_val = generate_expression(left_ops, rng)
    right_tokens, right_val = generate_expression(right_ops, rng)

    if op == "+":
        result = (left_val + right_val) % MODULUS
    elif op == "-":
        result = (left_val - right_val) % MODULUS
    else:  # "*"
        result = (left_val * right_val) % MODULUS

    return [op] + left_tokens + right_tokens, result


def generate_mano_example(
    num_ops: int, rng: random.Random
) -> tuple[list[str], int]:
    """Generate a full Mano example: <bos> <len_ℓ> expr <ans> answer.

    Args:
        num_ops: number of operators in the expression tree.
        rng: random number generator.

    Returns:
        (tokens, answer) where tokens is the full sequence including
        special markers, and answer is the integer result.
    """
    expr_tokens, answer = generate_expression(num_ops, rng)
    tokens = ["<bos>", f"<len_{num_ops}>"] + expr_tokens + ["<ans>", str(answer)]
    return tokens, answer


# ── Dataset ──────────────────────────────────────────────────────────────────


class ManoDataset(Dataset):
    """Packs Mano arithmetic problems into fixed-length windows for LM training.

    Each problem is terminated with <eos>.  Problems are concatenated and
    chunked into (seq_len + 1)-token windows.  Each token is tagged with a
    problem ID for block-causal masking.

    __getitem__ returns (tokens, problem_ids), both shape (seq_len + 1,).
    """

    def __init__(
        self,
        tokenizer: ManoTokenizer,
        n_examples: int,
        max_ops: int,
        seq_len: int = 1024,
        seed: int = 42,
    ):
        rng = random.Random(seed)
        chunk_len = seq_len + 1

        all_ids: list[int] = []
        all_problem_ids: list[int] = []

        for prob_idx in range(n_examples):
            # Uniformly sample difficulty ℓ from [1, max_ops]
            num_ops = rng.randint(1, max_ops)
            tokens, _ = generate_mano_example(num_ops, rng)
            ids = tokenizer.encode(tokens)
            all_ids.extend(ids)
            all_problem_ids.extend([prob_idx] * len(ids))
            # EOS separator
            all_ids.append(tokenizer.eos_token_id)
            all_problem_ids.append(prob_idx)

        n = len(all_ids) // chunk_len
        if n == 0:
            # Ensure at least one chunk by padding
            pad_needed = chunk_len - len(all_ids)
            all_ids.extend([tokenizer.pad_token_id] * pad_needed)
            all_problem_ids.extend([-1] * pad_needed)
            n = 1

        flat = torch.tensor(all_ids[: n * chunk_len], dtype=torch.long)
        flat_prob = torch.tensor(all_problem_ids[: n * chunk_len], dtype=torch.long)
        self._chunks = flat.view(n, chunk_len)
        self._problem_ids = flat_prob.view(n, chunk_len)

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self._chunks[idx], self._problem_ids[idx]


class ManoOnTheFlyDataset(Dataset):
    """Generates fresh Mano examples on every access — no memorization possible.

    Each __getitem__ call packs freshly generated examples into a single
    (seq_len + 1)-token window with unique problem IDs for block-causal masking.

    The ``length`` parameter controls how many windows the DataLoader will draw
    per "epoch" (it is arbitrary since examples are always fresh).
    """

    def __init__(
        self,
        tokenizer: ManoTokenizer,
        max_ops: int,
        seq_len: int = 1024,
        seed: int = 42,
        length: int = 10_000,
    ):
        self.tokenizer = tokenizer
        self.max_ops = max_ops
        self.seq_len = seq_len
        self.chunk_len = seq_len + 1
        self.seed = seed
        self._length = length
        self._call_count = 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        # Unique RNG per call — combines seed, idx, and a monotonic counter
        # so even the same idx in different epochs yields different data
        self._call_count += 1
        rng = random.Random((self.seed, idx, self._call_count))

        ids: list[int] = []
        prob_ids: list[int] = []
        prob_idx = 0

        while len(ids) < self.chunk_len:
            num_ops = rng.randint(1, self.max_ops)
            tokens, _ = generate_mano_example(num_ops, rng)
            token_ids = self.tokenizer.encode(tokens)
            token_ids.append(self.tokenizer.eos_token_id)
            ids.extend(token_ids)
            prob_ids.extend([prob_idx] * len(token_ids))
            prob_idx += 1

        return (
            torch.tensor(ids[: self.chunk_len], dtype=torch.long),
            torch.tensor(prob_ids[: self.chunk_len], dtype=torch.long),
        )


# ── Block-causal attention mask (same concept as Capo) ───────────────────────


def build_block_causal_mask(problem_ids: Tensor) -> Tensor:
    """Build an additive attention mask enforcing block-causal structure.

    Allows position i to attend to position j iff:
      1. same problem: problem_ids[:, j] == problem_ids[:, i]
      2. causal: j <= i

    Args:
        problem_ids: (B, S) integer problem-ID per token

    Returns:
        (B, 1, S, S) float mask; 0.0 = attend, -inf = blocked
    """
    B, S = problem_ids.shape
    device = problem_ids.device

    same_prob = problem_ids.unsqueeze(2) == problem_ids.unsqueeze(1)  # (B, S, S)
    causal = torch.ones(S, S, dtype=torch.bool, device=device).tril()
    allow = same_prob & causal.unsqueeze(0)

    mask = torch.zeros(B, 1, S, S, device=device)
    mask.masked_fill_(~allow.unsqueeze(1), torch.finfo(mask.dtype).min)
    return mask


# ── Evaluation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_mano(
    model: LoopLM,
    tokenizer: ManoTokenizer,
    num_steps: int,
    max_ops: int,
    n_eval: int,
    device: torch.device,
    seed: int = 99999,
) -> float:
    """Evaluate exact-match accuracy on the hardest difficulty (ℓ = max_ops).

    For each example, the model is given the full prefix expression up to
    and including <ans>, and must predict the correct answer token.

    Args:
        model: trained LoopLM.
        tokenizer: ManoTokenizer.
        num_steps: recurrent steps for inference.
        max_ops: evaluate only on expressions with this many operators.
        n_eval: number of evaluation examples.
        device: compute device.
        seed: RNG seed for reproducible evaluation set.

    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()

    rng = random.Random(seed)
    correct = 0

    for _ in range(n_eval):
        tokens, answer = generate_mano_example(max_ops, rng)
        ids = tokenizer.encode(tokens)

        # Input: everything up to and including <ans> (exclude answer token)
        # The model should predict the answer at the last position.
        input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids, num_steps=num_steps)
            # Use final recurrent step logits
            logits = out.logits[-1]  # (1, S, vocab)
            pred = logits[0, -1].argmax().item()

        expected = tokenizer.encode([str(answer)])[0]
        if pred == expected:
            correct += 1

    return correct / max(1, n_eval)


# ── Model configs ────────────────────────────────────────────────────────────


def make_mano_model_config(
    num_layers: int,
    loop_count: int,
    hidden_size: int,
    num_heads: int,
    vocab_size: int,
    seq_len: int = 1024,
) -> LoopLMConfig:
    """Create a LoopLMConfig for the Mano task."""
    return LoopLMConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=hidden_size * 2,  # SwiGLU intermediate
        max_seq_len=seq_len,
        max_recurrent_steps=loop_count,
    )


# Paper model configs: hidden=1024, depth L ∈ {10, 16, 24}
# For prototyping we also provide smaller configs.
_MODEL_PRESETS: dict[str, dict] = {
    "tiny": dict(hidden_size=64, num_heads=1),
    "small": dict(hidden_size=128, num_heads=2),
    "medium": dict(hidden_size=256, num_heads=4),
    "paper": dict(hidden_size=1024, num_heads=16),
}

# Paper Section 6.2 / Appendix B.2: exact model configurations.
# Baselines: {2,3,6,12}×1 (standard transformers)
# Looped: k×(12/k) for k ∈ {2,3,6} → 2×6, 3×4, 6×2
PAPER_MODEL_CONFIGS: list[tuple[int, int]] = [
    (2, 1), (3, 1), (6, 1), (12, 1),  # baselines
    (2, 6), (3, 4), (6, 2),            # looped (iso-param comparisons)
]

# Paper Appendix B.2: difficulty levels → training steps
PAPER_DIFFICULTIES: dict[int, int] = {
    10: 80_000,
    16: 110_000,
    24: 200_000,
}

# Paper Appendix B.2: LR search grid
PAPER_LR_CANDIDATES: list[float] = [5e-5, 1e-4, 2e-4, 5e-4]


# ── Experiment config & result ───────────────────────────────────────────────


@dataclass
class ManoConfig:
    """Configuration for the Mano knowledge manipulation experiment."""

    max_ops: int = 10  # Maximum expression tree depth (paper: 10, 16, 24)
    n_train_examples: int = 500_000  # Training examples to generate
    n_eval_examples: int = 1_000  # Evaluation examples (hardest difficulty)

    # Model configurations: list of (num_layers, loop_count) pairs.
    # Paper: baselines {2,3,6,12}×1, looped {2×6, 3×4, 6×2}
    model_configs: list[tuple[int, int]] = field(
        default_factory=lambda: list(PAPER_MODEL_CONFIGS)
    )
    model_preset: str = "paper"  # Key into _MODEL_PRESETS

    # Training hyperparams (paper Appendix B.2)
    lr: float = 1e-4
    lr_candidates: list[float] | None = None  # If set, search over these LRs
    weight_decay: float = 0.1
    beta2: float = 0.98
    eps: float = 1e-6
    batch_size: int = 128
    accumulation_steps: int = 1
    seq_len: int = 1024
    warmup_steps: int = 1_000
    grad_clip: float = 1.0
    train_steps: int = 80_000  # Paper: {80K, 110K, 200K} for different difficulties
    beta_kl: float = 0.1  # Entropy weight for LoopLM loss

    log_every: int = 500
    checkpoint_every: int = 5_000  # Save checkpoint every N optimizer steps (0=disable)
    device: str = "auto"
    seed: int = 42
    output_dir: str = "runs/mano"

    online_data: bool = False  # Generate fresh examples on the fly (no memorization)

    use_wandb: bool = False
    wandb_project: str = "looplm"
    wandb_run_name: str | None = None


@dataclass
class ManoResult:
    num_layers: int
    loop_count: int
    total_depth: int  # num_layers * loop_count (iso-FLOP comparison)
    n_params: int
    max_ops: int
    accuracy: float
    final_loss: float
    lr: float = 1e-4  # LR that produced this result


# ── Checkpointing ─────────────────────────────────────────────────────────────


def _save_mano_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    last_loss: float,
    nan_count: int,
) -> None:
    """Save a resumable training checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "last_loss": last_loss,
            "nan_count": nan_count,
        },
        tmp,
    )
    tmp.rename(path)  # atomic on POSIX
    print(f"    Checkpoint saved at step {step} → {path}")


# ── Single-run training ──────────────────────────────────────────────────────


def run_mano_single(
    tokenizer: ManoTokenizer,
    num_layers: int,
    loop_count: int,
    config: ManoConfig,
    device: torch.device,
) -> ManoResult:
    """Train one (num_layers, loop_count) model and evaluate accuracy.

    Training follows paper Appendix B.2:
      - AdamW with β₁=0.9, β₂=0.98, ε=1e-6
      - 1000-step warmup → cosine decay to 0.1× peak LR
      - Block-causal attention mask prevents cross-problem leakage
      - bf16 mixed precision
    """
    preset = _MODEL_PRESETS[config.model_preset]
    model_config = make_mano_model_config(
        num_layers=num_layers,
        loop_count=loop_count,
        hidden_size=preset["hidden_size"],
        num_heads=preset["num_heads"],
        vocab_size=tokenizer.vocab_size,
        seq_len=config.seq_len,
    )

    torch.manual_seed(config.seed)
    model = LoopLM(model_config).to(device)
    P = sum(p.numel() for p in model.parameters())

    # wandb init (per run, so each config×seed gets its own run)
    if config.use_wandb:
        import wandb

        run_name = config.wandb_run_name or (
            f"mano_L{num_layers}_T{loop_count}_s{config.seed}"
        )
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                "experiment": "mano",
                "num_layers": num_layers,
                "loop_count": loop_count,
                "total_depth": num_layers * loop_count,
                "n_params": P,
                "max_ops": config.max_ops,
                "hidden_size": preset["hidden_size"],
                "model_preset": config.model_preset,
                "lr": config.lr,
                "batch_size": config.batch_size,
                "train_steps": config.train_steps,
                "beta_kl": config.beta_kl,
                "seed": config.seed,
            },
            reinit=True,
        )

    if config.online_data:
        dataset = ManoOnTheFlyDataset(
            tokenizer,
            max_ops=config.max_ops,
            seq_len=config.seq_len,
            seed=config.seed,
        )
    else:
        dataset = ManoDataset(
            tokenizer,
            n_examples=config.n_train_examples,
            max_ops=config.max_ops,
            seq_len=config.seq_len,
            seed=config.seed,
        )

    micro_batch_size = config.batch_size
    accumulation_steps = config.accumulation_steps
    effective_batch_size = micro_batch_size * accumulation_steps

    print(
        f"    params={P:,}  micro_bs={micro_batch_size}  accum={accumulation_steps}"
        f"  eff_bs={effective_batch_size}  lr={config.lr:.1e}"
    )
    print(f"    dataset chunks={len(dataset):,}  train_steps={config.train_steps:,}")

    dataloader = DataLoader(
        dataset, batch_size=micro_batch_size, shuffle=True, drop_last=True
    )

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

    # ── Checkpoint resume ────────────────────────────────────────────────────
    ckpt_dir = Path(config.output_dir) / f"ckpt_L{num_layers}_T{loop_count}_lr{config.lr:.0e}_s{config.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint.pt"

    start_step = 0  # optimizer steps completed
    last_loss = float("nan")
    nan_count = 0
    consecutive_nans = 0

    if ckpt_path.exists():
        print(f"    Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        last_loss = ckpt.get("last_loss", float("nan"))
        nan_count = ckpt.get("nan_count", 0)
        print(f"    Resumed at optimizer step {start_step}/{config.train_steps}")
        del ckpt

    if start_step >= config.train_steps:
        print(f"    Already completed {config.train_steps} steps, skipping to eval")
    else:
        # ── Training loop ────────────────────────────────────────────────────
        model.train()
        data_iter = _infinite_iter(dataloader)
        t_start = time.monotonic()

        start_micro = start_step * accumulation_steps
        total_micro_steps = config.train_steps * accumulation_steps

        optimizer.zero_grad()
        for step in range(start_micro, total_micro_steps):
            tokens, prob_ids = next(data_iter)
            tokens = tokens.to(device)
            prob_ids = prob_ids.to(device)

            x, tgt = tokens[:, :-1], tokens[:, 1:]
            prob_ids_x = prob_ids[:, :-1]

            attn_mask = build_block_causal_mask(prob_ids_x)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x, num_steps=loop_count, attention_mask=attn_mask)
                loss, diags = compute_looplm_loss(
                    logits_per_step=out.logits,
                    exit_lambdas=out.exit_lambdas,
                    targets=tgt,
                    beta=config.beta_kl,
                )
                loss = loss / accumulation_steps

            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                update = (step + 1) // accumulation_steps
                if not torch.isfinite(grad_norm):
                    nan_count += 1
                    consecutive_nans += 1
                    optimizer.zero_grad()
                    if consecutive_nans <= 5 or nan_count % 50 == 0:
                        print(f"    ⚠ step {update}: non-finite gradients (count={nan_count}), skipping update")
                    if consecutive_nans > 10:
                        print(f"    ✗ step {update}: {consecutive_nans} consecutive NaN updates, stopping early")
                        break
                else:
                    consecutive_nans = 0
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    last_loss = loss.item() * accumulation_steps

                # Logging
                if update % config.log_every == 0 or update == config.train_steps:
                    elapsed = time.monotonic() - t_start
                    micro_done = step + 1 - start_micro
                    micro_left = total_micro_steps - step - 1
                    eta = elapsed / micro_done * micro_left if micro_done > 0 else 0

                    # Answer-token diagnostics
                    with torch.no_grad():
                        ans_mask = (x == tokenizer.ans_token_id).float()
                        n_ans = ans_mask.sum().clamp(min=1.0)
                        final_logits = out.logits[-1]  # (B, S, V)
                        B_log, S_log = x.shape
                        per_token_ce = F.cross_entropy(
                            final_logits.reshape(B_log * S_log, -1).float(),
                            tgt.reshape(B_log * S_log),
                            reduction="none",
                        ).view(B_log, S_log)
                        ans_loss = (per_token_ce * ans_mask).sum() / n_ans
                        non_ans_mask = 1.0 - ans_mask
                        non_ans_loss = (per_token_ce * non_ans_mask).sum() / non_ans_mask.sum().clamp(min=1.0)
                        preds = final_logits.argmax(dim=-1)
                        ans_acc = ((preds == tgt).float() * ans_mask).sum() / n_ans

                    print(
                        f"    step {update:6d}/{config.train_steps}  loss={last_loss:.4f}"
                        f"  ans_loss={ans_loss:.4f}  ans_acc={ans_acc:.4f}"
                        f"  elapsed={_fmt_duration(elapsed)}  eta={_fmt_duration(eta)}"
                    )

                    if config.use_wandb:
                        import wandb

                        log_dict: dict[str, float] = {
                            "loss": last_loss,
                            "task_loss": diags["task_loss"].item(),
                            "entropy": diags["entropy"].item(),
                            "avg_exit_step": diags["avg_exit_step"].item(),
                            "lr": scheduler.get_last_lr()[0],
                            "nan_count": nan_count,
                            "ans_loss": ans_loss.item(),
                            "non_ans_loss": non_ans_loss.item(),
                            "ans_accuracy": ans_acc.item(),
                        }
                        for i, v in enumerate(diags["per_step_losses"]):
                            log_dict[f"loss_step_{i+1}"] = v.item()
                        wandb.log(log_dict, step=update)

                # Checkpoint
                if config.checkpoint_every > 0 and update % config.checkpoint_every == 0:
                    _save_mano_checkpoint(
                        ckpt_path, model, optimizer, scheduler,
                        update, last_loss, nan_count,
                    )
                    # Evaluate at checkpoint
                    model.eval()
                    ckpt_acc = evaluate_mano(
                        model, tokenizer, num_steps=loop_count,
                        max_ops=config.max_ops,
                        n_eval=min(200, config.n_eval_examples),
                        device=device, seed=config.seed + 12345,
                    )
                    model.train()
                    print(f"    [eval@{update}] accuracy={ckpt_acc:.4f}")
                    if config.use_wandb:
                        import wandb
                        wandb.log({"eval/accuracy": ckpt_acc}, step=update)

        # Final checkpoint
        final_update = config.train_steps
        if config.checkpoint_every > 0:
            _save_mano_checkpoint(
                ckpt_path, model, optimizer, scheduler,
                final_update, last_loss, nan_count,
            )

    # Evaluate
    accuracy = evaluate_mano(
        model, tokenizer, num_steps=loop_count,
        max_ops=config.max_ops, n_eval=config.n_eval_examples,
        device=device, seed=config.seed + 12345,
    )

    if config.use_wandb:
        import wandb

        wandb.log({"eval/accuracy": accuracy, "eval/final_loss": last_loss})
        wandb.finish()

    total_depth = num_layers * loop_count
    return ManoResult(
        num_layers=num_layers,
        loop_count=loop_count,
        total_depth=total_depth,
        n_params=P,
        max_ops=config.max_ops,
        accuracy=accuracy,
        final_loss=last_loss,
        lr=config.lr,
    )


# ── Full experiment ──────────────────────────────────────────────────────────


def run_mano_experiment(config: ManoConfig) -> list[ManoResult]:
    """Run the full Mano experiment across all (num_layers, loop_count) pairs.

    When config.lr_candidates is set, tries each LR for every model config
    and keeps the result with the highest accuracy (paper: "report the best
    performance" across seeds and LRs).

    Expected result: looped models outperform iso-parameter non-looped models,
    confirming that looping adds knowledge manipulation capability.
    """
    device = _resolve_device(config.device)
    tokenizer = ManoTokenizer(config.max_ops)

    lr_list = config.lr_candidates or [config.lr]

    results: list[ManoResult] = []
    for num_layers, loop_count in config.model_configs:
        total_depth = num_layers * loop_count
        print(
            f"\n[mano] layers={num_layers}  loop={loop_count}  "
            f"total_depth={total_depth}  max_ops={config.max_ops}"
        )

        best_result: ManoResult | None = None
        for lr in lr_list:
            lr_config = ManoConfig(
                max_ops=config.max_ops,
                n_train_examples=config.n_train_examples,
                n_eval_examples=config.n_eval_examples,
                model_configs=config.model_configs,
                model_preset=config.model_preset,
                lr=lr,
                lr_candidates=None,  # prevent recursion
                weight_decay=config.weight_decay,
                beta2=config.beta2,
                eps=config.eps,
                batch_size=config.batch_size,
                accumulation_steps=config.accumulation_steps,
                seq_len=config.seq_len,
                warmup_steps=config.warmup_steps,
                grad_clip=config.grad_clip,
                train_steps=config.train_steps,
                beta_kl=config.beta_kl,
                log_every=config.log_every,
                checkpoint_every=config.checkpoint_every,
                device=config.device,
                seed=config.seed,
                output_dir=config.output_dir,
                use_wandb=config.use_wandb,
                wandb_project=config.wandb_project,
                wandb_run_name=config.wandb_run_name,
            )
            if len(lr_list) > 1:
                print(f"  [lr={lr:.1e}]")
            result = run_mano_single(
                tokenizer, num_layers, loop_count, lr_config, device
            )
            if best_result is None or result.accuracy > best_result.accuracy:
                best_result = result

        assert best_result is not None
        results.append(best_result)
        lr_tag = f"  best_lr={best_result.lr:.1e}" if len(lr_list) > 1 else ""
        print(
            f"  → accuracy={best_result.accuracy:.4f}  loss={best_result.final_loss:.4f}  "
            f"params={best_result.n_params:,}{lr_tag}"
        )

    return results


# ── Output helpers ───────────────────────────────────────────────────────────


def print_mano_results(results: list[ManoResult]) -> None:
    """Print a formatted table of Mano experiment results."""
    print("\n" + "=" * 85)
    print(f"{'MANO RESULTS — Knowledge Manipulation':^85}")
    print("=" * 85)
    print(
        f"  {'Layers':>6} {'Loop':>5} {'Depth':>6} {'Params':>10} "
        f"{'max_ops':>8} {'Accuracy':>10} {'Loss':>8} {'LR':>10}"
    )
    print("  " + "-" * 81)
    for r in results:
        print(
            f"  {r.num_layers:>6} {r.loop_count:>5} {r.total_depth:>6} "
            f"{r.n_params / 1e6:>8.2f}M {r.max_ops:>8} "
            f"{r.accuracy:>10.4f} {r.final_loss:>8.4f} {r.lr:>10.1e}"
        )
    print("=" * 85)
    print("Expected: looped models outperform non-looped at same param count")
    print("  iso-param pairs: (2×1 vs 2×6), (3×1 vs 3×4), (6×1 vs 6×2)")
    print("  iso-FLOP baseline: 12×1 vs all looped models (all 12 effective layers)")
    print()


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
