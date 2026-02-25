"""scripts/evaluate.py — Evaluate a trained LoopLM checkpoint via lm-eval-harness.

Usage:
    uv run scripts/evaluate.py \\
        --checkpoint checkpoints/step_0010000.pt \\
        --tasks arc_easy,hellaswag \\
        --limit 100 \\
        --device cuda

    # Sprint 3.3 — evaluate at T=1..max_recurrent_steps
    uv run scripts/evaluate.py \\
        --checkpoint checkpoints/step_0010000.pt \\
        --tasks arc_easy \\
        --limit 100 \\
        --eval-all-steps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM

# lm-eval imports — available after `pip install lm-eval`
import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# lm-eval wrapper
# ---------------------------------------------------------------------------

class LoopLMLM(LM):
    """Wraps a trained LoopLM so lm-eval-harness can evaluate it.

    All forward passes use `num_steps` recurrent iterations and take logits
    from the final step only.
    """

    def __init__(
        self,
        model: LoopLM,
        tokenizer: AutoTokenizer,
        device: torch.device,
        num_steps: int,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self.num_steps = num_steps
        self._batch_size = batch_size

    # ── Required properties ─────────────────────────────────────────────────

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self.model.config.max_seq_len

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    # ── Tokenization helpers ────────────────────────────────────────────────

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    # ── Forward pass ────────────────────────────────────────────────────────

    def _get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the model and return final-step log-probs.

        Args:
            input_ids: (1, S) token ids

        Returns:
            log_probs: (1, S, vocab_size)
        """
        with torch.no_grad():
            out = self.model(input_ids, num_steps=self.num_steps)
        logits = out.logits[-1]  # last recurrent step, (1, S, vocab_size)
        return F.log_softmax(logits, dim=-1)

    # ── lm-eval API ─────────────────────────────────────────────────────────

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute log-likelihood of a continuation given a context.

        Each request.args is (context_str, continuation_str).
        """
        results: list[tuple[float, bool]] = []

        for request in requests:
            ctx_str, cont_str = request.args

            ctx_tokens = self._encode(ctx_str)
            cont_tokens = self._encode(cont_str)

            if len(cont_tokens) == 0:
                results.append((0.0, True))
                continue

            all_tokens = ctx_tokens + cont_tokens

            # Truncate from the left to fit max_seq_len, keeping all cont_tokens
            max_len = self.model.config.max_seq_len
            if len(all_tokens) > max_len:
                # How many tokens we can keep from ctx
                keep_ctx = max_len - len(cont_tokens)
                ctx_tokens = ctx_tokens[-keep_ctx:] if keep_ctx > 0 else []
                all_tokens = ctx_tokens + cont_tokens

            input_ids = torch.tensor([all_tokens], dtype=torch.long, device=self._device)
            log_probs = self._get_logits(input_ids)  # (1, S, vocab)

            # Positions of continuation tokens in the sequence.
            # token[i] predicts token[i+1], so we read log_probs at positions
            # [ctx_len-1 .. ctx_len+cont_len-2] to score cont_tokens[0..cont_len-1].
            ctx_len = len(ctx_tokens)
            cont_start = ctx_len - 1  # log_probs at this position predicts cont[0]

            # Gather log-probs for each continuation token
            cont_token_ids = torch.tensor(cont_tokens, dtype=torch.long, device=self._device)
            token_log_probs = (
                log_probs[0, cont_start : cont_start + len(cont_tokens), :]
                .gather(1, cont_token_ids.unsqueeze(1))
                .squeeze(1)
            )

            total_log_prob = token_log_probs.sum().item()

            # Greedy check: did argmax match at every continuation position?
            argmax_ids = log_probs[0, cont_start : cont_start + len(cont_tokens), :].argmax(dim=-1)
            is_greedy = (argmax_ids == cont_token_ids).all().item()

            results.append((total_log_prob, bool(is_greedy)))

        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute rolling log-likelihood for long sequences via a sliding window."""
        results: list[tuple[float, bool]] = []
        max_len = self.model.config.max_seq_len

        for request in requests:
            (text,) = request.args
            tokens = self._encode(text)

            if len(tokens) == 0:
                results.append((0.0, True))
                continue

            total_log_prob = 0.0
            # Process in windows of max_len; stride by max_len-1 so each token
            # is scored exactly once with a non-empty context.
            stride = max_len - 1
            prev_end = 0

            for window_start in range(0, len(tokens), stride):
                window_end = min(window_start + max_len, len(tokens))
                chunk = tokens[window_start:window_end]

                input_ids = torch.tensor([chunk], dtype=torch.long, device=self._device)
                log_probs = self._get_logits(input_ids)  # (1, W, vocab)

                # Score tokens from (prev_end - window_start) onward, shifted by 1
                score_from = max(prev_end - window_start, 1)  # at least 1 (need context)
                for pos in range(score_from, len(chunk)):
                    tok = chunk[pos]
                    total_log_prob += log_probs[0, pos - 1, tok].item()

                prev_end = window_end
                if window_end == len(tokens):
                    break

            results.append((total_log_prob, False))

        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError(
            "generate_until is not implemented for LoopLMLM. "
            "Use logprob-mode tasks (arc_easy, hellaswag, etc.) instead."
        )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    num_steps_override: int | None = None,
) -> tuple[LoopLM, LoopLMConfig, int]:
    """Load a LoopLM checkpoint.

    Returns:
        (model, config, num_steps_to_use)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = LoopLMConfig(**ckpt["model_config"])
    model = LoopLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    num_steps = num_steps_override if num_steps_override is not None else config.max_recurrent_steps
    return model, config, num_steps


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _extract_acc(results: dict, task: str) -> float | None:
    """Pull the accuracy value from lm-eval results dict for a given task."""
    task_res = results.get("results", {}).get(task, {})
    # lm-eval uses "acc,none" or "acc_norm,none" depending on the task
    for key in ("acc_norm,none", "acc,none"):
        if key in task_res:
            return task_res[key]
    return None


def run_eval(
    wrapper: LoopLMLM,
    tasks: list[str],
    limit: int | None,
) -> dict[str, Any]:
    """Run lm-eval simple_evaluate and return the raw results dict."""
    return lm_eval.simple_evaluate(
        model=wrapper,
        tasks=tasks,
        limit=limit,
        log_samples=False,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a LoopLM checkpoint with lm-eval-harness.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    p.add_argument(
        "--tasks",
        default="arc_easy,hellaswag",
        help="Comma-separated lm-eval task names (default: arc_easy,hellaswag).",
    )
    p.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of recurrent steps (default: value from checkpoint config).",
    )
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1).")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max examples per task for quick sanity checks (default: full dataset).",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device: auto|cuda|cpu (default: auto).",
    )
    p.add_argument(
        "--output-path",
        default=None,
        help="Optional path to write JSON results.",
    )
    p.add_argument(
        "--eval-all-steps",
        action="store_true",
        help="Evaluate at T=1..max_recurrent_steps and print a depth comparison table.",
    )
    return p.parse_args()


def _print_depth_table(
    table: list[tuple[int, dict[str, float | None]]],
    tasks: list[str],
) -> None:
    """Print a Markdown table of accuracy vs. recurrent depth."""
    header = "| Steps | " + " | ".join(f"{t:^10}" for t in tasks) + " |"
    sep    = "|-------|" + "|".join("-" * 12 for _ in tasks) + "|"
    print()
    print(header)
    print(sep)
    for num_steps, accs in table:
        row_vals = " | ".join(
            f"{accs[t]:^10.4f}" if accs.get(t) is not None else f"{'N/A':^10}"
            for t in tasks
        )
        print(f"| {num_steps:^5d} | {row_vals} |")
    print()


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"Loading checkpoint: {args.checkpoint}")
    model, config, default_steps = load_checkpoint(args.checkpoint, device, args.num_steps)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    if args.eval_all_steps:
        # Sprint 3.3 — evaluate at each recurrent depth
        table: list[tuple[int, dict[str, float | None]]] = []
        for T in range(1, config.max_recurrent_steps + 1):
            print(f"\nEvaluating at T={T} recurrent steps …")
            wrapper = LoopLMLM(model, tokenizer, device, num_steps=T, batch_size=args.batch_size)
            raw = run_eval(wrapper, tasks, args.limit)
            accs = {t: _extract_acc(raw, t) for t in tasks}
            for t, acc in accs.items():
                print(f"  {t}: {acc:.4f}" if acc is not None else f"  {t}: N/A")
            table.append((T, accs))

        print("\n=== Recurrent Depth vs. Accuracy ===")
        _print_depth_table(table, tasks)

        if args.output_path:
            output = {
                "checkpoint": args.checkpoint,
                "tasks": tasks,
                "limit": args.limit,
                "depth_table": [
                    {"steps": T, "accs": accs} for T, accs in table
                ],
            }
            Path(args.output_path).write_text(json.dumps(output, indent=2))
            print(f"Results written to {args.output_path}")

    else:
        # Standard single-depth evaluation
        num_steps = default_steps
        print(f"Running evaluation: tasks={tasks}, steps={num_steps}, limit={args.limit}")
        wrapper = LoopLMLM(model, tokenizer, device, num_steps=num_steps, batch_size=args.batch_size)
        raw = run_eval(wrapper, tasks, args.limit)

        print("\n=== Results ===")
        for task in tasks:
            acc = _extract_acc(raw, task)
            print(f"  {task}: {acc:.4f}" if acc is not None else f"  {task}: N/A")

        if args.output_path:
            Path(args.output_path).write_text(json.dumps(raw, indent=2))
            print(f"Results written to {args.output_path}")


if __name__ == "__main__":
    main()
