"""src/inference/lm_eval_wrapper.py — Reusable lm-eval primitives for LoopLM.

Shared between scripts/evaluate.py and src/training/trainer.py so periodic
per-depth evaluation can happen during training without duplicating code.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.model.looplm import LoopLM

# lm-eval imports — available after `pip install lm-eval`
import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance


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
