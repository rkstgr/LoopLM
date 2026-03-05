from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from src.model.config import LoopLMConfig
from src.model.rope import RotaryEmbedding
from src.model.transformer import RMSNorm, TransformerBlock


@dataclass
class LoopLMOutput:
    # One tensor per recurrent step
    logits: list[Tensor]       # each (B, S, vocab_size)
    exit_lambdas: list[Tensor] # each (B, S) — raw sigmoid gate outputs λ_t


def compute_exit_distribution(exit_lambdas: list[Tensor]) -> Tensor:
    """Compute p(t|x) from raw gate outputs λ_t.

    For t < T_max:  p(t|x) = λ_t · S_{t-1}   where S_0 = 1
    For t = T_max:  p(T_max|x) = S_{T_max-1}

    Args:
        exit_lambdas: list of T tensors, each (B, S)

    Returns:
        exit_probs: (T, B, S) — sums to 1 over the T dimension
    """
    T = len(exit_lambdas)
    probs = []
    survival = torch.ones_like(exit_lambdas[0])  # S_0 = 1

    for t, lam in enumerate(exit_lambdas):
        if t < T - 1:
            p_t = lam * survival
            survival = survival * (1.0 - lam)
        else:
            # Last step absorbs remaining probability mass
            p_t = survival
        probs.append(p_t)

    return torch.stack(probs, dim=0)  # (T, B, S)


class LoopLM(nn.Module):
    """Looped Language Model with shared-weight recurrence.

    A stack of `num_layers` TransformerBlocks is applied `num_steps` times,
    reusing the same weights each pass. An LM head and an exit gate are applied
    after every recurrent step.
    """

    def __init__(self, config: LoopLMConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.exit_gate = nn.Linear(config.hidden_size, 1, bias=True)

        # Tie LM head weights to the embedding matrix (standard practice)
        self.lm_head.weight = self.embed.weight

        self.rope = RotaryEmbedding(
            head_dim=config.hidden_size // config.num_heads,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        nn.init.zeros_(self.exit_gate.bias)
        nn.init.normal_(self.exit_gate.weight, mean=0.0, std=std)
        for layer in self.layers:
            for p in layer.parameters():
                if p.dim() >= 2:
                    nn.init.normal_(p, mean=0.0, std=std)

    def forward(
        self,
        input_ids: Tensor,
        num_steps: int | None = None,
        attention_mask: Tensor | None = None,
    ) -> LoopLMOutput:
        """
        Args:
            input_ids:      (B, S) token indices
            num_steps:      number of recurrent steps; defaults to config.max_recurrent_steps
            attention_mask: optional (B, 1, S, S) additive float mask passed to every
                            attention layer.  0.0 = attend, -inf = blocked.
                            When None, standard causal masking is used.

        Returns:
            LoopLMOutput with logits and exit_lambdas for each step
        """
        if num_steps is None:
            num_steps = self.config.max_recurrent_steps

        B, S = input_ids.shape
        cos, sin = self.rope.get_cos_sin(S, input_ids.device)

        h = self.embed(input_ids)  # (B, S, hidden_size)

        logits_per_step: list[Tensor] = []
        exit_lambdas: list[Tensor] = []

        for _ in range(num_steps):
            # Apply the full layer stack (shared weights reused each step)
            for layer in self.layers:
                h = layer(h, cos, sin, attention_mask)

            # LM head output
            logits = self.lm_head(self.final_norm(h))  # (B, S, vocab_size)

            # Exit gate: one probability per token position
            lam = torch.sigmoid(self.exit_gate(h)).squeeze(-1)  # (B, S)

            logits_per_step.append(logits)
            exit_lambdas.append(lam)

        return LoopLMOutput(logits=logits_per_step, exit_lambdas=exit_lambdas)
