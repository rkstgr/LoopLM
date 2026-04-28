"""MoE-Layer Recurrence: route tokens to different TransformerBlock experts per step.

Each expert is a full TransformerBlock with independent weights. A lightweight
router produces per-token logits over N experts, and the top-k outputs are
blended with softmax-normalized weights.

Usage in LoopLM:
    At each recurrence step, instead of applying a fixed layer stack, the
    MoELayer routes each token to its top-k expert blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model.config import LoopLMConfig
from src.model.transformer import TransformerBlock, PostNormTransformerBlock


class MoELayer(nn.Module):
    """Mixture-of-Expert transformer blocks with top-k routing.

    Args:
        config: LoopLM config (defines each expert's architecture).
        num_experts: number of expert TransformerBlocks.
        top_k: number of experts activated per token (1 or 2).
        use_postnorm: use PostNormTransformerBlock instead of sandwich norm.
    """

    def __init__(
        self,
        config: LoopLMConfig,
        num_experts: int = 4,
        top_k: int = 2,
        use_postnorm: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        block_cls = PostNormTransformerBlock if use_postnorm else TransformerBlock
        self.experts = nn.ModuleList(
            [block_cls(config) for _ in range(num_experts)]
        )

        # Router: hidden_size -> num_experts
        self.router = nn.Linear(config.hidden_size, num_experts, bias=False)

    def forward(
        self,
        h: Tensor,
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Route tokens to top-k experts and blend outputs.

        Args:
            h: (B, S, H) hidden states.

        Returns:
            h_out: (B, S, H) blended expert outputs.
            load_balance_loss: scalar auxiliary loss for balanced routing.
        """
        B, S, H = h.shape

        # Router logits and gating weights
        router_logits = self.router(h.detach())  # (B, S, N) — detach to avoid router-through-expert grads
        router_probs = F.softmax(router_logits, dim=-1)  # (B, S, N)

        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # (B, S, k)

        # Re-normalize selected weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Compute all expert outputs (only efficient at small N; fine for testbed)
        expert_outputs = torch.stack(
            [expert(h, cos, sin, attn_mask) for expert in self.experts],
            dim=2,
        )  # (B, S, N, H)

        # Gather selected expert outputs and blend
        # top_k_indices: (B, S, k) -> expand to (B, S, k, H)
        idx_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, H)
        selected = torch.gather(expert_outputs, 2, idx_expanded)  # (B, S, k, H)
        h_out = (selected * top_k_weights.unsqueeze(-1)).sum(dim=2)  # (B, S, H)

        # Load balancing loss (Switch Transformer style)
        load_balance_loss = self._load_balance_loss(router_probs)

        return h_out, load_balance_loss

    def _load_balance_loss(self, router_probs: Tensor) -> Tensor:
        """Compute load balancing loss: alpha * N * sum(f_i * P_i).

        f_i = fraction of tokens routed to expert i
        P_i = mean router probability for expert i
        """
        # f_i: fraction of tokens where expert i is in top-k
        top_k_indices = router_probs.topk(self.top_k, dim=-1).indices  # (B, S, k)
        # One-hot assignment per expert
        one_hot = F.one_hot(top_k_indices, self.num_experts).float()  # (B, S, k, N)
        assignment = one_hot.sum(dim=2)  # (B, S, N) — could be >1 if top_k>1 (shouldn't be for same expert)
        assignment = assignment.clamp(max=1.0)  # binary: was expert selected?
        f = assignment.mean(dim=(0, 1))  # (N,) — fraction per expert

        # P_i: mean router probability per expert
        P = router_probs.mean(dim=(0, 1))  # (N,)

        # Loss: encourages f_i ≈ P_i ≈ 1/N
        loss = self.num_experts * (f * P).sum()
        return loss
