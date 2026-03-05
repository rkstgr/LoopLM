import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model.config import LoopLMConfig
from src.model.rope import apply_rope


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: out = W_down · (silu(W_gate · x) * (W_up · x))"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, num_heads, S, head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if attn_mask is not None:
            # Custom mask already encodes causality; disable built-in causal mask
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        else:
            # Default: standard causal attention
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # out: (B, num_heads, S, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Decoder transformer block with sandwich normalization.

    Sandwich norm wraps each sublayer with RMSNorm both before and after,
    which bounds the residual updates and is critical for recurrent stability:

        x = x + post_norm(sublayer(pre_norm(x)))
    """

    def __init__(self, config: LoopLMConfig):
        super().__init__()
        self.pre_attn_norm = RMSNorm(config.hidden_size)
        self.attn = MultiHeadAttention(config.hidden_size, config.num_heads)
        self.post_attn_norm = RMSNorm(config.hidden_size)

        self.pre_ffn_norm = RMSNorm(config.hidden_size)
        self.ffn = SwiGLUFFN(config.hidden_size, config.intermediate_size)
        self.post_ffn_norm = RMSNorm(config.hidden_size)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        # Attention with sandwich norm
        x = x + self.post_attn_norm(self.attn(self.pre_attn_norm(x), cos, sin, attn_mask))
        # FFN with sandwich norm
        x = x + self.post_ffn_norm(self.ffn(self.pre_ffn_norm(x)))
        return x
