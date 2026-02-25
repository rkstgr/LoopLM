import torch
from torch import Tensor


class RotaryEmbedding:
    """Rotary Position Embeddings (RoPE).

    Precomputes a cos/sin cache up to max_seq_len. The base frequency is
    configurable so it can be increased to 40K or 1M in later training stages
    without recreating the model.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        self._cos: Tensor | None = None
        self._sin: Tensor | None = None
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        # θ_i = base^(-2i / head_dim) for i in [0, head_dim/2)
        i = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        freqs = 1.0 / (self.base ** (i / self.head_dim))  # (head_dim/2,)

        positions = torch.arange(seq_len, dtype=torch.float32)  # (seq_len,)
        angles = torch.outer(positions, freqs)  # (seq_len, head_dim/2)

        self._cos = torch.cos(angles)  # (seq_len, head_dim/2)
        self._sin = torch.sin(angles)  # (seq_len, head_dim/2)

    def set_base(self, base: float) -> None:
        """Update base frequency (e.g. 10K → 40K → 1M across training stages)."""
        self.base = base
        self._build_cache(self.max_seq_len)

    def get_cos_sin(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return cos/sin tensors for the given sequence length."""
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self._build_cache(seq_len)
        cos = self._cos[:seq_len].to(device)  # (seq_len, head_dim/2)
        sin = self._sin[:seq_len].to(device)
        return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    """Rotate the second half of the last dimension to implement complex multiplication."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to a query or key tensor.

    Args:
        x:   (batch, num_heads, seq_len, head_dim)
        cos: (seq_len, head_dim/2)
        sin: (seq_len, head_dim/2)

    Returns:
        Rotated tensor of same shape as x.
    """
    seq_len = x.shape[2]
    # Slice to actual sequence length, then expand to full head_dim
    # [cos_0..cos_{d/2}, cos_0..cos_{d/2}] — matches split-half rotate_half convention
    cos = torch.cat([cos[:seq_len], cos[:seq_len]], dim=-1)  # (seq_len, head_dim)
    sin = torch.cat([sin[:seq_len], sin[:seq_len]], dim=-1)  # (seq_len, head_dim)

    # Broadcast over batch and heads: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return x * cos + rotate_half(x) * sin
