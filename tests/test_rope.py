import torch
import pytest
from src.model.rope import RotaryEmbedding, apply_rope, rotate_half


@pytest.fixture
def rope():
    return RotaryEmbedding(head_dim=64, max_seq_len=128, base=10000.0)


def test_cache_shape(rope):
    cos, sin = rope.get_cos_sin(seq_len=32, device=torch.device("cpu"))
    assert cos.shape == (32, 32)  # (seq_len, head_dim/2)
    assert sin.shape == (32, 32)


def test_apply_rope_output_shape(rope):
    batch, heads, seq_len, head_dim = 2, 4, 32, 64
    x = torch.randn(batch, heads, seq_len, head_dim)
    cos, sin = rope.get_cos_sin(seq_len, x.device)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape


def test_rotation_preserves_norm(rope):
    """RoPE is an isometry — it must not change the norm of each vector."""
    x = torch.randn(2, 4, 16, 64)
    cos, sin = rope.get_cos_sin(16, x.device)
    out = apply_rope(x, cos, sin)
    torch.testing.assert_close(
        x.norm(dim=-1), out.norm(dim=-1), rtol=1e-4, atol=1e-4
    )


def test_position_zero_is_identity(rope):
    """At position 0, cos=1 and sin=0, so RoPE should be identity."""
    x = torch.randn(1, 1, 1, 64)
    cos, sin = rope.get_cos_sin(1, x.device)
    out = apply_rope(x, cos, sin)
    torch.testing.assert_close(x, out, rtol=1e-5, atol=1e-5)


def test_different_positions_give_different_rotations(rope):
    """Different sequence positions must produce different outputs for the same input."""
    x = torch.ones(1, 1, 4, 64)
    cos, sin = rope.get_cos_sin(4, x.device)
    out = apply_rope(x, cos, sin)
    # Rows at different positions should differ
    assert not torch.allclose(out[0, 0, 0], out[0, 0, 1])


def test_set_base_updates_cache(rope):
    cos_before, _ = rope.get_cos_sin(16, torch.device("cpu"))
    rope.set_base(40000.0)
    cos_after, _ = rope.get_cos_sin(16, torch.device("cpu"))
    assert not torch.allclose(cos_before, cos_after)


def test_auto_extends_cache_for_longer_sequence(rope):
    """Cache should grow automatically when seq_len > max_seq_len."""
    cos, sin = rope.get_cos_sin(256, torch.device("cpu"))  # exceeds initial 128
    assert cos.shape[0] == 256


def test_gradients_flow_through_apply_rope(rope):
    x = torch.randn(2, 4, 16, 64, requires_grad=True)
    cos, sin = rope.get_cos_sin(16, x.device)
    out = apply_rope(x, cos, sin)
    out.sum().backward()
    assert x.grad is not None
    assert not torch.all(x.grad == 0)
