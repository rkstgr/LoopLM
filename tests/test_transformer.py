import torch
import pytest
from src.model.config import LoopLMConfig
from src.model.rope import RotaryEmbedding
from src.model.transformer import RMSNorm, SwiGLUFFN, MultiHeadAttention, TransformerBlock


@pytest.fixture
def cfg():
    return LoopLMConfig.small()


@pytest.fixture
def rope(cfg):
    return RotaryEmbedding(head_dim=cfg.hidden_size // cfg.num_heads, max_seq_len=64)


@pytest.fixture
def cos_sin(rope):
    cos, sin = rope.get_cos_sin(seq_len=16, device=torch.device("cpu"))
    return cos, sin


# ── RMSNorm ──────────────────────────────────────────────────────────────────

def test_rmsnorm_output_shape():
    norm = RMSNorm(64)
    x = torch.randn(2, 16, 64)
    assert norm(x).shape == x.shape


def test_rmsnorm_unit_rms():
    """After RMSNorm (with weight=1), the RMS of each vector should be ~1."""
    norm = RMSNorm(64)
    x = torch.randn(4, 16, 64) * 10  # large scale to make the test meaningful
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-4, rtol=1e-4)


# ── SwiGLUFFN ────────────────────────────────────────────────────────────────

def test_swiglu_output_shape(cfg):
    ffn = SwiGLUFFN(cfg.hidden_size, cfg.intermediate_size)
    x = torch.randn(2, 16, cfg.hidden_size)
    assert ffn(x).shape == x.shape


def test_swiglu_gradients(cfg):
    ffn = SwiGLUFFN(cfg.hidden_size, cfg.intermediate_size)
    x = torch.randn(2, 8, cfg.hidden_size, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad is not None
    for name, p in ffn.named_parameters():
        assert p.grad is not None, f"No grad for {name}"


# ── MultiHeadAttention ───────────────────────────────────────────────────────

def test_attention_output_shape(cfg, cos_sin):
    cos, sin = cos_sin
    attn = MultiHeadAttention(cfg.hidden_size, cfg.num_heads)
    x = torch.randn(2, 16, cfg.hidden_size)
    assert attn(x, cos, sin).shape == x.shape


def test_attention_causal_masking(cfg, rope):
    """Changing a future token must not affect past token outputs."""
    cos, sin = rope.get_cos_sin(8, torch.device("cpu"))
    attn = MultiHeadAttention(cfg.hidden_size, cfg.num_heads)
    attn.eval()

    x = torch.randn(1, 8, cfg.hidden_size)
    out1 = attn(x, cos, sin)

    x2 = x.clone()
    x2[0, 5:] = torch.randn(3, cfg.hidden_size)  # modify future tokens
    out2 = attn(x2, cos, sin)

    # First 5 positions should be identical
    torch.testing.assert_close(out1[0, :5], out2[0, :5], atol=1e-5, rtol=1e-5)


# ── TransformerBlock ─────────────────────────────────────────────────────────

def test_block_output_shape(cfg, cos_sin):
    cos, sin = cos_sin
    block = TransformerBlock(cfg)
    x = torch.randn(2, 16, cfg.hidden_size)
    assert block(x, cos, sin).shape == x.shape


def test_block_gradients_all_params(cfg, cos_sin):
    """Gradients must reach every parameter in the block."""
    cos, sin = cos_sin
    block = TransformerBlock(cfg)
    x = torch.randn(2, 8, cfg.hidden_size, requires_grad=True)
    block(x, cos, sin).sum().backward()
    assert x.grad is not None
    for name, p in block.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert not torch.all(p.grad == 0), f"Zero grad for {name}"


def test_block_residual_connection(cfg, cos_sin):
    """Output must differ from input (residual adds non-trivial updates)."""
    cos, sin = cos_sin
    block = TransformerBlock(cfg)
    x = torch.randn(2, 16, cfg.hidden_size)
    out = block(x, cos, sin)
    assert not torch.allclose(x, out)


def test_block_sandwich_norm_count(cfg):
    """Verify there are exactly 4 RMSNorm layers (sandwich = pre+post for attn and ffn)."""
    from src.model.transformer import RMSNorm
    block = TransformerBlock(cfg)
    norms = [m for m in block.modules() if isinstance(m, RMSNorm)]
    assert len(norms) == 4
