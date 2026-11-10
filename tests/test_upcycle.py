"""tests/test_upcycle.py — Unit tests for model upcycling (1.4B -> 2.6B) and ByteDance Ouro HF converter."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.model.upcycle import upcycle_model, upcycle_state_dict, upcycle_checkpoint
from src.model.hf_converter import (
    hf_to_looplm_state_dict,
    looplm_to_hf_state_dict,
    hf_config_to_looplm_config,
    looplm_config_to_hf_config,
    load_from_hf,
    save_to_hf,
)
from src.training.trainer import Trainer, TrainerConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_config(num_layers: int = 2) -> LoopLMConfig:
    return LoopLMConfig(
        vocab_size=128,
        hidden_size=64,
        num_layers=num_layers,
        num_heads=4,
        intermediate_size=128,
        max_seq_len=32,
        max_recurrent_steps=4,
        rope_base=10000.0,
        tie_word_embeddings=False,
    )


# ── 1. Model Upcycling Tests ─────────────────────────────────────────────────

def test_upcycle_model_doubles_layers():
    """Upcycling a 2-layer model should yield a 4-layer model with duplicated weights."""
    source = LoopLM(_tiny_config(num_layers=2))
    upcycled = upcycle_model(source)

    assert upcycled.config.num_layers == 4
    assert len(upcycled.layers) == 4

    # Verify layer weights are duplicated: layer 0 == layer 2, layer 1 == layer 3
    for p0, p2 in zip(upcycled.layers[0].parameters(), upcycled.layers[2].parameters()):
        assert torch.equal(p0, p2)

    for p1, p3 in zip(upcycled.layers[1].parameters(), upcycled.layers[3].parameters()):
        assert torch.equal(p1, p3)


def test_upcycle_model_preserves_non_layer_weights():
    """Embeddings, LM head, final norm, and exit gate must be identical."""
    source = LoopLM(_tiny_config(num_layers=2))
    upcycled = upcycle_model(source)

    assert torch.equal(source.embed.weight, upcycled.embed.weight)
    assert torch.equal(source.lm_head.weight, upcycled.lm_head.weight)
    assert torch.equal(source.final_norm.weight, upcycled.final_norm.weight)
    assert torch.equal(source.exit_gate.weight, upcycled.exit_gate.weight)
    assert torch.equal(source.exit_gate.bias, upcycled.exit_gate.bias)


def test_upcycle_model_forward_pass():
    """Forward pass through upcycled model should produce valid finite outputs."""
    source = LoopLM(_tiny_config(num_layers=2))
    upcycled = upcycle_model(source)

    x = torch.randint(0, 128, (2, 8))
    out = upcycled(x, num_steps=3)

    assert len(out.logits) == 3
    assert len(out.exit_lambdas) == 3
    for logits in out.logits:
        assert logits.shape == (2, 8, 128)
        assert torch.isfinite(logits).all()
    for lam in out.exit_lambdas:
        assert lam.shape == (2, 8)
        assert torch.isfinite(lam).all()
        assert (lam >= 0.0).all() and (lam <= 1.0).all()


def test_upcycle_model_gradient_flow():
    """Backprop through upcycled model must compute finite gradients for all duplicated layers."""
    source = LoopLM(_tiny_config(num_layers=2))
    upcycled = upcycle_model(source)

    x = torch.randint(0, 128, (2, 8))
    out = upcycled(x, num_steps=2)
    loss = out.logits[-1].sum()
    loss.backward()

    for idx, layer in enumerate(upcycled.layers):
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"Layer {idx} param {name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"Layer {idx} param {name} gradient has NaNs/Infs"


def test_upcycle_arbitrary_target_layers():
    """Target layer count can be specified explicitly (e.g. 3 -> 6 or 3 -> 9)."""
    source = LoopLM(_tiny_config(num_layers=3))
    upcycled_6 = upcycle_model(source, target_layers=6)
    assert upcycled_6.config.num_layers == 6
    assert len(upcycled_6.layers) == 6

    # Test error when target layers is not strictly greater
    with pytest.raises(ValueError):
        upcycle_model(source, target_layers=3)
    with pytest.raises(ValueError):
        upcycle_model(source, target_layers=2)


def test_ouro_1_4b_to_2_6b_config_upcycling():
    """Upcycling Ouro-1.4B config should scale to 48 layers (Ouro-2.6B specs)."""
    cfg_1_4b = LoopLMConfig.ouro_1_4b()
    assert cfg_1_4b.num_layers == 24
    params_1_4b = cfg_1_4b.num_parameters()
    assert 1.3e9 < params_1_4b < 1.6e9  # ~1.4B parameters

    # Upcycled state dict target config
    dummy_dict = {
        "embed.weight": torch.zeros(cfg_1_4b.vocab_size, cfg_1_4b.hidden_size),
        "lm_head.weight": torch.zeros(cfg_1_4b.vocab_size, cfg_1_4b.hidden_size),
        "final_norm.weight": torch.zeros(cfg_1_4b.hidden_size),
        "exit_gate.weight": torch.zeros(1, cfg_1_4b.hidden_size),
        "exit_gate.bias": torch.zeros(1),
    }
    for i in range(24):
        dummy_dict[f"layers.{i}.attn.q_proj.weight"] = torch.zeros(cfg_1_4b.hidden_size, cfg_1_4b.hidden_size)

    upcycled_dict, target_cfg = upcycle_state_dict(dummy_dict, cfg_1_4b)
    assert target_cfg.num_layers == 48
    params_2_6b = target_cfg.num_parameters()
    assert 2.5e9 < params_2_6b < 2.9e9  # ~2.6B parameters
    assert "layers.47.attn.q_proj.weight" in upcycled_dict


def test_upcycle_checkpoint_roundtrip(tmp_path):
    """Saving a checkpoint, upcycling on disk, and resuming training should work seamlessly."""
    cfg = _tiny_config(num_layers=2)
    trainer_cfg = TrainerConfig(max_steps=5, checkpoint_dir=str(tmp_path / "ckpts"), device="cpu")
    trainer = Trainer(cfg, trainer_cfg)

    # Save initial 2-layer checkpoint
    ckpt_path = trainer.save_checkpoint()
    assert ckpt_path.exists()

    # Upcycle to 4 layers on disk
    upcycled_ckpt_path = tmp_path / "ckpts" / "upcycled_step_0000000.pt"
    upcycle_checkpoint(ckpt_path, upcycled_ckpt_path)
    assert upcycled_ckpt_path.exists()

    # Verify upcycled checkpoint can be loaded into a 4-layer model/trainer
    cfg_4 = _tiny_config(num_layers=4)
    trainer_4 = Trainer(cfg_4, trainer_cfg)
    trainer_4.load_checkpoint(upcycled_ckpt_path)

    assert trainer_4.raw_model.config.num_layers == 4
    # Run a training step to verify operational sanity
    batch = torch.randint(0, 128, (2, 9))
    diag = trainer_4.train_step(batch)
    assert torch.isfinite(diag["loss"])


# ── 2. Hugging Face Converter Tests ──────────────────────────────────────────

def test_hf_converter_bidirectional_state_dict():
    """State dict conversion from LoopLM -> HF -> LoopLM should be bitwise lossless."""
    model = LoopLM(_tiny_config(num_layers=2))
    orig_dict = model.state_dict()

    # LoopLM -> HF
    hf_dict = looplm_to_hf_state_dict(orig_dict, model.config)
    assert "model.embed_tokens.weight" in hf_dict
    assert "model.layers.0.self_attn.q_proj.weight" in hf_dict
    assert "model.layers.1.mlp.gate_proj.weight" in hf_dict
    assert "model.early_exit_gate.weight" in hf_dict

    # HF -> LoopLM
    restored_dict, restored_cfg = hf_to_looplm_state_dict(hf_dict, model.config)

    assert set(orig_dict.keys()) == set(restored_dict.keys())
    for k in orig_dict:
        assert torch.equal(orig_dict[k], restored_dict[k]), f"Mismatch in restored tensor {k}"


def test_hf_config_conversion_roundtrip():
    """LoopLMConfig -> HF config dict -> LoopLMConfig should preserve all parameters."""
    cfg = _tiny_config(num_layers=3)
    cfg.rope_base = 40000.0

    hf_cfg = looplm_config_to_hf_config(cfg)
    assert hf_cfg["architectures"] == ["OuroForCausalLM"]
    assert hf_cfg["num_hidden_layers"] == 3
    assert hf_cfg["rope_theta"] == 40000.0

    restored_cfg = hf_config_to_looplm_config(hf_cfg)
    assert restored_cfg.vocab_size == cfg.vocab_size
    assert restored_cfg.hidden_size == cfg.hidden_size
    assert restored_cfg.num_layers == cfg.num_layers
    assert restored_cfg.num_heads == cfg.num_heads
    assert restored_cfg.intermediate_size == cfg.intermediate_size
    assert restored_cfg.rope_base == pytest.approx(cfg.rope_base)


def test_hf_safetensors_save_and_load(tmp_path):
    """Saving to HF format with safetensors and loading back should produce identical forward passes."""
    model = LoopLM(_tiny_config(num_layers=2)).eval()
    x = torch.randint(0, 128, (2, 8))

    with torch.no_grad():
        orig_out = model(x, num_steps=2)

    hf_dir = tmp_path / "ouro_hf"
    save_to_hf(model, hf_dir, safe_serialization=True)

    assert (hf_dir / "config.json").exists()
    assert (hf_dir / "model.safetensors").exists()

    # Load back using load_from_hf
    loaded_model = load_from_hf(hf_dir, device="cpu", dtype=torch.float32)
    assert loaded_model.config.num_layers == 2

    with torch.no_grad():
        loaded_out = loaded_model(x, num_steps=2)

    for orig_logits, loaded_logits in zip(orig_out.logits, loaded_out.logits):
        assert torch.allclose(orig_logits, loaded_logits, atol=1e-5)


# ── 3. CLI Smoke Tests ───────────────────────────────────────────────────────

def test_upcycle_cli(tmp_path):
    """Test running scripts/upcycle.py CLI."""
    cfg = _tiny_config(num_layers=2)
    trainer_cfg = TrainerConfig(max_steps=1, checkpoint_dir=str(tmp_path), device="cpu")
    trainer = Trainer(cfg, trainer_cfg)
    src_ckpt = trainer.save_checkpoint()

    dst_ckpt = tmp_path / "upcycled_cli.pt"
    cmd = [
        sys.executable,
        "scripts/upcycle.py",
        "--input", str(src_ckpt),
        "--output", str(dst_ckpt),
        "--target-layers", "4",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI error: {res.stderr}"
    assert dst_ckpt.exists()

    ckpt_data = torch.load(dst_ckpt, map_location="cpu", weights_only=False)
    assert ckpt_data["model_config"]["num_layers"] == 4


def test_convert_ouro_hf_cli(tmp_path):
    """Test running scripts/convert_ouro_hf.py CLI in both directions."""
    cfg = _tiny_config(num_layers=2)
    trainer_cfg = TrainerConfig(max_steps=1, checkpoint_dir=str(tmp_path), device="cpu")
    trainer = Trainer(cfg, trainer_cfg)
    src_ckpt = trainer.save_checkpoint()

    # 1. LoopLM -> HF directory
    hf_dir = tmp_path / "hf_cli_export"
    cmd1 = [
        sys.executable,
        "scripts/convert_ouro_hf.py",
        "--input-checkpoint", str(src_ckpt),
        "--output-dir", str(hf_dir),
        "--to-hf",
    ]
    res1 = subprocess.run(cmd1, capture_output=True, text=True)
    assert res1.returncode == 0, f"To-HF error: {res1.stderr}"
    assert (hf_dir / "config.json").exists()
    assert (hf_dir / "model.safetensors").exists()

    # 2. HF directory -> LoopLM checkpoint
    converted_ckpt = tmp_path / "converted_from_hf.pt"
    cmd2 = [
        sys.executable,
        "scripts/convert_ouro_hf.py",
        "--hf-model", str(hf_dir),
        "--output", str(converted_ckpt),
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert res2.returncode == 0, f"From-HF error: {res2.stderr}"
    assert converted_ckpt.exists()
