"""src/model/hf_converter.py — Bidirectional weight converter for ByteDance Ouro HuggingFace models.

Provides seamless conversion between official ByteDance Hugging Face weights
(e.g., ByteDance/Ouro-1.4B and ByteDance/Ouro-2.6B) and LoopLM:

    HF Key                                 LoopLM Key
    --------------------------------------------------------------------------------------
    model.embed_tokens.weight       <->    embed.weight
    lm_head.weight                  <->    lm_head.weight
    model.norm.weight               <->    final_norm.weight
    model.early_exit_gate.weight    <->    exit_gate.weight
    model.early_exit_gate.bias      <->    exit_gate.bias
    model.layers.i.input_layernorm  <->    layers.i.pre_attn_norm
    model.layers.i.self_attn.*      <->    layers.i.attn.*
    model.layers.i.input_layernorm_2<->    layers.i.post_attn_norm
    model.layers.i.post_attention_layernorm <-> layers.i.pre_ffn_norm
    model.layers.i.mlp.gate_proj    <->    layers.i.ffn.w_gate
    model.layers.i.mlp.up_proj      <->    layers.i.ffn.w_up
    model.layers.i.mlp.down_proj    <->    layers.i.ffn.w_down
    model.layers.i.post_attention_layernorm_2 <-> layers.i.post_ffn_norm
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM
from src.utils import load_checkpoint_safe


# ── Mapping tables ─────────────────────────────────────────────────────────────

# Per-layer sub-component mapping (HF layer suffix -> LoopLM layer suffix)
_HF_TO_LOOPLM_LAYER_MAP: dict[str, str] = {
    "input_layernorm.weight": "pre_attn_norm.weight",
    "self_attn.q_proj.weight": "attn.q_proj.weight",
    "self_attn.k_proj.weight": "attn.k_proj.weight",
    "self_attn.v_proj.weight": "attn.v_proj.weight",
    "self_attn.o_proj.weight": "attn.o_proj.weight",
    "input_layernorm_2.weight": "post_attn_norm.weight",
    "post_attention_layernorm.weight": "pre_ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn.w_gate.weight",
    "mlp.up_proj.weight": "ffn.w_up.weight",
    "mlp.down_proj.weight": "ffn.w_down.weight",
    "post_attention_layernorm_2.weight": "post_ffn_norm.weight",
}

_LOOPLM_TO_HF_LAYER_MAP: dict[str, str] = {
    v: k for k, v in _HF_TO_LOOPLM_LAYER_MAP.items()
}

# Global parameter mapping (non-layer)
_HF_TO_LOOPLM_GLOBAL_MAP: dict[str, str] = {
    "model.embed_tokens.weight": "embed.weight",
    "lm_head.weight": "lm_head.weight",
    "model.norm.weight": "final_norm.weight",
    "model.early_exit_gate.weight": "exit_gate.weight",
    "model.early_exit_gate.bias": "exit_gate.bias",
}

_LOOPLM_TO_HF_GLOBAL_MAP: dict[str, str] = {
    v: k for k, v in _HF_TO_LOOPLM_GLOBAL_MAP.items()
}


# ── State dict conversion ──────────────────────────────────────────────────────

def hf_to_looplm_state_dict(
    hf_state_dict: dict[str, Tensor],
    config: LoopLMConfig | None = None,
) -> tuple[dict[str, Tensor], LoopLMConfig]:
    """Convert a ByteDance Ouro HuggingFace state dict to LoopLM format.

    Args:
        hf_state_dict: state dict from HuggingFace OuroForCausalLM.
        config:        optional LoopLMConfig; if None, inferred from tensor shapes.

    Returns:
        (looplm_state_dict, looplm_config)
    """
    looplm_dict: dict[str, Tensor] = {}
    detected_layers: set[int] = set()

    for hf_key, tensor in hf_state_dict.items():
        # Global non-layer weights
        if hf_key in _HF_TO_LOOPLM_GLOBAL_MAP:
            target_key = _HF_TO_LOOPLM_GLOBAL_MAP[hf_key]
            looplm_dict[target_key] = tensor
            continue

        # Per-layer weights: "model.layers.{idx}.{suffix}"
        if hf_key.startswith("model.layers."):
            parts = hf_key.split(".", 3)
            layer_idx = int(parts[2])
            suffix = parts[3]
            detected_layers.add(layer_idx)

            if suffix in _HF_TO_LOOPLM_LAYER_MAP:
                looplm_suffix = _HF_TO_LOOPLM_LAYER_MAP[suffix]
                looplm_key = f"layers.{layer_idx}.{looplm_suffix}"
                looplm_dict[looplm_key] = tensor
            else:
                # Retain unrecognized layer parameters
                looplm_dict[f"layers.{layer_idx}.{suffix}"] = tensor
        else:
            # Pass-through any other custom weights
            looplm_dict[hf_key] = tensor

    # Infer or validate LoopLMConfig
    if config is None:
        vocab_size = looplm_dict["embed.weight"].shape[0]
        hidden_size = looplm_dict["embed.weight"].shape[1]
        num_layers = max(detected_layers) + 1 if detected_layers else 24
        intermediate_size = looplm_dict.get("layers.0.ffn.w_gate.weight", torch.zeros(5632, hidden_size)).shape[0]
        num_heads = hidden_size // 128  # standard Ouro head_dim is 128
        tie_weights = ("lm_head.weight" not in looplm_dict)

        config = LoopLMConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_seq_len=4096,
            max_recurrent_steps=4,
            rope_base=10000.0,
            tie_word_embeddings=tie_weights,
        )

    # Ensure lm_head.weight exists if tie_word_embeddings is False
    if not config.tie_word_embeddings and "lm_head.weight" not in looplm_dict:
        looplm_dict["lm_head.weight"] = looplm_dict["embed.weight"].clone()

    return looplm_dict, config


def looplm_to_hf_state_dict(
    looplm_state_dict: dict[str, Tensor],
    config: LoopLMConfig | None = None,
) -> dict[str, Tensor]:
    """Convert a LoopLM state dict to ByteDance Ouro HuggingFace format.

    Args:
        looplm_state_dict: state dict from LoopLM model.
        config:            LoopLMConfig.

    Returns:
        HuggingFace-compatible state dict for OuroForCausalLM.
    """
    hf_dict: dict[str, Tensor] = {}

    for looplm_key, tensor in looplm_state_dict.items():
        if looplm_key in _LOOPLM_TO_HF_GLOBAL_MAP:
            target_key = _LOOPLM_TO_HF_GLOBAL_MAP[looplm_key]
            hf_dict[target_key] = tensor
            continue

        if looplm_key.startswith("layers."):
            parts = looplm_key.split(".", 2)
            layer_idx = int(parts[1])
            suffix = parts[2]
            if suffix in _LOOPLM_TO_HF_LAYER_MAP:
                hf_suffix = _LOOPLM_TO_HF_LAYER_MAP[suffix]
                hf_key = f"model.layers.{layer_idx}.{hf_suffix}"
                hf_dict[hf_key] = tensor
            else:
                hf_dict[f"model.layers.{layer_idx}.{suffix}"] = tensor
        else:
            hf_dict[looplm_key] = tensor

    # Always ensure lm_head.weight is populated in HF format
    if "lm_head.weight" not in hf_dict and "model.embed_tokens.weight" in hf_dict:
        hf_dict["lm_head.weight"] = hf_dict["model.embed_tokens.weight"].clone()

    return hf_dict


# ── Config conversion ──────────────────────────────────────────────────────────

def hf_config_to_looplm_config(hf_config: dict[str, Any]) -> LoopLMConfig:
    """Convert an Ouro config.json dictionary to a LoopLMConfig instance."""
    hidden_size = hf_config.get("hidden_size", 2048)
    head_dim = hf_config.get("head_dim", 128)
    num_heads = hf_config.get("num_attention_heads", hidden_size // head_dim)

    return LoopLMConfig(
        vocab_size=hf_config.get("vocab_size", 49152),
        hidden_size=hidden_size,
        num_layers=hf_config.get("num_hidden_layers", 24),
        num_heads=num_heads,
        intermediate_size=hf_config.get("intermediate_size", 5632),
        max_seq_len=hf_config.get("max_position_embeddings", 4096),
        max_recurrent_steps=hf_config.get("total_ut_steps", 4),
        rope_base=hf_config.get("rope_theta", 10000.0),
        tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
    )


def looplm_config_to_hf_config(config: LoopLMConfig) -> dict[str, Any]:
    """Convert a LoopLMConfig instance to an official Ouro config.json dict."""
    head_dim = config.hidden_size // config.num_heads
    return {
        "architectures": ["OuroForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_ouro.OuroConfig",
            "AutoModel": "modeling_ouro.OuroModel",
            "AutoModelForCausalLM": "modeling_ouro.OuroForCausalLM",
        },
        "attention_dropout": config.dropout,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "hidden_size": config.hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": config.intermediate_size,
        "layer_types": ["full_attention"] * config.num_layers,
        "max_position_embeddings": config.max_seq_len,
        "max_window_layers": config.num_layers,
        "model_type": "ouro",
        "num_attention_heads": config.num_heads,
        "num_hidden_layers": config.num_layers,
        "num_key_value_heads": config.num_heads,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "rope_theta": config.rope_base,
        "sliding_window": None,
        "tie_word_embeddings": config.tie_word_embeddings,
        "torch_dtype": "bfloat16",
        "total_ut_steps": config.max_recurrent_steps,
        "early_exit_threshold": 1.0,
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": config.vocab_size,
    }


# ── Loading and Saving ────────────────────────────────────────────────────────

def load_from_hf(
    repo_id_or_path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> LoopLM:
    """Load a ByteDance Ouro model from a local directory or HuggingFace Hub.

    Supports both single safetensors / pytorch_model.bin files and sharded checkpoints.

    Args:
        repo_id_or_path: local directory path or HuggingFace repository ID
                         (e.g., 'ByteDance/Ouro-1.4B').
        device:          target device (e.g., 'cpu', 'cuda').
        dtype:           target torch dtype.

    Returns:
        Ready-to-use LoopLM model with loaded weights.
    """
    import safetensors.torch

    local_path = Path(repo_id_or_path)

    # 1. Resolve files (local folder vs HuggingFace Hub)
    if not local_path.exists():
        from huggingface_hub import snapshot_download
        print(f"Downloading {repo_id_or_path} from HuggingFace Hub...")
        downloaded_dir = snapshot_download(repo_id=str(repo_id_or_path))
        local_path = Path(downloaded_dir)

    # 2. Read config.json
    config_file = local_path / "config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            hf_cfg = json.load(f)
        looplm_config = hf_config_to_looplm_config(hf_cfg)
    else:
        looplm_config = None

    # 3. Load state dict
    raw_state_dict: dict[str, Tensor] = {}

    single_safetensors = local_path / "model.safetensors"
    single_bin = local_path / "pytorch_model.bin"
    sharded_safetensors_index = local_path / "model.safetensors.index.json"

    if single_safetensors.exists():
        raw_state_dict = safetensors.torch.load_file(str(single_safetensors), device="cpu")
    elif sharded_safetensors_index.exists():
        with open(sharded_safetensors_index, "r") as f:
            index_data = json.load(f)
        shard_files = set(index_data["weight_map"].values())
        for shard in shard_files:
            shard_path = local_path / shard
            shard_dict = safetensors.torch.load_file(str(shard_path), device="cpu")
            raw_state_dict.update(shard_dict)
    elif single_bin.exists():
        raw_state_dict = load_checkpoint_safe(single_bin, map_location="cpu")
    else:
        # Check any .safetensors in directory
        all_st = list(local_path.glob("*.safetensors"))
        if all_st:
            for st in all_st:
                raw_state_dict.update(safetensors.torch.load_file(str(st), device="cpu"))
        else:
            raise FileNotFoundError(f"No model weights (.safetensors or .bin) found in {local_path}")

    # 4. Convert state dict
    looplm_state, final_config = hf_to_looplm_state_dict(raw_state_dict, config=looplm_config)

    # 5. Instantiate LoopLM and load weights
    model = LoopLM(final_config).to(device=device, dtype=dtype)
    model.load_state_dict(looplm_state)
    model.eval()

    return model


def save_to_hf(
    model: LoopLM,
    save_dir: str | Path,
    safe_serialization: bool = True,
) -> Path:
    """Save a LoopLM model in official ByteDance Ouro HuggingFace format.

    Writes config.json and model.safetensors (or pytorch_model.bin) to save_dir.

    Args:
        model:              LoopLM model to export.
        save_dir:           target folder path.
        safe_serialization: whether to save as .safetensors (recommended).

    Returns:
        Path to save directory.
    """
    import safetensors.torch

    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convert config and save config.json
    hf_config = looplm_config_to_hf_config(model.config)
    with open(target_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # 2. Convert state dict
    hf_state_dict = looplm_to_hf_state_dict(model.state_dict(), model.config)

    # 3. Save weights
    if safe_serialization:
        weights_path = target_dir / "model.safetensors"
        safetensors.torch.save_file(hf_state_dict, str(weights_path))
    else:
        weights_path = target_dir / "pytorch_model.bin"
        torch.save(hf_state_dict, weights_path)

    return target_dir
