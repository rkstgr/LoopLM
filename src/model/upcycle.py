"""src/model/upcycle.py — Model upcycling (layer duplication) for LoopLM.

Following Section 4.3.1 of the Ouro paper (arXiv:2510.25741):
In Stage 1b, Ouro scales from 1.4B (24 layers) to 2.6B (48 layers) via layer
duplication. Because the weights are trained recurrently, the output of layer 23
is already in the distribution expected by layer 0, making layer duplication
seamless and avoiding the representation shock seen in standard non-recurrent transformers.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM


def upcycle_state_dict(
    source_state_dict: dict[str, Tensor],
    source_config: LoopLMConfig,
    target_layers: int | None = None,
) -> tuple[dict[str, Tensor], LoopLMConfig]:
    """Upcycle a LoopLM state dict by duplicating its recurrent layers.

    Memory-efficient: works directly on state dict tensors without instantiating
    both models simultaneously.

    Args:
        source_state_dict: state dict of the source LoopLM model.
        source_config:     LoopLMConfig of the source model.
        target_layers:     number of layers in target model (default: 2 * source_config.num_layers).

    Returns:
        (target_state_dict, target_config)
    """
    src_layers = source_config.num_layers
    dst_layers = target_layers if target_layers is not None else (src_layers * 2)

    if dst_layers <= src_layers:
        raise ValueError(
            f"target_layers ({dst_layers}) must be greater than source layers ({src_layers})"
        )

    target_config = copy.deepcopy(source_config)
    target_config.num_layers = dst_layers

    target_state_dict: dict[str, Tensor] = {}

    # Copy non-layer parameters (embeddings, norms, heads, exit gates)
    for key, tensor in source_state_dict.items():
        if not key.startswith("layers."):
            target_state_dict[key] = tensor.clone()

    # Find all parameter suffixes within a layer
    layer_param_suffixes: dict[int, list[str]] = {}
    for key in source_state_dict:
        if key.startswith("layers."):
            parts = key.split(".", 2)
            layer_idx = int(parts[1])
            suffix = parts[2]
            layer_param_suffixes.setdefault(layer_idx, []).append(suffix)

    # Replicate layers into the target state dict
    for dst_idx in range(dst_layers):
        src_idx = dst_idx % src_layers
        suffixes = layer_param_suffixes.get(src_idx, [])
        for suffix in suffixes:
            src_key = f"layers.{src_idx}.{suffix}"
            dst_key = f"layers.{dst_idx}.{suffix}"
            target_state_dict[dst_key] = source_state_dict[src_key].clone()

    return target_state_dict, target_config


def upcycle_model(
    source_model: LoopLM,
    target_layers: int | None = None,
) -> LoopLM:
    """Upcycle an instantiated LoopLM model by duplicating its transformer layers.

    Args:
        source_model:  trained or initialized source LoopLM.
        target_layers: number of layers in target model (default: 2 * source_model.config.num_layers).

    Returns:
        New LoopLM instance with target_layers and duplicated weights.
    """
    target_dict, target_config = upcycle_state_dict(
        source_model.state_dict(),
        source_model.config,
        target_layers=target_layers,
    )

    device = next(source_model.parameters()).device
    dtype = next(source_model.parameters()).dtype

    target_model = LoopLM(
        target_config,
        use_q_act=source_model.use_q_act,
        use_moe_recurrence=source_model.use_moe_recurrence,
    ).to(device=device, dtype=dtype)

    target_model.load_state_dict(target_dict)
    return target_model


def upcycle_checkpoint(
    source_checkpoint_path: str | Path,
    target_checkpoint_path: str | Path,
    target_layers: int | None = None,
) -> Path:
    """Load a LoopLM checkpoint, upcycle its layers, and save the resulting checkpoint.

    Supports both full Trainer checkpoints and raw state dict checkpoints.

    Args:
        source_checkpoint_path: path to input .pt checkpoint.
        target_checkpoint_path: path to write upcycled .pt checkpoint.
        target_layers:          target layer count (default: 2 * source layers).

    Returns:
        Path to saved upcycled checkpoint.
    """
    src_path = Path(source_checkpoint_path)
    dst_path = Path(target_checkpoint_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        # Full Trainer checkpoint
        model_state = ckpt["model_state_dict"]
        cfg_dict = ckpt.get("model_config", {})
        if isinstance(cfg_dict, dict):
            source_config = LoopLMConfig(**cfg_dict)
        elif isinstance(cfg_dict, LoopLMConfig):
            source_config = cfg_dict
        else:
            raise ValueError(f"Unexpected model_config type in checkpoint: {type(cfg_dict)}")

        upcycled_state, target_config = upcycle_state_dict(
            model_state, source_config, target_layers=target_layers
        )

        import dataclasses
        new_ckpt: dict[str, Any] = {
            "step": ckpt.get("step", 0),
            "model_state_dict": upcycled_state,
            "model_config": dataclasses.asdict(target_config),
            "stage_idx": ckpt.get("stage_idx", 0),
            "stage_start_step": ckpt.get("stage_start_step", 0),
        }
        if "trainer_config" in ckpt:
            new_ckpt["trainer_config"] = ckpt["trainer_config"]
        # Note: optimizer_state_dict is intentionally omitted/reset because parameter
        # shapes and counts have changed (48 layers vs 24 layers).
        torch.save(new_ckpt, dst_path)
    else:
        # Raw state dict checkpoint
        raise ValueError(
            f"Checkpoint at {src_path} is missing 'model_state_dict'. Ensure it is a valid LoopLM checkpoint."
        )

    return dst_path
