"""src/utils/checkpoint.py — Safe checkpoint loading utility for LoopLM.

Provides defense-in-depth against untrusted pickle deserialization vulnerabilities:
1. Tries `weights_only=True` first (safest, only tensors and primitive types).
2. If custom dataclasses are present (e.g. LoopLMConfig, TrainerConfig), attempts loading
   with an explicit safe globals allowlist using PyTorch's `safe_globals` mechanism.
3. If unpickling still fails:
   - If `trust=True`: logs a warning and falls back to `weights_only=False`.
   - If `trust=False`: raises a `SecurityException` with actionable instructions.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class UnsafeCheckpointError(RuntimeError):
    """Raised when a checkpoint contains non-primitive types and trust=False."""
    pass


def _get_default_allowlist() -> list[type]:
    """Return safe dataclasses and types commonly saved in LoopLM checkpoints."""
    from src.model.config import LoopLMConfig
    from src.training.trainer import StageConfig, TrainerConfig, GateFinetuneConfig

    types = [
        LoopLMConfig,
        TrainerConfig,
        StageConfig,
        GateFinetuneConfig,
    ]
    return types


def load_checkpoint_safe(
    path: str | Path,
    map_location: str | torch.device | None = None,
    weights_only: bool = True,
    allowlist: list[type] | None = None,
    trust: bool = False,
) -> dict[str, Any]:
    """Safely load a PyTorch checkpoint with fallback defenses.

    Args:
        path:          Path to the .pt checkpoint file.
        map_location:  Device to load tensors onto (e.g. 'cpu', 'cuda').
        weights_only:  Whether to restrict unpickling to tensors/primitives (default: True).
        allowlist:     Extra safe types to permit during unpickling.
        trust:         Explicit opt-in to allow arbitrary pickle execution if safe loading fails.

    Returns:
        The deserialized checkpoint dictionary.

    Raises:
        FileNotFoundError: If the checkpoint path does not exist.
        UnsafeCheckpointError: If the checkpoint cannot be safely loaded and trust=False.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    # Check environment variable override (e.g. LOOPLM_TRUST_CHECKPOINTS=1)
    if not trust and os.getenv("LOOPLM_TRUST_CHECKPOINTS", "0") in ("1", "true", "True"):
        trust = True

    # 1. First attempt: standard weights_only=True
    if weights_only:
        try:
            return torch.load(ckpt_path, map_location=map_location, weights_only=True)
        except Exception as err:
            # 2. Second attempt: safe_globals with allowlisted types
            effective_allowlist = _get_default_allowlist()
            if allowlist:
                effective_allowlist.extend(allowlist)

            if hasattr(torch.serialization, "safe_globals"):
                try:
                    with torch.serialization.safe_globals(effective_allowlist):
                        return torch.load(ckpt_path, map_location=map_location, weights_only=True)
                except Exception:
                    pass  # Fall through to trust check

            # 3. Third attempt: trust opt-in fallback
            if trust:
                warnings.warn(
                    f"Loading checkpoint '{ckpt_path}' with weights_only=False because safe loading "
                    "failed. Only load checkpoints from sources you trust.",
                    UserWarning,
                    stacklevel=2,
                )
                return torch.load(ckpt_path, map_location=map_location, weights_only=False)

            raise UnsafeCheckpointError(
                f"Failed to safely load checkpoint '{ckpt_path}' using weights_only=True.\n"
                f"Underlying error: {err}\n"
                "To load this checkpoint if you trust its origin, pass `trust=True` to "
                "`load_checkpoint_safe(...)` or set `export LOOPLM_TRUST_CHECKPOINTS=1`."
            ) from err

    # If caller explicitly asked for weights_only=False
    if not trust:
        warnings.warn(
            f"weights_only=False requested for '{ckpt_path}'. "
            "Consider using weights_only=True with allowlist for defense-in-depth.",
            UserWarning,
            stacklevel=2,
        )
    return torch.load(ckpt_path, map_location=map_location, weights_only=False)
