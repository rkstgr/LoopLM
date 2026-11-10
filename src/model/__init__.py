"""src/model — Model architectures, configs, upcycling, and HF converters for LoopLM."""

from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM, LoopLMOutput
from src.model.upcycle import upcycle_model, upcycle_state_dict, upcycle_checkpoint
from src.model.hf_converter import (
    hf_to_looplm_state_dict,
    looplm_to_hf_state_dict,
    hf_config_to_looplm_config,
    looplm_config_to_hf_config,
    load_from_hf,
    save_to_hf,
)

__all__ = [
    "LoopLMConfig",
    "LoopLM",
    "LoopLMOutput",
    "upcycle_model",
    "upcycle_state_dict",
    "upcycle_checkpoint",
    "hf_to_looplm_state_dict",
    "looplm_to_hf_state_dict",
    "hf_config_to_looplm_config",
    "looplm_config_to_hf_config",
    "load_from_hf",
    "save_to_hf",
]
