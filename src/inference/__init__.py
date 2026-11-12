"""src/inference — LoopLM inference utilities: early exit, KV-cache sharing, and evaluation."""

from src.inference.kv_cache import (
    KVCache,
    LayerKVCache,
    StepCache,
    KVCacheStrategy,
    GenerateResult,
    prefill,
    extract_decode_cache,
    decode_one_token,
    decode_one_token_full,
    decode_one_token_last_step,
    decode_one_token_first_step,
    decode_one_token_averaged,
    generate,
)
from src.inference.early_exit import (
    EarlyExitStrategy,
    EarlyExitResult,
    StaticExit,
    HiddenStateDiffExit,
    QExit,
    run_with_early_exit,
    compute_token_exit_distribution,
)
from src.inference.lm_eval_wrapper import LoopLMLM

__all__ = [
    "KVCache",
    "LayerKVCache",
    "StepCache",
    "KVCacheStrategy",
    "GenerateResult",
    "prefill",
    "extract_decode_cache",
    "decode_one_token",
    "decode_one_token_full",
    "decode_one_token_last_step",
    "decode_one_token_first_step",
    "decode_one_token_averaged",
    "generate",
    "EarlyExitStrategy",
    "EarlyExitResult",
    "StaticExit",
    "HiddenStateDiffExit",
    "QExit",
    "run_with_early_exit",
    "compute_token_exit_distribution",
    "LoopLMLM",
]
