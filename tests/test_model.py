import pytest
from src.model.config import LoopLMConfig


def test_small_config_fields():
    cfg = LoopLMConfig.small()
    assert cfg.vocab_size == 49152
    assert cfg.hidden_size == 768
    assert cfg.num_layers == 6
    assert cfg.num_heads == 12
    assert cfg.intermediate_size == 2048
    assert cfg.max_recurrent_steps == 4
    assert cfg.rope_base == 10000.0
    assert cfg.beta_kl == 0.1


def test_small_config_param_count():
    cfg = LoopLMConfig.small()
    n = cfg.num_parameters()
    # Expect roughly 85-115M params
    assert 80_000_000 < n < 120_000_000, f"Unexpected param count: {n:,}"


def test_ouro_1_4b_param_count():
    cfg = LoopLMConfig.ouro_1_4b()
    n = cfg.num_parameters()
    # Expect roughly 1.2-1.6B params
    assert 1_200_000_000 < n < 1_600_000_000, f"Unexpected param count: {n:,}"


def test_ouro_2_6b_param_count():
    cfg = LoopLMConfig.ouro_2_6b()
    n = cfg.num_parameters()
    # Expect roughly 2.4-2.8B params
    assert 2_400_000_000 < n < 2_800_000_000, f"Unexpected param count: {n:,}"


def test_2_6b_shares_hidden_dim_with_1_4b():
    cfg_1_4b = LoopLMConfig.ouro_1_4b()
    cfg_2_6b = LoopLMConfig.ouro_2_6b()
    assert cfg_1_4b.hidden_size == cfg_2_6b.hidden_size
    assert cfg_2_6b.num_layers == 2 * cfg_1_4b.num_layers


def test_custom_config():
    cfg = LoopLMConfig(hidden_size=512, num_layers=4, num_heads=8, intermediate_size=1024)
    assert cfg.num_parameters() > 0


def test_num_heads_divides_hidden_size():
    cfg = LoopLMConfig.small()
    assert cfg.hidden_size % cfg.num_heads == 0
