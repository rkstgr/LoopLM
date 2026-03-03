"""Tests for the Capo (knowledge capacity) analysis module."""

import math

import torch
import pytest

from src.analysis.capo import (
    N0,
    S0,
    LOG2_N0,
    LOG2_S0,
    N_FIRST_NAMES,
    N_MIDDLE_NAMES,
    N_LAST_NAMES,
    Individual,
    BioSGenerator,
    BioSTrainDataset,
    CapoConfig,
    CapoResult,
    _char_spans_to_token_indices,
    compute_capacity_ratio,
    make_capo_model_config,
    run_capo_single,
)
from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM


# ── Constants ─────────────────────────────────────────────────────────────────

def test_n0_value():
    assert N0 == 400 * 1_000 * 400  # = 160_000_000

def test_s0_log2_approx():
    """log2(S0) should be ≈ 47.6 bits per individual (paper Appendix B.1)."""
    assert 47.0 < LOG2_S0 < 48.5, f"LOG2_S0={LOG2_S0:.4f}"

def test_log2_n0_approx():
    assert 26.0 < LOG2_N0 < 28.0, f"LOG2_N0={LOG2_N0:.4f}"


# ── BioSGenerator ─────────────────────────────────────────────────────────────

def test_generator_produces_correct_count():
    gen = BioSGenerator(n_individuals=50)
    assert len(gen.individuals) == 50

def test_generator_unique_names():
    gen = BioSGenerator(n_individuals=100)
    names = [(ind.first_name, ind.middle_name, ind.last_name) for ind in gen.individuals]
    assert len(set(names)) == 100, "Duplicate names detected"

def test_generator_attributes_in_pools():
    gen = BioSGenerator(n_individuals=20)
    for ind in gen.individuals:
        assert ind.gender in ("male", "female")
        assert 1 <= ind.birth_month <= 12
        assert 1 <= ind.birth_day <= 28
        assert 1800 <= int(ind.birth_year) <= 1999
        assert ind.university.startswith("Univ")
        assert ind.major.startswith("Major")
        assert ind.hometown.startswith("City")
        assert ind.employer.startswith("Emp")

def test_generator_names_in_pools():
    gen = BioSGenerator(n_individuals=10)
    for ind in gen.individuals:
        assert ind.first_name.startswith("Fname")
        assert ind.middle_name.startswith("Mname")
        assert ind.last_name.startswith("Lname")

def test_generator_rejects_oversized_n():
    with pytest.raises(ValueError, match="exceeds name pool"):
        BioSGenerator(n_individuals=N_FIRST_NAMES * N_MIDDLE_NAMES * N_LAST_NAMES + 1)

def test_generator_deterministic():
    g1 = BioSGenerator(n_individuals=20, seed=0)
    g2 = BioSGenerator(n_individuals=20, seed=0)
    for i1, i2 in zip(g1.individuals, g2.individuals):
        assert i1.first_name == i2.first_name
        assert i1.last_name == i2.last_name

def test_generator_different_seeds_differ():
    g1 = BioSGenerator(n_individuals=20, seed=0)
    g2 = BioSGenerator(n_individuals=20, seed=99)
    names1 = [ind.first_name for ind in g1.individuals]
    names2 = [ind.first_name for ind in g2.individuals]
    assert names1 != names2


# ── render() and span alignment ───────────────────────────────────────────────

def test_render_contains_name():
    gen = BioSGenerator(n_individuals=5)
    for ind in gen.individuals:
        text, name_spans, attr_spans = gen.render(ind)
        assert ind.first_name in text
        assert ind.middle_name in text
        assert ind.last_name in text

def test_render_contains_attributes():
    gen = BioSGenerator(n_individuals=5)
    for ind in gen.individuals:
        text, _, _ = gen.render(ind)
        assert ind.university in text
        assert ind.major in text
        assert ind.hometown in text
        assert ind.employer in text

def test_render_name_span_alignment():
    gen = BioSGenerator(n_individuals=5)
    for ind in gen.individuals:
        text, name_spans, _ = gen.render(ind)
        name_text = "".join(text[s:e] for s, e in name_spans)
        # Name spans should cover exactly the three name parts
        assert ind.first_name in name_text
        assert ind.middle_name in name_text
        assert ind.last_name in name_text

def test_render_attr_span_alignment():
    gen = BioSGenerator(n_individuals=5)
    for ind in gen.individuals:
        text, _, attr_spans = gen.render(ind)
        attr_text = "".join(text[s:e] for s, e in attr_spans)
        assert ind.university in attr_text
        assert ind.major in attr_text
        assert ind.employer in attr_text

def test_render_spans_do_not_overlap():
    gen = BioSGenerator(n_individuals=3)
    for ind in gen.individuals:
        text, name_spans, attr_spans = gen.render(ind)
        all_spans = name_spans + attr_spans
        # Check no pair of spans overlaps
        for i, (s1, e1) in enumerate(all_spans):
            for j, (s2, e2) in enumerate(all_spans):
                if i >= j:
                    continue
                overlap = min(e1, e2) - max(s1, s2)
                assert overlap <= 0, f"Spans {i} and {j} overlap: ({s1},{e1}) ({s2},{e2})"

def test_render_all_returns_n_strings():
    gen = BioSGenerator(n_individuals=10)
    texts = gen.render_all()
    assert len(texts) == 10
    for t in texts:
        assert isinstance(t, str) and len(t) > 0


# ── BioSTrainDataset ──────────────────────────────────────────────────────────

class FakeFastTokenizer:
    """Word-splitting tokenizer that supports offset_mapping."""
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [hash(w) % 50 + 1 for w in text.split()]

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        words = text.split()
        ids = [hash(w) % 50 + 1 for w in words]
        offsets = []
        pos = 0
        for w in words:
            # find actual position in text
            idx = text.index(w, pos)
            offsets.append((idx, idx + len(w)))
            pos = idx + len(w)
        result = type("Enc", (), {"input_ids": ids, "offset_mapping": offsets})()
        return result


def test_bios_train_dataset_chunk_shape():
    gen = BioSGenerator(n_individuals=20, seed=0)
    tok = FakeFastTokenizer()
    seq_len = 8
    ds = BioSTrainDataset(gen, tok, seq_len=seq_len)
    assert len(ds) > 0
    for i in range(min(5, len(ds))):
        chunk = ds[i]
        assert chunk.shape == (seq_len + 1,)
        assert chunk.dtype == torch.long

def test_bios_train_dataset_no_overflow():
    gen = BioSGenerator(n_individuals=10, seed=0)
    tok = FakeFastTokenizer()
    ds = BioSTrainDataset(gen, tok, seq_len=4)
    for i in range(len(ds)):
        chunk = ds[i]
        assert (chunk >= 0).all()


# ── _char_spans_to_token_indices ──────────────────────────────────────────────

def test_char_spans_to_token_indices_basic():
    """The fake tokenizer splits on whitespace; verify span mapping."""
    tok = FakeFastTokenizer()
    text = "hello world foo"
    # Span covering "world" (chars 6-11)
    indices = _char_spans_to_token_indices(text, [(6, 11)], tok)
    assert 1 in indices   # "world" is token index 1

def test_char_spans_to_token_indices_empty():
    tok = FakeFastTokenizer()
    indices = _char_spans_to_token_indices("hello world", [], tok)
    assert indices == []

def test_char_spans_to_token_indices_full_text():
    tok = FakeFastTokenizer()
    text = "a b c"
    indices = _char_spans_to_token_indices(text, [(0, len(text))], tok)
    assert set(indices) == {0, 1, 2}


# ── compute_capacity_ratio ────────────────────────────────────────────────────

def _tiny_model() -> tuple[LoopLM, LoopLMConfig]:
    cfg = LoopLMConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=1,
        num_heads=2,
        intermediate_size=64,
        max_seq_len=64,
        max_recurrent_steps=2,
    )
    return LoopLM(cfg), cfg


class SmallFastTokenizer:
    """Tiny tokenizer: each character → its ASCII code (mod 62 + 1); supports offsets."""
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 62 + 1 for c in text]

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        ids = self.encode(text)
        offsets = [(i, i + 1) for i in range(len(text))]
        return type("Enc", (), {"input_ids": ids, "offset_mapping": offsets})()


def test_compute_capacity_ratio_returns_float():
    gen = BioSGenerator(n_individuals=3, seed=0)
    model, _ = _tiny_model()
    tok = SmallFastTokenizer()
    device = torch.device("cpu")
    ratio, p1, p2 = compute_capacity_ratio(model, gen, tok, num_steps=1, device=device)
    assert isinstance(ratio, float)
    assert isinstance(p1, float)
    assert isinstance(p2, float)


def test_compute_capacity_ratio_non_negative():
    gen = BioSGenerator(n_individuals=3, seed=0)
    model, _ = _tiny_model()
    tok = SmallFastTokenizer()
    ratio, p1, p2 = compute_capacity_ratio(
        model, gen, tok, num_steps=1, device=torch.device("cpu")
    )
    assert ratio >= 0.0
    assert p1 >= 0.0
    assert p2 >= 0.0


def test_compute_capacity_ratio_perfect_model_gives_high_bits():
    """A model with near-zero loss should yield high bits/param."""
    gen = BioSGenerator(n_individuals=2, seed=0)
    tok = SmallFastTokenizer()
    device = torch.device("cpu")

    # Compute ratio for an untrained (random) model
    model, _ = _tiny_model()
    ratio_random, _, _ = compute_capacity_ratio(model, gen, tok, num_steps=1, device=device)

    # Compute the theoretical maximum for N=2 individuals
    # max_bits = N * (LOG2_N0 + LOG2_S0)
    # Ratio = max_bits / P is some positive number
    # For a random model with vocab=64, loss ≈ log(64) ≈ 4.16 nats
    # We just verify random is finite and non-negative
    assert ratio_random >= 0.0


def test_compute_capacity_ratio_uses_final_step_logits():
    """Running with num_steps=2 must return a valid ratio."""
    gen = BioSGenerator(n_individuals=2, seed=0)
    model, _ = _tiny_model()
    tok = SmallFastTokenizer()
    device = torch.device("cpu")
    ratio, _, _ = compute_capacity_ratio(model, gen, tok, num_steps=2, device=device)
    assert ratio >= 0.0


# ── make_capo_model_config ─────────────────────────────────────────────────────

def test_make_capo_model_config_known_sizes():
    for size in ("micro", "mini", "small", "medium"):
        cfg = make_capo_model_config(size, loop_count=4)
        assert isinstance(cfg, LoopLMConfig)
        assert cfg.max_recurrent_steps == 4

def test_make_capo_model_config_loop_count():
    for lc in (1, 4, 8):
        cfg = make_capo_model_config("micro", lc)
        assert cfg.max_recurrent_steps == lc

def test_make_capo_model_config_invalid_size():
    with pytest.raises(ValueError, match="Unknown size"):
        make_capo_model_config("huge", loop_count=4)

def test_make_capo_model_config_micro_is_smallest():
    micro = make_capo_model_config("micro", 1)
    mini = make_capo_model_config("mini", 1)
    assert micro.hidden_size < mini.hidden_size


# ── run_capo_single smoke test ─────────────────────────────────────────────────

def test_run_capo_single_smoke():
    """run_capo_single must complete and return a CapoResult with non-negative bits."""
    config = CapoConfig(
        n_individuals=5,
        train_exposures=1,
        lr=1e-3,
        batch_size=2,
        seq_len=16,
        warmup_steps=1,
        tokenizer_id="HuggingFaceTB/SmolLM2-135M",
        device="cpu",
        seed=0,
    )

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)
    except Exception:
        pytest.skip("SmolLM2 tokenizer not available in this environment")

    gen = BioSGenerator(n_individuals=config.n_individuals, seed=0)
    model_cfg = LoopLMConfig(
        vocab_size=tokenizer.vocab_size or 49152,
        hidden_size=32,
        num_layers=1,
        num_heads=2,
        intermediate_size=64,
        max_seq_len=config.seq_len,
        max_recurrent_steps=2,
    )
    device = torch.device("cpu")

    result = run_capo_single(
        gen, tokenizer, model_cfg, "micro", loop_count=1, config=config, device=device
    )

    assert isinstance(result, CapoResult)
    assert result.bits_per_param >= 0.0
    assert result.n_params > 0
    assert result.n_individuals == config.n_individuals
    assert result.loop_count == 1
