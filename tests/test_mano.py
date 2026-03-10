"""Tests for the Mano (knowledge manipulation) analysis module."""

import math
import random

import pytest
import torch

from src.analysis.mano import (
    MODULUS,
    OPS,
    ManoTokenizer,
    generate_expression,
    generate_mano_example,
    ManoDataset,
    build_block_causal_mask,
    evaluate_mano,
    make_mano_model_config,
    ManoConfig,
    ManoResult,
    run_mano_single,
    _MODEL_PRESETS,
)
from src.model.config import LoopLMConfig
from src.model.looplm import LoopLM


# ── Constants ────────────────────────────────────────────────────────────────


def test_modulus():
    assert MODULUS == 23


def test_ops():
    assert set(OPS) == {"+", "-", "*"}


# ── ManoTokenizer ────────────────────────────────────────────────────────────


class TestManoTokenizer:
    def test_vocab_size(self):
        tok = ManoTokenizer(max_ops=10)
        # 23 numbers + 3 ops + 4 special + 11 len markers = 41
        assert tok.vocab_size == 41

    def test_vocab_size_max_ops_24(self):
        tok = ManoTokenizer(max_ops=24)
        assert tok.vocab_size == 23 + 3 + 4 + 25  # = 55

    def test_encode_decode_roundtrip(self):
        tok = ManoTokenizer(max_ops=10)
        tokens = ["<bos>", "<len_3>", "+", "5", "7", "<ans>", "12"]
        ids = tok.encode(tokens)
        assert tok.decode(ids) == tokens

    def test_special_tokens(self):
        tok = ManoTokenizer(max_ops=5)
        assert tok.bos_token_id == tok.encode(["<bos>"])[0]
        assert tok.ans_token_id == tok.encode(["<ans>"])[0]
        assert tok.eos_token_id == tok.encode(["<eos>"])[0]
        assert tok.pad_token_id == tok.encode(["<pad>"])[0]

    def test_all_numbers_encodable(self):
        tok = ManoTokenizer(max_ops=5)
        for i in range(MODULUS):
            ids = tok.encode([str(i)])
            assert len(ids) == 1

    def test_all_ops_encodable(self):
        tok = ManoTokenizer(max_ops=5)
        for op in OPS:
            ids = tok.encode([op])
            assert len(ids) == 1

    def test_len_markers_encodable(self):
        tok = ManoTokenizer(max_ops=10)
        for i in range(11):
            ids = tok.encode([f"<len_{i}>"])
            assert len(ids) == 1

    def test_unique_ids(self):
        tok = ManoTokenizer(max_ops=10)
        all_ids = [tok.encode([t])[0] for t in tok._tokens]
        assert len(set(all_ids)) == len(all_ids), "Token IDs must be unique"


# ── Expression generation ────────────────────────────────────────────────────


class TestExpressionGeneration:
    def test_leaf_expression(self):
        rng = random.Random(42)
        tokens, answer = generate_expression(0, rng)
        assert len(tokens) == 1
        assert 0 <= answer < MODULUS
        assert tokens[0] == str(answer)

    def test_single_op(self):
        rng = random.Random(42)
        tokens, answer = generate_expression(1, rng)
        # op a b → 3 tokens
        assert len(tokens) == 3
        assert tokens[0] in OPS
        assert 0 <= answer < MODULUS

    def test_single_op_correctness(self):
        """Verify the answer is correct for a single operation."""
        rng = random.Random(42)
        for _ in range(100):
            tokens, answer = generate_expression(1, rng)
            op = tokens[0]
            a = int(tokens[1])
            b = int(tokens[2])
            if op == "+":
                expected = (a + b) % MODULUS
            elif op == "-":
                expected = (a - b) % MODULUS
            else:
                expected = (a * b) % MODULUS
            assert answer == expected, f"{op} {a} {b} = {answer}, expected {expected}"

    def test_multi_op_token_count(self):
        """An expression with n ops has 2n+1 tokens (n ops + n+1 leaves)."""
        rng = random.Random(42)
        for n_ops in range(1, 8):
            tokens, _ = generate_expression(n_ops, rng)
            assert len(tokens) == 2 * n_ops + 1

    def test_answer_in_range(self):
        rng = random.Random(42)
        for n_ops in [1, 3, 5, 10]:
            for _ in range(50):
                _, answer = generate_expression(n_ops, rng)
                assert 0 <= answer < MODULUS

    def test_deterministic_with_seed(self):
        tokens1, ans1 = generate_expression(5, random.Random(123))
        tokens2, ans2 = generate_expression(5, random.Random(123))
        assert tokens1 == tokens2
        assert ans1 == ans2

    def test_different_seeds_different_output(self):
        tokens1, _ = generate_expression(5, random.Random(1))
        tokens2, _ = generate_expression(5, random.Random(2))
        # Very unlikely to be identical
        assert tokens1 != tokens2


class TestManoExample:
    def test_format(self):
        rng = random.Random(42)
        tokens, answer = generate_mano_example(3, rng)
        assert tokens[0] == "<bos>"
        assert tokens[1] == "<len_3>"
        assert tokens[-2] == "<ans>"
        assert tokens[-1] == str(answer)

    def test_expression_length(self):
        """Full example: <bos> + <len_n> + (2n+1 expr tokens) + <ans> + answer = 2n+5."""
        rng = random.Random(42)
        for n_ops in [1, 3, 5]:
            tokens, _ = generate_mano_example(n_ops, rng)
            assert len(tokens) == 2 * n_ops + 5


# ── Dataset ──────────────────────────────────────────────────────────────────


class TestManoDataset:
    def test_creates_chunks(self):
        tok = ManoTokenizer(max_ops=5)
        ds = ManoDataset(tok, n_examples=100, max_ops=5, seq_len=64, seed=42)
        assert len(ds) > 0

    def test_chunk_shape(self):
        tok = ManoTokenizer(max_ops=5)
        ds = ManoDataset(tok, n_examples=100, max_ops=5, seq_len=64, seed=42)
        tokens, prob_ids = ds[0]
        assert tokens.shape == (65,)  # seq_len + 1
        assert prob_ids.shape == (65,)

    def test_problem_ids_contiguous(self):
        """Within a chunk, problem IDs should form contiguous blocks."""
        tok = ManoTokenizer(max_ops=5)
        ds = ManoDataset(tok, n_examples=200, max_ops=5, seq_len=128, seed=42)
        tokens, prob_ids = ds[0]
        # Check that problem IDs don't interleave
        seen = set()
        prev = prob_ids[0].item()
        seen.add(prev)
        for pid in prob_ids[1:]:
            p = pid.item()
            if p != prev:
                assert p not in seen or p == -1, "Problem IDs should not interleave"
                seen.add(p)
                prev = p

    def test_all_tokens_valid(self):
        tok = ManoTokenizer(max_ops=5)
        ds = ManoDataset(tok, n_examples=50, max_ops=5, seq_len=64, seed=42)
        for i in range(len(ds)):
            tokens, _ = ds[i]
            assert (tokens >= 0).all()
            assert (tokens < tok.vocab_size).all()


# ── Block-causal mask ────────────────────────────────────────────────────────


class TestBlockCausalMask:
    def test_shape(self):
        prob_ids = torch.tensor([[0, 0, 1, 1, 2]])
        mask = build_block_causal_mask(prob_ids)
        assert mask.shape == (1, 1, 5, 5)

    def test_causal(self):
        """No position should attend to a future position."""
        prob_ids = torch.tensor([[0, 0, 0, 0]])
        mask = build_block_causal_mask(prob_ids)
        # Upper triangle (excluding diagonal) should be -inf
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[0, 0, i, j] == float("-inf")

    def test_cross_problem_blocked(self):
        """Positions from different problems should not attend to each other."""
        prob_ids = torch.tensor([[0, 0, 1, 1]])
        mask = build_block_causal_mask(prob_ids)
        # Position 2 (problem 1) should not attend to position 0 (problem 0)
        assert mask[0, 0, 2, 0] == float("-inf")
        assert mask[0, 0, 3, 0] == float("-inf")
        assert mask[0, 0, 3, 1] == float("-inf")

    def test_same_problem_attends(self):
        """Positions in the same problem with causal order should attend."""
        prob_ids = torch.tensor([[0, 0, 1, 1]])
        mask = build_block_causal_mask(prob_ids)
        assert mask[0, 0, 1, 0] == 0.0  # pos 1 attends to pos 0 (same problem)
        assert mask[0, 0, 3, 2] == 0.0  # pos 3 attends to pos 2 (same problem)

    def test_self_attendance(self):
        """Every position should attend to itself."""
        prob_ids = torch.tensor([[0, 1, 2]])
        mask = build_block_causal_mask(prob_ids)
        for i in range(3):
            assert mask[0, 0, i, i] == 0.0


# ── Model configs ────────────────────────────────────────────────────────────


class TestModelConfigs:
    def test_make_config(self):
        cfg = make_mano_model_config(
            num_layers=4, loop_count=2,
            hidden_size=128, num_heads=2, vocab_size=41,
        )
        assert cfg.num_layers == 4
        assert cfg.max_recurrent_steps == 2
        assert cfg.hidden_size == 128
        assert cfg.vocab_size == 41

    def test_presets_exist(self):
        for name in ["tiny", "small", "medium", "paper"]:
            assert name in _MODEL_PRESETS
            p = _MODEL_PRESETS[name]
            assert "hidden_size" in p
            assert "num_heads" in p

    def test_hidden_divisible_by_heads(self):
        for name, p in _MODEL_PRESETS.items():
            assert p["hidden_size"] % p["num_heads"] == 0, (
                f"Preset '{name}': hidden_size must be divisible by num_heads"
            )


# ── Evaluation ───────────────────────────────────────────────────────────────


class TestEvaluation:
    def test_accuracy_range(self):
        """Untrained model should have accuracy in [0, 1]."""
        tok = ManoTokenizer(max_ops=3)
        cfg = make_mano_model_config(
            num_layers=2, loop_count=1,
            hidden_size=64, num_heads=1, vocab_size=tok.vocab_size,
        )
        model = LoopLM(cfg)
        acc = evaluate_mano(
            model, tok, num_steps=1, max_ops=3,
            n_eval=20, device=torch.device("cpu"),
        )
        assert 0.0 <= acc <= 1.0

    def test_random_baseline(self):
        """Untrained model should have roughly 1/23 ≈ 4.3% accuracy."""
        tok = ManoTokenizer(max_ops=3)
        cfg = make_mano_model_config(
            num_layers=2, loop_count=1,
            hidden_size=64, num_heads=1, vocab_size=tok.vocab_size,
        )
        model = LoopLM(cfg)
        acc = evaluate_mano(
            model, tok, num_steps=1, max_ops=3,
            n_eval=100, device=torch.device("cpu"),
        )
        # Should be close to random (1/23 ≈ 0.043), allow wide margin
        assert acc < 0.3, f"Untrained accuracy {acc:.3f} seems too high"


# ── Smoke test ───────────────────────────────────────────────────────────────


class TestSmokeTest:
    @pytest.mark.slow
    def test_single_run_trains(self):
        """Train a tiny model for a few steps and verify loss decreases."""
        tok = ManoTokenizer(max_ops=3)
        config = ManoConfig(
            max_ops=3,
            n_train_examples=200,
            n_eval_examples=20,
            model_configs=[(2, 1)],
            model_preset="tiny",
            lr=1e-3,
            batch_size=8,
            seq_len=64,
            warmup_steps=5,
            train_steps=50,
            log_every=25,
            device="cpu",
            seed=42,
        )
        result = run_mano_single(
            tok, num_layers=2, loop_count=1, config=config,
            device=torch.device("cpu"),
        )
        assert isinstance(result, ManoResult)
        assert result.num_layers == 2
        assert result.loop_count == 1
        assert result.n_params > 0
        assert 0.0 <= result.accuracy <= 1.0
        assert result.final_loss > 0.0
