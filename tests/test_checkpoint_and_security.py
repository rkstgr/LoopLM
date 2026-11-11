"""tests/test_checkpoint_and_security.py — Tests for safe checkpoint loading, artifact verification, and secret scanning."""

import os
import tempfile
from pathlib import Path
import pytest
import torch

from src.model.config import LoopLMConfig
from src.utils.checkpoint import load_checkpoint_safe, UnsafeCheckpointError
from scripts.verify_artifacts import compute_sha256, verify_file, verify_cached_artifact
from scripts.check_secrets import scan_file


class DummyCustomClass:
    """Class not in the default allowlist to simulate untrusted pickle payloads."""
    def __init__(self, value: int):
        self.value = value


def test_load_checkpoint_safe_tensors_only():
    """Verify that pure tensor state dicts load cleanly under weights_only=True."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        data = {"weight": torch.randn(4, 4), "step": 100}
        torch.save(data, tmp_path)

        loaded = load_checkpoint_safe(tmp_path, weights_only=True)
        assert loaded["step"] == 100
        assert torch.allclose(loaded["weight"], data["weight"])
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_load_checkpoint_safe_allowlisted_dataclass():
    """Verify that checkpoints with LoopLMConfig load safely using safe_globals."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        cfg = LoopLMConfig.small()
        data = {"model_config": cfg, "step": 50}
        torch.save(data, tmp_path)

        loaded = load_checkpoint_safe(tmp_path, weights_only=True)
        assert loaded["step"] == 50
        assert isinstance(loaded["model_config"], LoopLMConfig)
        assert loaded["model_config"].hidden_size == cfg.hidden_size
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_load_checkpoint_safe_untrusted_blocked():
    """Verify that untrusted custom classes raise UnsafeCheckpointError when trust=False."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        data = {"payload": DummyCustomClass(42)}
        torch.save(data, tmp_path)

        # Ensure env var is not set
        os.environ.pop("LOOPLM_TRUST_CHECKPOINTS", None)

        with pytest.raises(UnsafeCheckpointError):
            load_checkpoint_safe(tmp_path, weights_only=True, trust=False)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_load_checkpoint_safe_trust_opt_in():
    """Verify that trust=True or LOOPLM_TRUST_CHECKPOINTS=1 allows fallback loading."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        data = {"payload": DummyCustomClass(42)}
        torch.save(data, tmp_path)

        # 1. Via explicit trust=True
        loaded = load_checkpoint_safe(tmp_path, weights_only=True, trust=True)
        assert loaded["payload"].value == 42

        # 2. Via environment variable
        os.environ["LOOPLM_TRUST_CHECKPOINTS"] = "1"
        try:
            loaded_env = load_checkpoint_safe(tmp_path, weights_only=True, trust=False)
            assert loaded_env["payload"].value == 42
        finally:
            os.environ.pop("LOOPLM_TRUST_CHECKPOINTS", None)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_artifact_checksum_verification():
    """Verify that compute_sha256 and verify_file detect matches and mismatches."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world\n")
        tmp_path = Path(f.name)

    try:
        h = compute_sha256(tmp_path)
        valid, _ = verify_file(tmp_path, h)
        assert valid is True

        # Incorrect hash
        bad_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        invalid, _ = verify_file(tmp_path, bad_hash)
        assert invalid is False
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_secret_scanner_detection():
    """Verify that check_secrets flags real-looking credentials and passes placeholders."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            "# Safe line\n"
            "HF_TOKEN = '<PLACEHOLDER>'\n"
            "WANDB_API_KEY = 'example_fake_key_placeholder'\n"
            "# Bad line below\n"
            "leaked_wandb = 'WANDB_API_KEY=\"1234567890abcdef1234567890abcdef12345678\"'\n"
        )
        tmp_path = Path(f.name)

    try:
        findings = scan_file(tmp_path)
        assert len(findings) == 1
        line_no, rule, _ = findings[0]
        assert rule == "WandB API Key assignment"
        assert line_no == 5
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
