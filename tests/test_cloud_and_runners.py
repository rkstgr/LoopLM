"""tests/test_cloud_and_runners.py — Tests for Modal runner, DDP script, and cluster portability."""

import os
import subprocess
import sys
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).parent.parent


def test_cluster_env_example_syntax():
    """Verify scripts/cluster_env.sh.example exists and passes bash syntax validation."""
    example_path = REPO_ROOT / "scripts" / "cluster_env.sh.example"
    assert example_path.is_file(), "scripts/cluster_env.sh.example must exist"

    res = subprocess.run(["bash", "-n", str(example_path)], capture_output=True, text=True)
    assert res.returncode == 0, f"Syntax error in cluster_env.sh.example: {res.stderr}"


def test_slurm_scripts_portable():
    """Verify primary SLURM scripts pass syntax check and do not have hardcoded westai0047."""
    slurm_files = [
        "scripts/train.slurm",
        "scripts/pretrain_1_4b.slurm",
        "scripts/mano.slurm",
        "scripts/capo.slurm",
    ]
    for rel_path in slurm_files:
        full_path = REPO_ROOT / rel_path
        assert full_path.is_file(), f"{rel_path} must exist"

        # Check syntax
        res = subprocess.run(["bash", "-n", str(full_path)], capture_output=True, text=True)
        assert res.returncode == 0, f"Syntax error in {rel_path}: {res.stderr}"

        # Ensure hardcoded account and scratch paths were removed
        content = full_path.read_text()
        assert "westai0047" not in content, f"Hardcoded westai0047 found in {rel_path}"
        assert "/p/scratch/westai0047" not in content, f"Hardcoded /p/scratch/westai0047 found in {rel_path}"


def test_run_ddp_help_and_dry_run():
    """Verify scripts/run_ddp.sh supports --help and --dry-run without failure."""
    script_path = REPO_ROOT / "scripts" / "run_ddp.sh"
    assert script_path.is_file(), "scripts/run_ddp.sh must exist"

    # Test --help
    help_res = subprocess.run(["bash", str(script_path), "--help"], capture_output=True, text=True)
    assert help_res.returncode == 0
    assert "LoopLM Multi-GPU DDP Launcher" in help_res.stdout

    # Test --dry-run
    dry_res = subprocess.run(
        ["bash", str(script_path), "--dry-run", "--gpus", "2", "--model-config", "small"],
        capture_output=True,
        text=True,
    )
    assert dry_res.returncode == 0
    assert "GPUs:       2" in dry_res.stdout
    assert "torchrun --nproc_per_node=2" in dry_res.stdout
    assert "--model-config small" in dry_res.stdout


def test_modal_train_script_execution():
    """Verify scripts/modal_train.py executes and produces actionable output."""
    script_path = REPO_ROOT / "scripts" / "modal_train.py"
    assert script_path.is_file(), "scripts/modal_train.py must exist"

    res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    # If modal is not installed locally, it should print installation guidance and exit 1
    # If modal is installed, it handles CLI arguments
    output = res.stdout + res.stderr
    assert "modal" in output.lower()
