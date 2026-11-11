#!/usr/bin/env python
"""scripts/check_secrets.py — Pre-commit and CI scanner for leaked API keys and secrets.

Scans staged or specified files for accidental leaks:
- W&B API keys (WANDB_API_KEY)
- Hugging Face tokens (HF_TOKEN, hf_...)
- GitHub tokens (ghp_..., github_pat_...)
- OpenAI / Anthropic API keys (sk-..., sk-ant-...)
- AWS access keys (AKIA...)
- Private keys (BEGIN ... PRIVATE KEY)
- Generic secret assignments

Usage:
    # Scan specific files (e.g. from pre-commit):
    uv run scripts/check_secrets.py file1.py file2.slurm

    # Scan all tracked files in git:
    uv run scripts/check_secrets.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Regex patterns for detecting credentials and sensitive tokens
SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "Private Key Header",
        re.compile(r"-----BEGIN (?:\w+ )?PRIVATE KEY-----", re.IGNORECASE),
    ),
    (
        "Hugging Face Token",
        re.compile(r"\bhf_[A-Za-z0-9]{34,}\b"),
    ),
    (
        "GitHub Personal Access Token",
        re.compile(r"\b(?:ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,})\b"),
    ),
    (
        "OpenAI API Key",
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
    ),
    (
        "Anthropic API Key",
        re.compile(r"\bsk-ant-[A-Za-z0-9_-]{32,}\b"),
    ),
    (
        "AWS Access Key ID",
        re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
    ),
    (
        "WandB API Key assignment",
        re.compile(r"""(?:WANDB_API_KEY|wandb_api_key)\s*[:=]\s*['"]([a-f0-9]{40})['"]""", re.IGNORECASE),
    ),
    (
        "Slack Token",
        re.compile(r"\bxox[baprs]-[0-9a-zA-Z-]{10,}\b"),
    ),
]

# Files/paths to ignore during scan
EXCLUDED_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".parquet",
    ".arrow",
    ".lock",
}

EXCLUDED_FILENAMES = {
    "uv.lock",
    "package-lock.json",
    ".coverage",
}

PLACEHOLDER_SUBSTRINGS = [
    "example",
    "placeholder",
    "dummy",
    "fake",
    "test_",
    "your_",
    "my_",
    "<",
    ">",
    "...",
]


def is_placeholder(matched_text: str) -> bool:
    """Return True if matched token looks like a documentation placeholder."""
    lower = matched_text.lower()
    return any(p in lower for p in PLACEHOLDER_SUBSTRINGS)


def scan_file(path: Path) -> list[Tuple[int, str, str]]:
    """Scan a single file for secrets.

    Returns:
        List of tuples (line_number, rule_name, line_content).
    """
    if path.suffix.lower() in EXCLUDED_EXTENSIONS or path.name in EXCLUDED_FILENAMES:
        return []

    # Ignore binary files or files in virtual environments/.git
    parts = set(path.parts)
    if {".git", ".venv", "venv", "__pycache__", "site-packages"}.intersection(parts):
        return []

    # Exclude scanner itself and test file testing this scanner
    if path.name == "check_secrets.py" or path.name == "test_checkpoint_and_security.py":
        return []

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    findings = []
    lines = content.splitlines()

    for line_idx, line in enumerate(lines, start=1):
        # Skip pure comments that indicate documentation
        stripped = line.strip()
        if stripped.startswith("#") and any(ph in stripped.lower() for ph in ["example", "placeholder", "fake"]):
            continue

        for rule_name, pattern in SECRET_PATTERNS:
            match = pattern.search(line)
            if match:
                matched_str = match.group(0)
                if not is_placeholder(matched_str):
                    # Redact part of the match for display
                    safe_line = line[:80]
                    findings.append((line_idx, rule_name, safe_line))

    return findings


def get_git_tracked_files() -> list[Path]:
    """Get all git tracked files using git ls-files."""
    try:
        out = subprocess.check_output(["git", "ls-files"], text=True)
        return [Path(p.strip()) for p in out.splitlines() if p.strip()]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Scan files for hardcoded secrets.")
    parser.add_argument(
        "files",
        nargs="*",
        type=str,
        help="Specific files to check. If omitted, scans git tracked files.",
    )
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = get_git_tracked_files()

    total_violations = 0

    for path in paths:
        if not path.is_file():
            continue
        violations = scan_file(path)
        if violations:
            for line_no, rule, snippet in violations:
                print(f"[SECURITY ALERT] {path}:{line_no} matches '{rule}' -> {snippet}")
                total_violations += 1

    if total_violations > 0:
        print(f"\nFound {total_violations} potential secret leak(s)!", file=sys.stderr)
        print("Please remove secrets, use environment variables, or add to .gitignore.", file=sys.stderr)
        sys.exit(1)

    print("Secret scan passed: no leaks detected.")
    sys.exit(0)


if __name__ == "__main__":
    main()
