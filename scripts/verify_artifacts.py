#!/usr/bin/env python
"""scripts/verify_artifacts.py — Checksum verification for downloaded artifacts.

Pins SHA-256 hashes for externally-downloaded artifacts (tokenizers, configs)
to guard against upstream tampering or cache corruption ("tokenizer/dataset
got silently swapped" gap).

Usage:
    # Verify default artifacts in HF cache:
    uv run scripts/verify_artifacts.py

    # Verify artifacts in specific cache directory:
    uv run scripts/verify_artifacts.py --cache-dir /path/to/hf_cache

    # Verify a single file against expected hash:
    uv run scripts/verify_artifacts.py --file path/to/file --expected-sha256 <hash>
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Pinned SHA-256 digests for externally pulled artifacts
PINNED_ARTIFACTS: dict[str, dict[str, str]] = {
    "HuggingFaceTB/SmolLM2-135M": {
        "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
        "tokenizer_config.json": "4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424",
        "special_tokens_map.json": "e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3",
        "config.json": "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843",
    }
}


class ChecksumMismatchError(RuntimeError):
    """Raised when an artifact's SHA-256 does not match the pinned digest."""
    pass


def compute_sha256(filepath: str | Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def find_file_in_cache(cache_dir: Path, model_id: str, filename: str) -> Optional[Path]:
    """Locate a cached file either directly, in HF hub snapshot directory, or recursively."""
    # 1. Direct path: cache_dir / filename
    direct = cache_dir / filename
    if direct.is_file():
        return direct

    # 2. HF Hub layout: hub/models--<owner>--<repo>/snapshots/<rev>/<filename>
    hf_folder_name = "models--" + model_id.replace("/", "--")
    hub_dir = cache_dir / "hub" / hf_folder_name / "snapshots"
    if not hub_dir.exists():
        hub_dir = cache_dir / hf_folder_name / "snapshots"

    if hub_dir.exists():
        for snap in sorted(hub_dir.iterdir(), reverse=True):
            if snap.is_dir() and (snap / filename).is_file():
                return snap / filename

    # 3. Recursive glob fallback within cache_dir
    matches = list(cache_dir.rglob(filename))
    if matches:
        return matches[0]

    return None


def verify_file(filepath: str | Path, expected_sha256: str) -> Tuple[bool, str]:
    """Verify a file against an expected SHA-256 hex string."""
    actual_hash = compute_sha256(filepath)
    is_valid = actual_hash.lower() == expected_sha256.lower()
    return is_valid, actual_hash


def verify_cached_artifact(
    model_id: str,
    cache_dir: str | Path,
    strict: bool = True,
) -> bool:
    """Verify all pinned files for a given model ID within the cache directory."""
    cache_path = Path(cache_dir).expanduser()
    if not cache_path.exists():
        print(f"[FAIL] Cache directory does not exist: {cache_path}")
        return False

    if model_id not in PINNED_ARTIFACTS:
        print(f"[WARN] No pinned checksums found for model: {model_id}")
        return False

    expected_files = PINNED_ARTIFACTS[model_id]
    all_ok = True
    print(f"\n==> Verifying artifacts for {model_id} in {cache_path}")

    for filename, expected_hash in expected_files.items():
        found_path = find_file_in_cache(cache_path, model_id, filename)
        if not found_path:
            msg = f"[MISSING] {filename} not found in {cache_path}"
            print(msg)
            if strict:
                all_ok = False
            continue

        valid, actual_hash = verify_file(found_path, expected_hash)
        if valid:
            print(f"[OK] {filename} -> {actual_hash[:12]}... (matched)")
        else:
            print(
                f"[FAILED] {filename} checksum mismatch!\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}\n"
                f"  Location: {found_path}"
            )
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Verify artifact checksums.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HF cache directory (defaults to HF_HOME, HF_HUB_CACHE, or ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model ID to verify",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a single file to verify",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help="Expected SHA-256 hash for --file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail if any pinned file is missing",
    )
    args = parser.parse_args()

    # Single file check mode
    if args.file:
        if not args.expected_sha256:
            print("Error: --file requires --expected-sha256", file=sys.stderr)
            sys.exit(1)
        valid, actual = verify_file(args.file, args.expected_sha256)
        if valid:
            print(f"[OK] {args.file} matches expected SHA-256 ({actual[:12]}...)")
            sys.exit(0)
        else:
            print(f"[FAILED] Checksum mismatch for {args.file}:\n  Expected: {args.expected_sha256}\n  Actual:   {actual}", file=sys.stderr)
            sys.exit(1)

    # Cache directory verification mode
    cache_dir = args.cache_dir
    if not cache_dir:
        cache_dir = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface")

    success = verify_cached_artifact(args.model_id, cache_dir, strict=args.strict)
    if not success:
        sys.exit(1)
    print("\nAll checksums verified successfully.")


if __name__ == "__main__":
    main()
