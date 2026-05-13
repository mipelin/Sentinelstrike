"""SHA-256 checksums for run artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for chunk in open(path, "rb"):
        h.update(chunk)
    return h.hexdigest()


def compute_checksums(run_dir: Path) -> dict[str, str]:
    """Return {relative_filename: sha256_hex} for every JSON/JSONL/MD file in run_dir."""
    result: dict[str, str] = {}
    for p in sorted(run_dir.iterdir()):
        if p.is_file() and p.suffix in {".json", ".jsonl", ".md", ".mp4"}:
            result[p.name] = file_sha256(p)
    return result
