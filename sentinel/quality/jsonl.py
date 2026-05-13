"""JSONL file validation."""

from __future__ import annotations

import json
from pathlib import Path


def validate_jsonl_file(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False, "valid": False, "line_count": 0, "error": "file not found"}

    lines = path.read_text(encoding="utf-8").splitlines()
    count = 0
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            return {
                "path": str(path),
                "exists": True,
                "valid": False,
                "line_count": i,
                "error": f"line {i}: {exc}",
            }
        count += 1

    return {"path": str(path), "exists": True, "valid": True, "line_count": count, "error": None}


def validate_jsonl_files(paths: list[Path]) -> list[dict]:
    return [validate_jsonl_file(p) for p in paths]
