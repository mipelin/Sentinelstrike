"""YAML config loader."""

from __future__ import annotations

from pathlib import Path

import yaml

from sentinel.config.schema import AppConfig


def load_config(path: str | Path) -> AppConfig:
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return AppConfig(**data)
