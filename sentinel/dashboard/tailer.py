"""JSONL file tailer — reads new lines incrementally using byte offset tracking."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger


class JsonlTailer:
    """Tails a JSONL file, returning only lines written since the last poll."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset: int = 0
        self._line_count: int = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def line_count(self) -> int:
        return self._line_count

    @property
    def offset(self) -> int:
        return self._offset

    def poll(self) -> list[dict]:
        """Read and parse new lines since last poll. Returns parsed dicts."""
        if not self._path.exists():
            return []

        items: list[dict] = []
        try:
            with open(self._path, "rb") as f:
                f.seek(self._offset)
                raw = f.read()
                if not raw:
                    return []
                text = raw.decode("utf-8", errors="replace")
                # Handle partial last line — only process complete lines
                if not text.endswith("\n"):
                    last_newline = text.rfind("\n")
                    if last_newline == -1:
                        # No complete lines yet — don't advance offset
                        return []
                    # Keep only complete lines, rewind offset for partial
                    complete = text[:last_newline + 1]
                    self._offset += len(complete.encode("utf-8"))
                    text = complete
                else:
                    self._offset += len(raw)

                for line in text.strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                        self._line_count += 1
                    except json.JSONDecodeError:
                        logger.warning("Malformed JSONL line in {}", self._path)
        except OSError:
            return []

        return items

    def reset(self) -> None:
        """Reset to beginning of file."""
        self._offset = 0
        self._line_count = 0
