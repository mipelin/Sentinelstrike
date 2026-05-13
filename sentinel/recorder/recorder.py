"""Mission recorder — persists events to JSONL."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

from sentinel.common.events import SystemEvent
from sentinel.common.time import utc_now_iso


class MissionRecorder:
    def __init__(self, base_dir: str | Path, mission_id: str) -> None:
        self._base_dir = Path(base_dir)
        self._mission_id = mission_id
        self._run_dir: Path | None = None
        self._events_file: IO[str] | None = None  # type: ignore[type-arg]
        self._closed = False

    def start_run(self) -> Path:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._run_dir = self._base_dir / f"{ts}_{self._mission_id}"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._events_file = (self._run_dir / "events.jsonl").open("a", encoding="utf-8")
        metadata = {"mission_id": self._mission_id, "started_at_utc": utc_now_iso()}
        (self._run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return self._run_dir

    def record_event(self, event: SystemEvent) -> None:
        if self._closed or self._events_file is None:
            return
        self._events_file.write(event.model_dump_json() + "\n")
        self._events_file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._events_file is not None:
            self._events_file.close()
