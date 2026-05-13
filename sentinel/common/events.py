"""System-wide event model and factory."""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from sentinel.common.time import utc_now_iso


class SystemEvent(BaseModel):
    event_id: str
    mission_id: str
    timestamp_utc: str
    source: str
    event_type: str
    severity: str = "info"
    payload: dict = {}


def make_event(
    mission_id: str,
    source: str,
    event_type: str,
    payload: dict | None = None,
    severity: str = "info",
) -> SystemEvent:
    return SystemEvent(
        event_id=uuid.uuid4().hex[:12],
        mission_id=mission_id,
        timestamp_utc=utc_now_iso(),
        source=source,
        event_type=event_type,
        severity=severity,
        payload=payload or {},
    )
