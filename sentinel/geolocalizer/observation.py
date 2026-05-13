"""Observation helpers."""

from __future__ import annotations

import hashlib

from sentinel.common.types import GeoObservation


def make_observation_id(track_id: str, timestamp_utc: str) -> str:
    h = hashlib.sha256(f"{track_id}:{timestamp_utc}".encode()).hexdigest()[:12]
    return f"obs_{h}"


def observation_to_json_dict(obs: GeoObservation) -> dict:
    return obs.model_dump()
