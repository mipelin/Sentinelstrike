"""TelemetryCache — lock-protected latest-state cache for drone telemetry.

Stores the most recent telemetry snapshot with monotonic timestamps and
sequence IDs. Supports freshness validation, age checks, and atomic
snapshots. Subscription-driven updates overwrite previous values.

No queues, no FIFO, no history. Only latest.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from sentinel.common.types import GeoPoint


@dataclass
class TelemetryEntry:
    """Single telemetry field with monotonic timestamp."""

    value: object = None
    ts_monotonic: float = 0.0
    seq: int = 0


class TelemetryCache:
    """Lock-protected latest-state cache for drone telemetry.

    Each telemetry field is stored independently with its own monotonic
    timestamp and sequence number. Consumers read atomic snapshots.

    Usage::

        cache = TelemetryCache()
        cache.update_position(lat=38.0, lon=-8.0, alt_m=50.0)
        cache.update_heading(heading_deg=180.0)
        snap = cache.snapshot()
        if not snap.stale(5.0):
            print(snap.position)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fields: dict[str, TelemetryEntry] = {}

    # ── Write API ────────────────────────────────────────────────────────

    def _update(self, key: str, value: object) -> int:
        with self._lock:
            entry = self._fields.get(key, TelemetryEntry())
            entry.value = value
            entry.ts_monotonic = time.monotonic()
            entry.seq += 1
            self._fields[key] = entry
            return entry.seq

    def update_position(self, lat: float, lon: float, alt_m: float) -> int:
        return self._update("position", GeoPoint(lat=lat, lon=lon, alt_m=alt_m))

    def update_heading(self, heading_deg: float) -> int:
        return self._update("heading_deg", heading_deg)

    def update_groundspeed(self, groundspeed_mps: float) -> int:
        return self._update("groundspeed_mps", groundspeed_mps)

    def update_armed(self, armed: bool) -> int:
        return self._update("armed", armed)

    def update_mode(self, mode: str) -> int:
        return self._update("mode", mode)

    def update_battery(self, battery_pct: float) -> int:
        return self._update("battery_pct", battery_pct)

    def update_connected(self, connected: bool) -> int:
        return self._update("connected", connected)

    # ── Read API ─────────────────────────────────────────────────────────

    def _read(self, key: str) -> TelemetryEntry:
        with self._lock:
            return self._fields.get(key, TelemetryEntry())

    def get_position(self) -> tuple[GeoPoint | None, float]:
        entry = self._read("position")
        return entry.value, entry.ts_monotonic

    def get_heading(self) -> tuple[float | None, float]:
        entry = self._read("heading_deg")
        return entry.value, entry.ts_monotonic

    def get_groundspeed(self) -> tuple[float | None, float]:
        entry = self._read("groundspeed_mps")
        return entry.value, entry.ts_monotonic

    def get_armed(self) -> tuple[bool, float]:
        entry = self._read("armed")
        return bool(entry.value) if entry.value is not None else False, entry.ts_monotonic

    def get_mode(self) -> tuple[str, float]:
        entry = self._read("mode")
        return entry.value or "UNKNOWN", entry.ts_monotonic

    def get_battery(self) -> tuple[float | None, float]:
        entry = self._read("battery_pct")
        return entry.value, entry.ts_monotonic

    def is_connected(self) -> bool:
        entry = self._read("connected")
        return bool(entry.value) if entry.value is not None else False

    def age_s(self, key: str) -> float:
        """Seconds since last update for a given field. inf if never updated."""
        entry = self._read(key)
        if entry.ts_monotonic == 0.0:
            return float("inf")
        return time.monotonic() - entry.ts_monotonic

    def is_fresh(self, key: str, threshold_s: float) -> bool:
        """True if field was updated within threshold_s."""
        return self.age_s(key) <= threshold_s

    # ── Snapshot ─────────────────────────────────────────────────────────

    def snapshot(self) -> TelemetrySnapshot:
        """Return an atomic snapshot of all telemetry fields."""
        with self._lock:
            pos_entry = self._fields.get("position", TelemetryEntry())
            head_entry = self._fields.get("heading_deg", TelemetryEntry())
            spd_entry = self._fields.get("groundspeed_mps", TelemetryEntry())
            armed_entry = self._fields.get("armed", TelemetryEntry())
            mode_entry = self._fields.get("mode", TelemetryEntry())
            bat_entry = self._fields.get("battery_pct", TelemetryEntry())
            conn_entry = self._fields.get("connected", TelemetryEntry())

            return TelemetrySnapshot(
                connected=bool(conn_entry.value) if conn_entry.value is not None else False,
                armed=bool(armed_entry.value) if armed_entry.value is not None else False,
                mode=mode_entry.value or "UNKNOWN",
                position=pos_entry.value,
                heading_deg=head_entry.value,
                groundspeed_mps=spd_entry.value,
                battery_pct=bat_entry.value,
                position_age_s=(
                    time.monotonic() - pos_entry.ts_monotonic
                    if pos_entry.ts_monotonic > 0
                    else float("inf")
                ),
            )


@dataclass
class TelemetrySnapshot:
    """Immutable snapshot of all telemetry fields at a point in time."""

    connected: bool = False
    armed: bool = False
    mode: str = "UNKNOWN"
    position: GeoPoint | None = None
    heading_deg: float | None = None
    groundspeed_mps: float | None = None
    battery_pct: float | None = None
    position_age_s: float = float("inf")

    def stale(self, threshold_s: float) -> bool:
        """True if position telemetry is older than threshold_s."""
        return self.position_age_s > threshold_s
