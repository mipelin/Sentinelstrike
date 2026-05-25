"""Lightweight geospatial layer for ISR target history.

Persists:
- World-space target trails (lat/lon paths)
- Last-seen positions per target
- Simple activity heatmap (grid-based)
- Behavioral heatmaps: loitering zones, concealment hotspots, interaction zones

NOT a GIS database. Uses in-memory dicts optimized for ISR overlay use.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .terrain_reasoning import TerrainReasoner


@dataclass
class TargetHistory:
    """Historical world-space record for a single target."""
    track_id: str
    class_name: str = ""
    trail: list[tuple[float, float, float]] = field(default_factory=list)  # (lat, lon, timestamp)
    last_seen_lat: float = 0.0
    last_seen_lon: float = 0.0
    last_seen_time: float = 0.0
    first_seen_time: float = 0.0
    total_detections: int = 0
    max_speed_mps: float = 0.0

    _max_trail: int = 1000

    def add_point(self, lat: float, lon: float, speed_mps: float = 0.0) -> None:
        now = time.monotonic()
        if not self.first_seen_time:
            self.first_seen_time = now
        self.last_seen_lat = lat
        self.last_seen_lon = lon
        self.last_seen_time = now
        self.total_detections += 1
        self.max_speed_mps = max(self.max_speed_mps, speed_mps)
        self.trail.append((lat, lon, now))
        if len(self.trail) > self._max_trail:
            self.trail = self.trail[-self._max_trail:]


_LOITERING_SPEED_MPS = 0.3
_CELL_DECAY = 0.999


class GeospatialLayer:
    """In-memory geospatial history for tracked targets."""

    def __init__(
        self,
        grid_resolution_m: float = 5.0,
        terrain: TerrainReasoner | None = None,
    ) -> None:
        self._targets: dict[str, TargetHistory] = {}
        self._heatmap: dict[tuple[int, int], int] = {}
        self._grid_res = grid_resolution_m
        self._origin_lat: float | None = None
        self._origin_lon: float | None = None
        self._terrain = terrain
        # Behavioral heatmaps
        self._loitering_heatmap: dict[tuple[int, int], float] = {}
        self._concealment_heatmap: dict[tuple[int, int], int] = {}
        self._interaction_heatmap: dict[tuple[int, int], int] = {}

    def update(
        self,
        tracks: list[dict],
        interaction_pairs: list[tuple[str, str]] | None = None,
    ) -> None:
        """Update geospatial layer from tracker output dicts."""
        for t in tracks:
            tid = t.get("track_id")
            if not tid:
                continue
            wp = t.get("world_position")
            if not wp or not wp.get("lat"):
                continue

            if tid not in self._targets:
                self._targets[tid] = TargetHistory(
                    track_id=tid,
                    class_name=t.get("class", ""),
                )

            wv = t.get("world_velocity", {})
            speed = wv.get("speed_mps", 0) if wv else 0
            self._targets[tid].add_point(wp["lat"], wp["lon"], speed)

            if self._origin_lat is None:
                self._origin_lat = wp["lat"]
                self._origin_lon = wp["lon"]

            # Update heatmap
            cell = self._to_cell(wp["lat"], wp["lon"])
            self._heatmap[cell] = self._heatmap.get(cell, 0) + 1

            # Loitering: accumulate time where speed is near zero
            if speed < _LOITERING_SPEED_MPS:
                self._loitering_heatmap[cell] = self._loitering_heatmap.get(cell, 0.0) + 1.0

            # Concealment: near vegetation zones
            if self._terrain and self._terrain.zone_count > 0:
                ttype = self._terrain.terrain_type_at(wp["lat"], wp["lon"])
                if ttype == "vegetation":
                    self._concealment_heatmap[cell] = self._concealment_heatmap.get(cell, 0) + 1

        # Interaction heatmap: midpoint of interacting pairs
        if interaction_pairs:
            for id_a, id_b in interaction_pairs:
                ta = self._targets.get(id_a)
                tb = self._targets.get(id_b)
                if ta and tb and ta.last_seen_lat and tb.last_seen_lat:
                    mid_lat = (ta.last_seen_lat + tb.last_seen_lat) / 2.0
                    mid_lon = (ta.last_seen_lon + tb.last_seen_lon) / 2.0
                    cell = self._to_cell(mid_lat, mid_lon)
                    self._interaction_heatmap[cell] = self._interaction_heatmap.get(cell, 0) + 1

    def get_trail(self, track_id: str, max_points: int = 200) -> list[tuple[float, float]]:
        """Get world-space trail for a target as [(lat, lon), ...]."""
        target = self._targets.get(track_id)
        if not target:
            return []
        return [(lat, lon) for lat, lon, _ in target.trail[-max_points:]]

    def get_last_seen(self, track_id: str) -> tuple[float, float] | None:
        """Get last known position for a target."""
        target = self._targets.get(track_id)
        if not target or not target.last_seen_lat:
            return None
        return (target.last_seen_lat, target.last_seen_lon)

    def get_all_last_seen(self) -> dict[str, tuple[float, float, float]]:
        """Get last known positions for all targets.

        Returns: {track_id: (lat, lon, seconds_ago)}
        """
        now = time.monotonic()
        result = {}
        for tid, t in self._targets.items():
            if t.last_seen_lat:
                result[tid] = (t.last_seen_lat, t.last_seen_lon, now - t.last_seen_time)
        return result

    def get_target_summary(self, track_id: str) -> dict | None:
        """Get summary stats for a target."""
        target = self._targets.get(track_id)
        if not target:
            return None
        now = time.monotonic()
        return {
            "track_id": track_id,
            "class_name": target.class_name,
            "total_detections": target.total_detections,
            "trail_points": len(target.trail),
            "first_seen_s_ago": round(now - target.first_seen_time, 1) if target.first_seen_time else 0,
            "last_seen_s_ago": round(now - target.last_seen_time, 1) if target.last_seen_time else 0,
            "max_speed_mps": round(target.max_speed_mps, 2),
        }

    def get_heatmap(self, bounds: tuple[float, float, float, float] | None = None) -> list[dict]:
        """Get activity heatmap cells."""
        return self._render_heatmap(self._heatmap, bounds, value_key="count")

    def get_loitering_zones(self, min_seconds: float = 5.0) -> list[dict]:
        """Get cells where targets spent significant time stationary."""
        fps_approx = 15.0
        min_frames = min_seconds * fps_approx
        filtered = {k: v for k, v in self._loitering_heatmap.items() if v >= min_frames}
        return self._render_heatmap(filtered, value_key="seconds",
                                     transform=lambda v: round(v / fps_approx, 1))

    def get_concealment_hotspots(self, min_count: int = 3) -> list[dict]:
        """Get cells with repeated concealment-seeking visits."""
        filtered = {k: v for k, v in self._concealment_heatmap.items() if v >= min_count}
        return self._render_heatmap(filtered, value_key="count")

    def get_interaction_zones(self) -> list[dict]:
        """Get cells where target interactions occurred."""
        return self._render_heatmap(self._interaction_heatmap, value_key="count")

    def _render_heatmap(
        self,
        data: dict[tuple[int, int], float | int],
        bounds: tuple[float, float, float, float] | None = None,
        value_key: str = "count",
        transform=None,
    ) -> list[dict]:
        if self._origin_lat is None or not data:
            return []
        results = []
        for (ci, cj), value in data.items():
            lat = self._origin_lat + (ci * self._grid_res) / 111320.0
            lon = self._origin_lon + (cj * self._grid_res) / (111320.0 * math.cos(math.radians(self._origin_lat)))
            if bounds:
                lat_min, lon_min, lat_max, lon_max = bounds
                if lat < lat_min or lat > lat_max or lon < lon_min or lon > lon_max:
                    continue
            results.append({
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                value_key: transform(value) if transform else value,
            })
        return results

    def _to_cell(self, lat: float, lon: float) -> tuple[int, int]:
        if self._origin_lat is None:
            return (0, 0)
        dn = (lat - self._origin_lat) * 111320.0
        de = (lon - self._origin_lon) * 111320.0 * math.cos(math.radians(self._origin_lat))
        return (int(dn / self._grid_res), int(de / self._grid_res))

    @property
    def target_count(self) -> int:
        return len(self._targets)

    @property
    def total_trail_points(self) -> int:
        return sum(len(t.trail) for t in self._targets.values())
