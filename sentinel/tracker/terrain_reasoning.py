"""Lightweight terrain-aware occlusion reasoning for ISR tracking.

Provides:
- Vegetation zone awareness (static obstacles from world definition)
- Per-track occlusion probability estimation
- Emergence point prediction for hidden targets
- Movement path priors based on terrain type (road vs open vs vegetation)

NOT a full SLAM system. Uses pre-defined vegetation zones and simple
geometric reasoning for ISR-grade estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class VegetationZone:
    """A known vegetation/obstacle region in world coordinates."""
    name: str
    center_lat: float
    center_lon: float
    radius_m: float
    height_m: float = 3.0
    density: float = 0.7  # 0-1, how opaque/dense
    zone_type: str = "tree_cluster"  # tree_cluster, bush, building, wall

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a lat/lon point is inside this zone."""
        dn = (lat - self.center_lat) * 111320.0
        de = (lon - self.center_lon) * 111320.0 * math.cos(math.radians(self.center_lat))
        return math.sqrt(dn * dn + de * de) <= self.radius_m

    def distance_to(self, lat: float, lon: float) -> float:
        """Distance in meters from point to zone edge (negative if inside)."""
        dn = (lat - self.center_lat) * 111320.0
        de = (lon - self.center_lon) * 111320.0 * math.cos(math.radians(self.center_lat))
        dist = math.sqrt(dn * dn + de * de)
        return dist - self.radius_m


@dataclass
class RoadSegment:
    """A known road or path segment for movement priors."""
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    width_m: float = 3.0
    name: str = ""

    def distance_to(self, lat: float, lon: float) -> float:
        """Minimum distance in meters from point to road center line."""
        # Convert to local meters
        origin_lat = self.start_lat
        origin_lon = self.start_lon
        s_n = 0.0
        s_e = 0.0
        e_n = (self.end_lat - origin_lat) * 111320.0
        e_e = (self.end_lon - origin_lon) * 111320.0 * math.cos(math.radians(origin_lat))
        p_n = (lat - origin_lat) * 111320.0
        p_e = (lon - origin_lon) * 111320.0 * math.cos(math.radians(origin_lat))

        dx, dy = e_n - s_n, e_e - s_e
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-6:
            return math.sqrt(p_n * p_n + p_e * p_e)

        t = max(0.0, min(1.0, (p_n * dx + p_e * dy) / length_sq))
        proj_n = s_n + t * dx
        proj_e = s_e + t * dy
        d = math.sqrt((p_n - proj_n) ** 2 + (p_e - proj_e) ** 2)
        return max(0.0, d - self.width_m / 2.0)

    def nearest_point(self, lat: float, lon: float) -> tuple[float, float]:
        """Nearest point on road center line as (lat, lon)."""
        origin_lat = self.start_lat
        origin_lon = self.start_lon
        e_n = (self.end_lat - origin_lat) * 111320.0
        e_e = (self.end_lon - origin_lon) * 111320.0 * math.cos(math.radians(origin_lat))
        p_n = (lat - origin_lat) * 111320.0
        p_e = (lon - origin_lon) * 111320.0 * math.cos(math.radians(origin_lat))

        length_sq = e_n * e_n + e_e * e_e
        if length_sq < 1e-6:
            return self.start_lat, self.start_lon

        t = max(0.0, min(1.0, (p_n * e_n + p_e * e_e) / length_sq))
        n = t * e_n
        e = t * e_e
        result_lat = origin_lat + n / 111320.0
        result_lon = self.start_lon + e / (111320.0 * math.cos(math.radians(origin_lat)))
        return result_lat, result_lon


@dataclass
class EmergencePoint:
    """Predicted re-appearance location for an occluded target."""
    lat: float
    lon: float
    probability: float  # 0-1
    source: str  # "velocity_extrapolation", "nearest_road", "zone_edge"
    radius_m: float = 5.0  # search radius around point


class TerrainReasoner:
    """Lightweight terrain-aware occlusion and movement reasoning."""

    def __init__(self) -> None:
        self._vegetation_zones: list[VegetationZone] = []
        self._roads: list[RoadSegment] = []

    def add_vegetation_zone(self, zone: VegetationZone) -> None:
        self._vegetation_zones.append(zone)

    def add_road(self, road: RoadSegment) -> None:
        self._roads.append(road)

    def load_zones_from_list(self, zones: list[dict]) -> None:
        """Load vegetation zones from a list of dicts.

        Each dict: {name, lat, lon, radius_m, height_m?, density?, type?}
        """
        for z in zones:
            self._vegetation_zones.append(VegetationZone(
                name=z.get("name", ""),
                center_lat=z["lat"],
                center_lon=z["lon"],
                radius_m=z["radius_m"],
                height_m=z.get("height_m", 3.0),
                density=z.get("density", 0.7),
                zone_type=z.get("type", "tree_cluster"),
            ))

    def load_roads_from_list(self, roads: list[dict]) -> None:
        """Load roads from a list of dicts.

        Each dict: {start_lat, start_lon, end_lat, end_lon, width_m?, name?}
        """
        for r in roads:
            self._roads.append(RoadSegment(
                start_lat=r["start_lat"],
                start_lon=r["start_lon"],
                end_lat=r["end_lat"],
                end_lon=r["end_lon"],
                width_m=r.get("width_m", 3.0),
                name=r.get("name", ""),
            ))

    def occlusion_probability(
        self,
        lat: float,
        lon: float,
        drone_alt_m: float,
        drone_lat: float,
        drone_lon: float,
    ) -> float:
        """Estimate probability that a target at (lat, lon) is occluded.

        Based on:
        - Whether the line-of-sight from drone to target passes through vegetation
        - Zone density
        - Viewing angle (steeper = less occlusion)

        Returns 0.0 (fully visible) to 1.0 (fully occluded).
        """
        if not self._vegetation_zones:
            return 0.0

        dn = (lat - drone_lat) * 111320.0
        de = (lon - drone_lon) * 111320.0 * math.cos(math.radians(drone_lat))
        ground_dist = math.sqrt(dn * dn + de * de)

        if ground_dist < 1.0:
            return 0.0

        # Steeper viewing angle (more overhead) reduces occlusion
        view_angle_rad = math.atan2(drone_alt_m, ground_dist)
        angle_factor = max(0.0, 1.0 - view_angle_rad / (math.pi / 4.0))

        max_occ = 0.0
        for zone in self._vegetation_zones:
            if not zone.contains(lat, lon) and not self._los_intersects_zone(
                drone_lat, drone_lon, lat, lon, zone,
            ):
                continue

            # Target inside zone or LOS passes through it
            dist_to_target = zone.distance_to(lat, lon)
            if dist_to_target <= 0:
                # Target is inside the zone
                depth_factor = min(1.0, zone.radius_m / 10.0)
                occ = zone.density * angle_factor * depth_factor
            else:
                # LOS passes through zone on the way to target
                penetration = max(0.0, zone.radius_m - dist_to_target)
                depth_factor = min(1.0, penetration / 10.0)
                occ = zone.density * angle_factor * depth_factor * 0.5

            max_occ = max(max_occ, occ)

        return min(1.0, max_occ)

    def _los_intersects_zone(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float,
        zone: VegetationZone,
    ) -> bool:
        """Check if line-of-sight between two points intersects a vegetation zone."""
        # Convert to local meters from lat1
        origin_lat = lat1
        p2_n = (lat2 - origin_lat) * 111320.0
        p2_e = (lon2 - lon1) * 111320.0 * math.cos(math.radians(origin_lat))
        z_n = (zone.center_lat - origin_lat) * 111320.0
        z_e = (zone.center_lon - lon1) * 111320.0 * math.cos(math.radians(origin_lat))

        seg_len_sq = p2_n * p2_n + p2_e * p2_e
        if seg_len_sq < 1e-6:
            return False

        # Project zone center onto LOS segment
        t = (z_n * p2_n + z_e * p2_e) / seg_len_sq
        if t < 0.0 or t > 1.0:
            return False

        closest_n = t * p2_n
        closest_e = t * p2_e
        dist = math.sqrt((z_n - closest_n) ** 2 + (z_e - closest_e) ** 2)
        return dist <= zone.radius_m

    def predict_emergence_points(
        self,
        last_lat: float,
        last_lon: float,
        speed_mps: float,
        heading_rad: float,
        frames_lost: int,
        fps: float = 15.0,
        max_points: int = 5,
    ) -> list[EmergencePoint]:
        """Predict where an occluded target might reappear.

        Generates multiple hypotheses:
        1. Velocity extrapolation (continue moving in same direction)
        2. Nearest road/path (if target was near a road)
        3. Zone edge exits (closest edges of vegetation zone)

        Returns sorted by probability (highest first), up to max_points.
        """
        dt = frames_lost / fps
        candidates: list[EmergencePoint] = []

        # Hypothesis 1: Velocity extrapolation
        if speed_mps > 0.1:
            for factor in [1.0, 0.7, 1.3]:
                dist = speed_mps * dt * factor
                dn = dist * math.cos(heading_rad)
                de = dist * math.sin(heading_rad)
                pred_lat = last_lat + dn / 111320.0
                pred_lon = last_lon + de / (111320.0 * math.cos(math.radians(last_lat)))
                prob = max(0.0, 0.6 - abs(factor - 1.0) * 0.5)
                candidates.append(EmergencePoint(
                    lat=pred_lat, lon=pred_lon,
                    probability=prob,
                    source="velocity_extrapolation",
                    radius_m=max(3.0, speed_mps * dt * 0.3),
                ))

        # Hypothesis 2: Nearest road exit
        for road in self._roads:
            road_dist = road.distance_to(last_lat, last_lon)
            if road_dist < 15.0:
                rlat, rlon = road.nearest_point(last_lat, last_lon)
                # Walk along road in both directions
                for sign in [1.0, -1.0]:
                    dn = sign * speed_mps * dt
                    r_n = (rlat - road.start_lat) * 111320.0
                    r_e = (rlon - road.start_lon) * 111320.0 * math.cos(math.radians(road.start_lat))
                    rd_n = (road.end_lat - road.start_lat) * 111320.0
                    rd_e = (road.end_lon - road.start_lon) * 111320.0 * math.cos(math.radians(road.start_lat))
                    rd_len = math.sqrt(rd_n * rd_n + rd_e * rd_e)
                    if rd_len > 0.1:
                        along = dn / rd_len
                        t_road = (r_n * rd_n + r_e * rd_e) / (rd_len * rd_len)
                        t_new = max(0.0, min(1.0, t_road + along))
                        new_n = t_new * rd_n
                        new_e = t_new * rd_e
                        pred_lat = road.start_lat + new_n / 111320.0
                        pred_lon = road.start_lon + new_e / (111320.0 * math.cos(math.radians(road.start_lat)))
                        prob = max(0.0, 0.4 - road_dist / 30.0)
                        candidates.append(EmergencePoint(
                            lat=pred_lat, lon=pred_lon,
                            probability=prob,
                            source="nearest_road",
                            radius_m=road.width_m + 2.0,
                        ))

        # Hypothesis 3: Zone edge exits
        for zone in self._vegetation_zones:
            if zone.distance_to(last_lat, last_lon) < 0:
                # Target was inside this zone — predict exits at cardinal edges
                for angle_deg in [0, 90, 180, 270]:
                    angle_rad = math.radians(angle_deg)
                    exit_n = zone.radius_m * math.cos(angle_rad)
                    exit_e = zone.radius_m * math.sin(angle_rad)
                    exit_lat = zone.center_lat + exit_n / 111320.0
                    exit_lon = zone.center_lon + exit_e / (111320.0 * math.cos(math.radians(zone.center_lat)))
                    # Weight exits in heading direction higher
                    heading_diff = abs(((angle_deg - math.degrees(heading_rad) + 180) % 360) - 180)
                    prob = max(0.0, 0.3 - heading_diff / 360.0)
                    candidates.append(EmergencePoint(
                        lat=exit_lat, lon=exit_lon,
                        probability=prob,
                        source="zone_edge",
                        radius_m=5.0,
                    ))

        # Sort by probability, return top N
        candidates.sort(key=lambda p: p.probability, reverse=True)
        return candidates[:max_points]

    def terrain_type_at(self, lat: float, lon: float) -> str:
        """Classify terrain at a point: 'open', 'vegetation', 'road'."""
        for road in self._roads:
            if road.distance_to(lat, lon) <= 0:
                return "road"
        for zone in self._vegetation_zones:
            if zone.contains(lat, lon):
                return "vegetation"
        return "open"

    def movement_prior(
        self,
        lat: float,
        lon: float,
        heading_rad: float,
        speed_mps: float,
    ) -> tuple[float, float]:
        """Adjust velocity prediction based on terrain.

        Returns (adjusted_speed_mps, adjusted_heading_rad).
        Targets near roads tend to stay on roads. Targets in vegetation
        move slower. Targets in open terrain maintain speed.
        """
        terrain = self.terrain_type_at(lat, lon)

        if terrain == "road" and speed_mps > 0.1:
            # Snap heading to nearest road direction
            best_heading = heading_rad
            best_dist = float("inf")
            for road in self._roads:
                if road.distance_to(lat, lon) > 0:
                    continue
                rd_n = (road.end_lat - road.start_lat) * 111320.0
                rd_e = (road.end_lon - road.start_lon) * 111320.0 * math.cos(math.radians(road.start_lat))
                road_heading = math.atan2(rd_e, rd_n)
                for rh in [road_heading, road_heading + math.pi]:
                    diff = abs(((math.degrees(rh - heading_rad) + 180) % 360) - 180)
                    if diff < best_dist:
                        best_dist = diff
                        best_heading = rh
            # Gentle nudge toward road heading
            if best_dist < 45:
                heading_diff = best_heading - heading_rad
                heading_rad += heading_diff * 0.3

        elif terrain == "vegetation":
            speed_mps *= 0.5  # Slower in vegetation

        return speed_mps, heading_rad

    @property
    def zone_count(self) -> int:
        return len(self._vegetation_zones)

    @property
    def road_count(self) -> int:
        return len(self._roads)
