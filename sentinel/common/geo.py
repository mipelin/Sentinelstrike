"""Geodetic helpers — no external dependencies."""

from __future__ import annotations

import math

from sentinel.common.types import GeoPoint

_EARTH_RADIUS_M = 6_371_000.0


def haversine_distance_m(a: GeoPoint, b: GeoPoint) -> float:
    lat1, lon1 = math.radians(a.lat), math.radians(a.lon)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(h))


def centroid(points: list[GeoPoint]) -> GeoPoint:
    n = len(points)
    mean_lat = sum(p.lat for p in points) / n
    mean_lon = sum(p.lon for p in points) / n
    alts = [p.alt_m for p in points if p.alt_m is not None]
    mean_alt = sum(alts) / len(alts) if alts else None
    return GeoPoint(lat=mean_lat, lon=mean_lon, alt_m=mean_alt)


def meters_per_degree_lat() -> float:
    return 111320.0


def meters_per_degree_lon(lat: float) -> float:
    return 111320.0 * math.cos(math.radians(lat))


def offset_point(origin: GeoPoint, north_m: float, east_m: float, alt_m: float | None = None) -> GeoPoint:
    dlat = north_m / meters_per_degree_lat()
    dlon = east_m / meters_per_degree_lon(origin.lat)
    return GeoPoint(
        lat=origin.lat + dlat,
        lon=origin.lon + dlon,
        alt_m=alt_m if alt_m is not None else origin.alt_m,
    )


def bounding_box(points: list[GeoPoint]) -> dict:
    lats = [p.lat for p in points]
    lons = [p.lon for p in points]
    return {
        "min_lat": min(lats),
        "max_lat": max(lats),
        "min_lon": min(lons),
        "max_lon": max(lons),
    }


def path_distance_m(points: list[GeoPoint]) -> float:
    total = 0.0
    for i in range(len(points) - 1):
        total += haversine_distance_m(points[i], points[i + 1])
    return total


def is_point_in_bbox(point: GeoPoint, bbox: dict) -> bool:
    return bbox["min_lat"] <= point.lat <= bbox["max_lat"] and bbox["min_lon"] <= point.lon <= bbox["max_lon"]
