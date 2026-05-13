"""Search pattern generators for mission planning.

v1 uses bounding-box approximation. Polygon clipping is future work.
"""

from __future__ import annotations

from sentinel.common.geo import meters_per_degree_lat, meters_per_degree_lon
from sentinel.common.types import GeoPoint


def generate_lawnmower_pattern(
    area_of_interest: list[GeoPoint],
    spacing_m: float,
    altitude_m: float,
    max_waypoints: int,
) -> list[GeoPoint]:
    lats = [p.lat for p in area_of_interest]
    lons = [p.lon for p in area_of_interest]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    mid_lat = (min_lat + max_lat) / 2
    dlat = spacing_m / meters_per_degree_lat()
    _dlon = spacing_m / meters_per_degree_lon(mid_lat)  # noqa: F841 — reserved for polygon clipping

    rows = []
    lat = min_lat
    while lat <= max_lat:
        rows.append(lat)
        lat += dlat

    if len(rows) < 2:
        rows = [min_lat, max_lat]

    waypoints: list[GeoPoint] = []
    for i, row_lat in enumerate(rows):
        if len(waypoints) >= max_waypoints:
            break
        if i % 2 == 0:
            waypoints.append(GeoPoint(lat=row_lat, lon=min_lon, alt_m=altitude_m))
            if len(waypoints) < max_waypoints:
                waypoints.append(GeoPoint(lat=row_lat, lon=max_lon, alt_m=altitude_m))
        else:
            waypoints.append(GeoPoint(lat=row_lat, lon=max_lon, alt_m=altitude_m))
            if len(waypoints) < max_waypoints:
                waypoints.append(GeoPoint(lat=row_lat, lon=min_lon, alt_m=altitude_m))

    if len(waypoints) < 4:
        waypoints = [
            GeoPoint(lat=min_lat, lon=min_lon, alt_m=altitude_m),
            GeoPoint(lat=min_lat, lon=max_lon, alt_m=altitude_m),
            GeoPoint(lat=max_lat, lon=max_lon, alt_m=altitude_m),
            GeoPoint(lat=max_lat, lon=min_lon, alt_m=altitude_m),
        ]

    return waypoints[:max_waypoints]


def generate_perimeter_pattern(
    area_of_interest: list[GeoPoint],
    altitude_m: float,
) -> list[GeoPoint]:
    points = [GeoPoint(lat=p.lat, lon=p.lon, alt_m=altitude_m) for p in area_of_interest]
    if points and points[0] != points[-1]:
        points.append(GeoPoint(lat=points[0].lat, lon=points[0].lon, alt_m=altitude_m))
    return points
