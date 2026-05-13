"""Tests for search pattern generators."""

from sentinel.common.types import GeoPoint
from sentinel.mission_planner.patterns import (
    generate_lawnmower_pattern,
    generate_perimeter_pattern,
)


def _area():
    return [
        GeoPoint(lat=38.000, lon=-8.000, alt_m=80),
        GeoPoint(lat=38.000, lon=-7.998, alt_m=80),
        GeoPoint(lat=38.002, lon=-7.998, alt_m=80),
        GeoPoint(lat=38.002, lon=-8.000, alt_m=80),
    ]


def test_lawnmower_returns_points():
    pts = generate_lawnmower_pattern(_area(), spacing_m=50, altitude_m=80, max_waypoints=100)
    assert len(pts) >= 4
    assert all(p.alt_m == 80 for p in pts)


def test_lawnmower_alternates_direction():
    pts = generate_lawnmower_pattern(_area(), spacing_m=50, altitude_m=80, max_waypoints=100)
    assert len(pts) >= 4
    first_pair_min_lon = min(pts[0].lon, pts[1].lon)
    if len(pts) >= 4:
        second_pair_first_lon = pts[2].lon
        assert second_pair_first_lon > first_pair_min_lon


def test_lawnmower_respects_max_waypoints():
    pts = generate_lawnmower_pattern(_area(), spacing_m=10, altitude_m=80, max_waypoints=6)
    assert len(pts) <= 6


def test_lawnmower_all_points_have_altitude():
    pts = generate_lawnmower_pattern(_area(), spacing_m=50, altitude_m=80, max_waypoints=100)
    for p in pts:
        assert p.alt_m == 80


def test_perimeter_closes_loop():
    area = _area()
    pts = generate_perimeter_pattern(area, altitude_m=80)
    assert pts[0].lat == pts[-1].lat
    assert pts[0].lon == pts[-1].lon


def test_perimeter_all_points_have_altitude():
    pts = generate_perimeter_pattern(_area(), altitude_m=90)
    for p in pts:
        assert p.alt_m == 90


def test_perimeter_returns_input_plus_close():
    area = _area()
    pts = generate_perimeter_pattern(area, altitude_m=80)
    assert len(pts) == len(area) + 1
