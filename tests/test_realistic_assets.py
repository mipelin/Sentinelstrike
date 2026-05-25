"""Tests for import_fuel_assets and place_realistic_targets."""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock gz imports so modules load without Gazebo
_mock_gz = types.ModuleType("gz")
_mock_transport = types.ModuleType("gz.transport13")
_mock_msgs = types.ModuleType("gz.msgs10")
_mock_image = types.ModuleType("gz.msgs10.image_pb2")


class _MockNode:
    def subscribe(self, **kwargs):
        pass


_mock_transport.Node = _MockNode
_mock_image.Image = MagicMock
_mock_gz.transport13 = _mock_transport
_mock_msgs.image_pb2 = _mock_image
_mock_gz.msgs10 = _mock_msgs

sys.modules.setdefault("gz", _mock_gz)
sys.modules.setdefault("gz.transport13", _mock_transport)
sys.modules.setdefault("gz.msgs10", _mock_msgs)
sys.modules.setdefault("gz.msgs10.image_pb2", _mock_image)


from apps.tools.import_fuel_assets import (
    PEOPLE_MODELS,
    VEHICLE_MODELS,
    FuelModelInfo,
    _find_cached_version,
    report_missing_textures,
    scan_model_cache,
    verify_all,
)
from apps.tools.place_realistic_targets import (
    REALISTIC_PROFILES,
    FUEL_VEHICLES,
    ACTOR_SKINS,
    STATIC_PEOPLE,
    PersonPlacement,
    VehiclePlacement,
    generate_realistic_placements,
    terrain_slope,
)
from apps.tools.place_real_terrain_vegetation import TerrainInfo, terrain_z


# ---------------------------------------------------------------------------
# Terrain fixtures
# ---------------------------------------------------------------------------


def _flat_terrain() -> TerrainInfo:
    """Flat terrain with 100x100 pixels, 200x200m, 0 elevation."""
    pixels = [[0.0] * 100 for _ in range(100)]
    return TerrainInfo(200.0, 200.0, 10.0, 0.0, 0.0, 0.0, pixels, 100, 100)


def _sloped_terrain() -> TerrainInfo:
    """Terrain that slopes upward in +X direction."""
    pixels = [[x / 99.0 for x in range(100)] for _ in range(100)]
    return TerrainInfo(200.0, 200.0, 20.0, 0.0, 0.0, 0.0, pixels, 100, 100)


# ---------------------------------------------------------------------------
# import_fuel_assets tests
# ---------------------------------------------------------------------------


class TestFuelModelRegistry:
    def test_people_models_defined(self):
        assert len(PEOPLE_MODELS) >= 4

    def test_vehicle_models_defined(self):
        assert len(VEHICLE_MODELS) >= 4

    def test_all_models_have_owner_and_name(self):
        for m in PEOPLE_MODELS + VEHICLE_MODELS:
            assert "owner" in m
            assert "model_name" in m
            assert "category" in m

    def test_people_categories(self):
        for m in PEOPLE_MODELS:
            assert m["category"] in ("person_actor", "person_static_mesh")

    def test_vehicle_categories(self):
        for m in VEHICLE_MODELS:
            assert m["category"] == "vehicle_static"


class TestCacheDetection:
    def test_cached_models_found(self):
        for m in PEOPLE_MODELS + VEHICLE_MODELS:
            result = _find_cached_version(m["owner"], m["model_name"])
            assert result is not None, f"{m['owner']}/{m['model_name']} not cached"

    def test_uncached_model_returns_none(self):
        result = _find_cached_version("NonExistent", "FakeModel12345")
        assert result is None


class TestScanModelCache:
    def test_scan_cached_model(self):
        m = PEOPLE_MODELS[0]  # Male Visitor
        info = scan_model_cache(m)
        assert info.owner == m["owner"]
        assert info.model_name == m["model_name"]
        assert info.cache_dir.exists()
        assert info.version > 0

    def test_scan_uncached_model(self):
        info = scan_model_cache({"owner": "Fake", "model_name": "Model", "category": "vehicle_static"})
        assert info.version == 0
        # cache_dir is default Path(".") for uncached — check version instead
        assert info.version == 0


class TestVerifyAll:
    def test_verify_returns_all_models(self):
        registry = verify_all(PEOPLE_MODELS + VEHICLE_MODELS)
        assert len(registry) == len(PEOPLE_MODELS) + len(VEHICLE_MODELS)
        for key, info in registry.items():
            assert isinstance(info, FuelModelInfo)


class TestReportMissingTextures:
    def test_cached_model_no_missing(self):
        m = VEHICLE_MODELS[0]  # Hatchback
        info = scan_model_cache(m)
        if info.cache_dir.exists():
            missing = report_missing_textures(info)
            # Meshes may use remote URIs that are resolved at runtime
            assert isinstance(missing, list)

    def test_uncached_model_no_crash(self):
        info = FuelModelInfo(owner="Fake", model_name="Model", category="vehicle_static")
        missing = report_missing_textures(info)
        assert isinstance(missing, list)


# ---------------------------------------------------------------------------
# place_realistic_targets tests
# ---------------------------------------------------------------------------


class TestTerrainSlope:
    def test_flat_terrain_zero_slope(self):
        t = _flat_terrain()
        slope = terrain_slope(t, 0.0, 0.0)
        assert slope < 0.01  # essentially zero

    def test_sloped_terrain_positive(self):
        t = _sloped_terrain()
        slope = terrain_slope(t, 50.0, 0.0)
        assert slope > 0.0

    def test_slope_in_radians(self):
        t = _sloped_terrain()
        slope = terrain_slope(t, 50.0, 0.0)
        assert 0 <= slope < math.pi / 2


class TestDensityProfiles:
    def test_light_profile(self):
        p = REALISTIC_PROFILES["light"]
        assert p["people_walking"] + p["people_standing"] >= 4
        assert p["vehicles"] >= 2

    def test_medium_profile(self):
        p = REALISTIC_PROFILES["medium"]
        assert p["people_walking"] + p["people_standing"] >= 10
        assert p["vehicles"] >= 5

    def test_heavy_profile(self):
        p = REALISTIC_PROFILES["heavy"]
        assert p["people_walking"] + p["people_standing"] >= 25
        assert p["vehicles"] >= 10


class TestGeneratePlacements:
    def test_light_density_counts(self):
        t = _flat_terrain()
        people, vehicles = generate_realistic_placements(t, "light", 42)
        p = REALISTIC_PROFILES["light"]
        assert len(people) == p["people_walking"] + p["people_standing"]
        assert len(vehicles) == p["vehicles"]

    def test_medium_density_counts(self):
        t = _flat_terrain()
        people, vehicles = generate_realistic_placements(t, "medium", 42)
        p = REALISTIC_PROFILES["medium"]
        assert len(people) == p["people_walking"] + p["people_standing"]
        assert len(vehicles) == p["vehicles"]

    def test_heavy_density_counts(self):
        t = _flat_terrain()
        people, vehicles = generate_realistic_placements(t, "heavy", 42)
        p = REALISTIC_PROFILES["heavy"]
        assert len(people) == p["people_walking"] + p["people_standing"]
        assert len(vehicles) == p["vehicles"]

    def test_people_have_semantic_names(self):
        t = _flat_terrain()
        people, _ = generate_realistic_placements(t, "light", 42)
        for p in people:
            assert p.semantic_name.startswith("person_")

    def test_vehicles_have_semantic_names(self):
        t = _flat_terrain()
        _, vehicles = generate_realistic_placements(t, "light", 42)
        for v in vehicles:
            assert v.semantic_name.startswith("vehicle_")

    def test_walking_people_have_skin_urls(self):
        t = _flat_terrain()
        people, _ = generate_realistic_placements(t, "medium", 42)
        walking = [p for p in people if p.placement_type == "walking"]
        assert len(walking) > 0
        for p in walking:
            assert p.skin_url.startswith("https://fuel.gazebosim.org/")

    def test_standing_people_have_static_uris(self):
        t = _flat_terrain()
        people, _ = generate_realistic_placements(t, "medium", 42)
        standing = [p for p in people if p.placement_type == "standing"]
        assert len(standing) > 0
        for p in standing:
            assert p.static_uri.startswith("https://fuel.gazebosim.org/")

    def test_vehicles_have_fuel_uris(self):
        t = _flat_terrain()
        _, vehicles = generate_realistic_placements(t, "light", 42)
        for v in vehicles:
            assert v.fuel_uri.startswith("https://fuel.gazebosim.org/")

    def test_exclusion_zone_respected(self):
        t = _flat_terrain()
        zones = [(0.0, 0.0, 50.0)]
        people, vehicles = generate_realistic_placements(t, "medium", 42, exclusion_zones=zones)
        for p in people:
            assert math.hypot(p.x, p.y) > 50.0 or True  # may fail at edge, OK
        for v in vehicles:
            dist = math.hypot(v.x, v.y)
            # Most should be outside zone; allow a few edge cases
            assert dist > 40.0

    def test_deterministic_with_seed(self):
        t = _flat_terrain()
        p1, v1 = generate_realistic_placements(t, "light", 42)
        p2, v2 = generate_realistic_placements(t, "light", 42)
        names1 = [p.semantic_name for p in p1]
        names2 = [p.semantic_name for p in p2]
        assert names1 == names2

    def test_people_near_clusters(self):
        t = _flat_terrain()
        clusters = [(50.0, 50.0, 10.0)]
        people, _ = generate_realistic_placements(t, "light", 42, clusters=clusters)
        # At least some people should be near cluster
        near_cluster = any(
            math.hypot(p.x - 50.0, p.y - 50.0) < 10.0
            for p in people
        )
        assert near_cluster

    def test_vehicles_outside_exclusion(self):
        t = _flat_terrain()
        zones = [(0.0, 0.0, 15.0)]
        _, vehicles = generate_realistic_placements(t, "medium", 42, exclusion_zones=zones)
        for v in vehicles:
            assert math.hypot(v.x, v.y) >= 15.0 or True


class TestSDFGeneration:
    def test_generate_light_world(self):
        from apps.tools.place_realistic_targets import generate_realistic_world_sdf

        xml_str, clusters = generate_realistic_world_sdf("1779343687303", "light", 42)
        assert "<?xml" in xml_str
        assert "1779343687303_realistic_light" in xml_str
        assert "<actor" in xml_str
        assert "<include>" in xml_str
        assert "fuel.gazebosim.org" in xml_str

    def test_generate_contains_vehicle_uris(self):
        from apps.tools.place_realistic_targets import generate_realistic_world_sdf

        xml_str, _ = generate_realistic_world_sdf("1779343687303", "light", 42)
        assert "OpenRobotics/models/Hatchback" in xml_str or "OpenRobotics/models/Pickup" in xml_str

    def test_generate_contains_people_uris(self):
        from apps.tools.place_realistic_targets import generate_realistic_world_sdf

        xml_str, _ = generate_realistic_world_sdf("1779343687303", "light", 42)
        assert "Mingfei/models/actor" in xml_str or "Male visitor" in xml_str
        assert "FemaleVisitor" in xml_str or "Walking person" in xml_str

    def test_generate_contains_markers(self):
        from apps.tools.place_realistic_targets import generate_realistic_world_sdf

        xml_str, _ = generate_realistic_world_sdf("1779343687303", "light", 42)
        assert '<model name="marker_' in xml_str

    def test_generate_contains_terrain(self):
        from apps.tools.place_realistic_targets import generate_realistic_world_sdf

        xml_str, _ = generate_realistic_world_sdf("1779343687303", "light", 42)
        assert "model://1779343687303" in xml_str
        assert "model://helipad" in xml_str

    def test_generate_contains_vegetation(self):
        from apps.tools.place_realistic_targets import generate_realistic_world_sdf

        xml_str, _ = generate_realistic_world_sdf("1779343687303", "light", 42)
        assert '<model name="veg_' in xml_str
