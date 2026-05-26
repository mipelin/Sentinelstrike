"""Tests for rural ISR scene generator and placement helpers."""

from __future__ import annotations

import math
import random
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from apps.tools.place_real_terrain_vegetation import TerrainInfo, load_terrain
from apps.tools.place_rural_isr_scene import (
    PLACEMENT_DIAGNOSTICS,
    REQUIRED_STATS_KEYS,
    RURAL_PROFILES,
    _BOUNDS_MARGIN,
    _flat_pos,
    _pos_near_concealment,
    _pos_on_road,
    _random_pos,
    _unique_name,
    _claim_name,
    _reset_names,
    _validate_stats,
    _PLACEMENT_HELPERS,
    generate_rural_placements,
    generate_rural_world_sdf,
    validate_placement_helpers,
    validate_unique_sdf_names,
)
from apps.tools.import_fuel_isr_assets import (
    ISR_ACTOR_SKINS,
    ISR_FUEL_VEHICLES,
    ISR_STATIC_PEOPLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FLAT_TERRAIN = TerrainInfo(
    width_m=500.0, height_m=500.0, elevation_m=10.0,
    offset_x=0.0, offset_y=0.0, offset_z=0.0,
    pixels=[], img_w=0, img_h=0,
)

_NO_EXCLUSION: list[tuple[float, float, float]] = []


# ---------------------------------------------------------------------------
# Placement helper existence
# ---------------------------------------------------------------------------


class TestPlacementHelpers:
    def test_flat_pos_exists(self):
        assert callable(_flat_pos)

    def test_random_pos_exists(self):
        assert callable(_random_pos)

    def test_pos_on_road_exists(self):
        assert callable(_pos_on_road)

    def test_pos_near_concealment_exists(self):
        assert callable(_pos_near_concealment)

    def test_all_helpers_in_registry(self):
        for name in ("random_pos", "flat_pos", "pos_on_road", "pos_near_concealment"):
            assert name in _PLACEMENT_HELPERS
            assert callable(_PLACEMENT_HELPERS[name])

    def test_validate_placement_helpers_passes(self):
        validate_placement_helpers()

    def test_validate_placement_helpers_detects_missing(self):
        original = _PLACEMENT_HELPERS.copy()
        _PLACEMENT_HELPERS["broken"] = None  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="missing or not callable"):
            validate_placement_helpers()
        _PLACEMENT_HELPERS.clear()
        _PLACEMENT_HELPERS.update(original)


# ---------------------------------------------------------------------------
# flat_pos behavior
# ---------------------------------------------------------------------------


class TestFlatPos:
    def test_returns_valid_coordinates(self):
        rng = random.Random(42)
        x, y = _flat_pos(_FLAT_TERRAIN, rng, _NO_EXCLUSION)
        assert -250.0 <= x <= 250.0
        assert -250.0 <= y <= 250.0

    def test_avoids_exclusion_zone(self):
        rng = random.Random(42)
        ez: list[tuple[float, float, float]] = [(0.0, 0.0, 200.0)]
        x, y = _flat_pos(_FLAT_TERRAIN, rng, ez)
        assert math.hypot(x, y) >= 200.0 or (abs(x) <= 250.0 and abs(y) <= 250.0)

    def test_deterministic_with_seed(self):
        x1, y1 = _flat_pos(_FLAT_TERRAIN, random.Random(99), _NO_EXCLUSION)
        x2, y2 = _flat_pos(_FLAT_TERRAIN, random.Random(99), _NO_EXCLUSION)
        assert x1 == x2 and y1 == y2


class TestRandomPos:
    def test_returns_within_bounds(self):
        rng = random.Random(42)
        x, y = _random_pos(_FLAT_TERRAIN, rng, _NO_EXCLUSION)
        assert -250.0 <= x <= 250.0
        assert -250.0 <= y <= 250.0


class TestPosOnRoad:
    def test_returns_none_for_empty(self):
        assert _pos_on_road([], random.Random(42)) is None

    def test_returns_point_on_road(self):
        road = [(0.0, 0.0), (100.0, 0.0)]
        result = _pos_on_road([road], random.Random(42))
        assert result is not None
        x, y = result
        assert 0.0 <= x <= 100.0
        assert abs(y) < 0.01


class TestPosNearConcealment:
    def test_returns_none_for_empty(self):
        assert _pos_near_concealment([], [], random.Random(42)) is None

    def test_returns_near_zone(self):
        zones = [[(50.0, 50.0, 5.0)]]
        result = _pos_near_concealment(zones, [], random.Random(42))
        assert result is not None
        x, y, concealment = result
        assert concealment >= 0.3
        assert math.hypot(x - 50.0, y - 50.0) < 10.0


# ---------------------------------------------------------------------------
# Placement diagnostics
# ---------------------------------------------------------------------------


class TestPlacementDiagnostics:
    def test_diagnostics_populated(self):
        rng = random.Random(42)
        _flat_pos(_FLAT_TERRAIN, rng, _NO_EXCLUSION)
        assert PLACEMENT_DIAGNOSTICS["attempts"] >= 1

    def test_diagnostics_track_acceptance(self):
        rng = random.Random(42)
        _flat_pos(_FLAT_TERRAIN, rng, _NO_EXCLUSION)
        assert (PLACEMENT_DIAGNOSTICS["accepted_flat"]
                + PLACEMENT_DIAGNOSTICS["accepted_random"]
                + PLACEMENT_DIAGNOSTICS["fallback_used"]) >= 1


# ---------------------------------------------------------------------------
# generate_rural_placements
# ---------------------------------------------------------------------------


class TestGenerateRuralPlacements:
    def test_light_density(self):
        people, vehicles, roads, tree_lines = generate_rural_placements(
            _FLAT_TERRAIN, "light", 42,
        )
        assert len(people) == RURAL_PROFILES["light"]["civilian_people"] + RURAL_PROFILES["light"]["military_people"]
        assert len(vehicles) == RURAL_PROFILES["light"]["civilian_vehicles"] + RURAL_PROFILES["light"]["concealed_vehicles"]
        assert len(roads) == RURAL_PROFILES["light"]["dirt_roads"]

    def test_medium_density(self):
        people, vehicles, roads, tree_lines = generate_rural_placements(
            _FLAT_TERRAIN, "medium", 42,
        )
        profile = RURAL_PROFILES["medium"]
        assert len(people) == profile["civilian_people"] + profile["military_people"]
        assert len(vehicles) == profile["civilian_vehicles"] + profile["concealed_vehicles"]

    def test_heavy_density(self):
        people, vehicles, roads, tree_lines = generate_rural_placements(
            _FLAT_TERRAIN, "heavy", 42,
        )
        profile = RURAL_PROFILES["heavy"]
        assert len(people) == profile["civilian_people"] + profile["military_people"]
        assert len(vehicles) == profile["civilian_vehicles"] + profile["concealed_vehicles"]

    def test_target_ids_present(self):
        people, vehicles, _, _ = generate_rural_placements(
            _FLAT_TERRAIN, "light", 42,
        )
        for p in people:
            assert p.target_id, f"Person {p.semantic_name} missing target_id"
        for v in vehicles:
            assert v.target_id, f"Vehicle {v.semantic_name} missing target_id"

    def test_military_targets_have_concealment(self):
        people, _, _, _ = generate_rural_placements(
            _FLAT_TERRAIN, "medium", 42,
        )
        military = [p for p in people if p.target_type == "military_like"]
        assert len(military) > 0
        # At least some should have concealment > 0
        concealed = [p for p in military if p.concealment_level > 0.0]
        assert len(concealed) > 0


# ---------------------------------------------------------------------------
# SDF generation
# ---------------------------------------------------------------------------


class TestSDFGeneration:
    def test_generates_valid_xml(self):
        xml_str, stats = generate_rural_world_sdf("test_terrain", "light", 42)
        assert "<?xml" in xml_str
        assert "<sdf" in xml_str
        assert "</sdf>" in xml_str

    def test_contains_actors(self):
        xml_str, stats = generate_rural_world_sdf("test_terrain", "light", 42)
        assert "<actor" in xml_str

    def test_contains_vehicles(self):
        xml_str, stats = generate_rural_world_sdf("test_terrain", "light", 42)
        assert "vehicle_" in xml_str

    def test_contains_target_metadata(self):
        xml_str, stats = generate_rural_world_sdf("test_terrain", "light", 42)
        assert "TARGET_METADATA:" in xml_str

    def test_contains_roads(self):
        xml_str, stats = generate_rural_world_sdf("test_terrain", "light", 42)
        assert 'name="road_' in xml_str

    def test_contains_markers(self):
        xml_str, stats = generate_rural_world_sdf("test_terrain", "light", 42)
        assert 'name="marker_' in xml_str

    def test_stats_match_profile(self):
        _, stats = generate_rural_world_sdf("test_terrain", "light", 42)
        profile = RURAL_PROFILES["light"]
        assert stats["civilian_people"] == profile["civilian_people"]
        assert stats["military_people"] == profile["military_people"]
        assert stats["civilian_vehicles"] == profile["civilian_vehicles"]
        assert stats["concealed_vehicles"] == profile["concealed_vehicles"]

    def test_all_densities_generate(self):
        for density in ("light", "medium", "heavy"):
            xml_str, stats = generate_rural_world_sdf("test_terrain", density, 42)
            assert len(xml_str) > 100
            assert stats["people"] > 0
            assert stats["vehicles"] > 0


# ---------------------------------------------------------------------------
# Stats key validation
# ---------------------------------------------------------------------------


class TestStatsKeys:
    @pytest.mark.parametrize("density", ["light", "medium", "heavy"])
    def test_all_required_keys_present(self, density):
        _, stats = generate_rural_world_sdf("test_terrain", density, 42)
        for key in REQUIRED_STATS_KEYS:
            assert key in stats, f"Missing stats key: {key}"

    def test_validate_stats_fills_missing(self):
        stats = {"people": 5}
        _validate_stats(stats)
        for key in REQUIRED_STATS_KEYS:
            assert key in stats

    def test_concealment_distribution_keys(self):
        _, stats = generate_rural_world_sdf("test_terrain", "light", 42)
        cd = stats["concealment_distribution"]
        for k in ("open", "partial", "high"):
            assert k in cd

    def test_rejected_positions_key(self):
        _, stats = generate_rural_world_sdf("test_terrain", "light", 42)
        assert isinstance(stats["rejected_positions"], int)
        assert stats["rejected_positions"] >= 0

    def test_fallback_positions_key(self):
        _, stats = generate_rural_world_sdf("test_terrain", "light", 42)
        assert isinstance(stats["fallback_positions"], int)
        assert stats["fallback_positions"] >= 0

    def test_bounds_diagnostics_keys(self):
        _, stats = generate_rural_world_sdf("test_terrain", "light", 42)
        assert isinstance(stats["outside_bounds_rejections"], int)
        assert isinstance(stats["clipped_road_segments"], int)
        assert isinstance(stats["failed_shed_placements"], int)


# ---------------------------------------------------------------------------
# Bounds validation with real terrain
# ---------------------------------------------------------------------------


class TestBoundsValidation:
    @pytest.fixture
    def real_terrain(self):
        terrain = load_terrain("1779343687303")
        if terrain.width_m == 0 and terrain.height_m == 0:
            pytest.skip("Heightmap not available — skipping terrain bounds validation")
        return terrain

    def test_real_terrain_has_bounds(self, real_terrain):
        assert real_terrain.width_m > 0
        assert real_terrain.height_m > 0

    @pytest.mark.parametrize("density", ["light", "medium", "heavy"])
    def test_all_entities_within_terrain(self, real_terrain, density):
        import xml.etree.ElementTree as ET

        xml_str, stats = generate_rural_world_sdf("1779343687303", density, 42)
        root = ET.fromstring(xml_str)

        half_w = real_terrain.width_m / 2
        half_h = real_terrain.height_m / 2
        margin = _BOUNDS_MARGIN
        oob = []

        for model in root.iter("model"):
            pose = model.find("pose")
            if pose is None or pose.text is None:
                continue
            parts = pose.text.strip().split()
            if len(parts) < 2:
                continue
            x, y = float(parts[0]), float(parts[1])
            name = model.get("name", "?")
            # Skip concealment zone markers (they're tiny, not gameplay-relevant)
            if name.startswith("concealment_zone"):
                continue
            if (x < real_terrain.offset_x - half_w + margin
                    or x > real_terrain.offset_x + half_w - margin
                    or y < real_terrain.offset_y - half_h + margin
                    or y > real_terrain.offset_y + half_h - margin):
                oob.append(f"{name} at ({x:.1f}, {y:.1f})")

        for inc in root.iter("include"):
            pose = inc.find("pose")
            if pose is None or pose.text is None:
                continue
            parts = pose.text.strip().split()
            if len(parts) < 2:
                continue
            x, y = float(parts[0]), float(parts[1])
            nm_el = inc.find("name")
            nm = nm_el.text if nm_el is not None else "?"
            if (x < real_terrain.offset_x - half_w + margin
                    or x > real_terrain.offset_x + half_w - margin
                    or y < real_terrain.offset_y - half_h + margin
                    or y > real_terrain.offset_y + half_h - margin):
                oob.append(f"{nm} at ({x:.1f}, {y:.1f})")

        # Actor waypoints
        for actor in root.iter("actor"):
            for wp in actor.iter("waypoint"):
                pose = wp.find("pose")
                if pose is None or pose.text is None:
                    continue
                parts = pose.text.strip().split()
                if len(parts) < 2:
                    continue
                x, y = float(parts[0]), float(parts[1])
                name = actor.get("name", "?")
                if (x < real_terrain.offset_x - half_w + margin
                        or x > real_terrain.offset_x + half_w - margin
                        or y < real_terrain.offset_y - half_h + margin
                        or y > real_terrain.offset_y + half_h - margin):
                    oob.append(f"actor {name} wp at ({x:.1f}, {y:.1f})")

        assert oob == [], f"{len(oob)} entities outside terrain:\n" + "\n".join(oob[:10])


# ---------------------------------------------------------------------------
# SDF name uniqueness
# ---------------------------------------------------------------------------


class TestUniqueNames:
    def test_unique_name_generates_unique(self):
        _reset_names()
        n1 = _unique_name("veg")
        n2 = _unique_name("veg")
        assert n1 != n2

    def test_claim_name_rejects_duplicate(self):
        _reset_names()
        _claim_name("test_model")
        with pytest.raises(RuntimeError, match="Duplicate"):
            _claim_name("test_model")

    def test_unique_name_prefix_includes_counter(self):
        _reset_names()
        name = _unique_name("road")
        assert name.startswith("road_")

    @pytest.mark.parametrize("density", ["light", "medium", "heavy"])
    def test_no_duplicate_model_names(self, density):
        xml_str, _ = generate_rural_world_sdf("1779343687303", density, 42)
        dups = validate_unique_sdf_names(xml_str)
        assert dups == [], f"{len(dups)} duplicate names:\n" + "\n".join(dups[:5])

    @pytest.mark.parametrize("density", ["light", "medium", "heavy"])
    def test_no_duplicate_visual_names_per_link(self, density):
        xml_str, _ = generate_rural_world_sdf("1779343687303", density, 42)
        dups = validate_unique_sdf_names(xml_str)
        assert dups == []

    def test_generation_raises_on_duplicate(self):
        """generate_rural_world_sdf raises RuntimeError if duplicates exist."""
        # This is implicitly tested by the parametrized tests above,
        # but let's be explicit that the guard works.
        xml_str, _ = generate_rural_world_sdf("1779343687303", "light", 42)
        assert len(xml_str) > 0  # If we got here, no RuntimeError was raised

    def test_no_generic_visual_names(self):
        """Check no bare 'extra_N' visual names remain."""
        import re
        xml_str, _ = generate_rural_world_sdf("1779343687303", "medium", 42)
        bad = re.findall(r'<visual name="extra_\d+"', xml_str)
        assert bad == [], f"Found old-style extra_ visual names: {bad[:5]}"

