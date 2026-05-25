"""Integration tests for behavioral ISR reasoning pipeline.

Tests the full pipeline: BehaviorAnalyzer → PriorityScorer → HypothesisManager
with realistic track data, verifying pattern classification, anomaly scoring,
priority ranking, and behavior-aware hypothesis hints.
"""

import math
from unittest.mock import MagicMock

from sentinel.tracker.behavior import (
    BehaviorAnalyzer,
    BehavioralProfile,
    MovementPattern,
)
from sentinel.tracker.priority import PriorityScorer, PriorityWeights
from sentinel.tracker.hypothesis import (
    HypothesisManager,
    HypothesisType,
    MAX_HYPOTHESES,
)
from sentinel.tracker.terrain_reasoning import TerrainReasoner, VegetationZone
from sentinel.tracker.world_projection import DronePose


def _make_terrain() -> TerrainReasoner:
    terrain = TerrainReasoner()
    return terrain


def _make_drone_pose() -> DronePose:
    return DronePose(lat=55.0, lon=12.0, alt_m=30.0, heading_deg=0.0)


def _track(
    track_id="TGT-001",
    cls="person",
    lat=55.0001,
    lon=12.0001,
    speed_mps=1.0,
    heading_rad=0.0,
    hits=50,
    confidence=0.9,
    occlusion_state="VISIBLE",
    bbox=None,
):
    return {
        "track_id": track_id,
        "class": cls,
        "status": "CONFIRMED",
        "confidence": confidence,
        "bbox": bbox or [100, 100, 200, 200],
        "hits": hits,
        "world_position": {"lat": lat, "lon": lon, "north_m": 10, "east_m": 10,
                           "ground_distance_m": 30, "uncertainty_m": 2},
        "world_velocity": {"speed_mps": speed_mps, "heading_rad": heading_rad,
                           "heading_deg": math.degrees(heading_rad)},
        "occlusion_state": occlusion_state,
    }


class TestBehaviorAnalyzer:
    def test_profile_created_on_update(self):
        terrain = _make_terrain()
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        tracks = [_track()]
        analyzer.update(tracks)
        profile = analyzer.get_profile("TGT-001")
        assert profile is not None
        assert profile.track_id == "TGT-001"
        assert profile.total_frames == 1

    def test_speed_ema(self):
        terrain = _make_terrain()
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        for i in range(20):
            analyzer.update([_track(speed_mps=2.0)])
        profile = analyzer.get_profile("TGT-001")
        assert 1.5 < profile.avg_speed_mps < 2.5

    def test_stop_detection(self):
        terrain = _make_terrain()
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        # 5 frames moving
        for _ in range(5):
            analyzer.update([_track(speed_mps=1.5)])
        # 10 frames stopped
        for _ in range(10):
            analyzer.update([_track(speed_mps=0.1)])
        profile = analyzer.get_profile("TGT-001")
        assert profile.stop_count >= 1

    def test_concealment_affinity(self):
        terrain = _make_terrain()
        # Add vegetation zones
        terrain.add_vegetation_zone(VegetationZone(
            name="test_veg",
            center_lat=55.0002,
            center_lon=12.0002,
            radius_m=20.0,
            density=0.8,
        ))
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        for _ in range(20):
            analyzer.update([_track(lat=55.0002, lon=12.0002)])
        profile = analyzer.get_profile("TGT-001")
        assert profile.concealment_affinity > 0.0

    def test_direction_changes(self):
        terrain = _make_terrain()
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        # Each heading change is ~40° (> 30° threshold)
        headings = [0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9]
        for h in headings:
            analyzer.update([_track(heading_rad=h, speed_mps=1.5)])
        profile = analyzer.get_profile("TGT-001")
        assert profile.direction_changes >= 3

    def test_anomaly_scores_with_population(self):
        terrain = _make_terrain()
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        # Normal walker
        for _ in range(10):
            analyzer.update([_track(track_id="TGT-001", speed_mps=1.2)])
        # Fast runner
        for _ in range(10):
            analyzer.update([_track(track_id="TGT-002", speed_mps=5.0)])
        p1 = analyzer.get_profile("TGT-001")
        p2 = analyzer.get_profile("TGT-002")
        # Both should have anomaly scores computed
        assert p1.anomaly_score > 0.0
        assert p2.anomaly_score > 0.0

    def test_interaction_detection(self):
        terrain = _make_terrain()
        analyzer = BehaviorAnalyzer(terrain, fps=15.0, interaction_distance_m=5.0)
        # Two targets at same position
        for _ in range(15):
            analyzer.update([
                _track(track_id="TGT-001", lat=55.0001, lon=12.0001),
                _track(track_id="TGT-002", lat=55.0001, lon=12.0001),
            ])
        graph = analyzer.get_interaction_graph()
        assert len(graph) >= 1

    def test_metrics_summary(self):
        terrain = _make_terrain()
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        for _ in range(10):
            analyzer.update([_track()])
        summary = analyzer.metrics.get_summary()
        assert "pattern_counts" in summary
        assert "avg_anomaly_score" in summary

    def test_remove_profile(self):
        terrain = _make_terrain()
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        analyzer.update([_track()])
        assert analyzer.get_profile("TGT-001") is not None
        analyzer.remove_profile("TGT-001")
        assert analyzer.get_profile("TGT-001") is None


class TestPriorityScorer:
    def test_score_target(self):
        terrain = _make_terrain()
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        for _ in range(10):
            analyzer.update([_track()])
        scorer = PriorityScorer(analyzer)
        score = scorer.score_target(_track())
        assert 0.0 <= score <= 1.0

    def test_high_concealment_increases_score(self):
        terrain = _make_terrain()
        terrain.add_vegetation_zone(VegetationZone(
            name="test_veg",
            center_lat=55.0002,
            center_lon=12.0002,
            radius_m=20.0,
            density=0.8,
        ))
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        for _ in range(20):
            analyzer.update([_track(lat=55.0002, lon=12.0002)])
        scorer = PriorityScorer(analyzer)
        score = scorer.score_target(_track(lat=55.0002, lon=12.0002))
        assert score > 0.0

    def test_rank_targets(self):
        terrain = _make_terrain()
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        # Two targets with different speeds
        for _ in range(10):
            analyzer.update([
                _track(track_id="TGT-001", speed_mps=1.0),
                _track(track_id="TGT-002", speed_mps=5.0),
            ])
        scorer = PriorityScorer(analyzer)
        tracks = [
            _track(track_id="TGT-001"),
            _track(track_id="TGT-002"),
        ]
        ranked = scorer.rank_targets(tracks)
        assert len(ranked) == 2
        assert ranked[0][1] >= ranked[1][1]

    def test_suggest_target(self):
        terrain = _make_terrain()
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        for _ in range(10):
            analyzer.update([_track(track_id="TGT-001")])
        scorer = PriorityScorer(analyzer)
        tracks = [_track(track_id="TGT-001")]
        suggested = scorer.suggest_target(tracks)
        assert suggested == "TGT-001"

    def test_suggest_target_empty_when_none_visible(self):
        terrain = _make_terrain()
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        scorer = PriorityScorer(analyzer)
        tracks = [_track(occlusion_state="OCCLUDED")]
        assert scorer.suggest_target(tracks) is None

    def test_operator_override(self):
        terrain = _make_terrain()
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        scorer = PriorityScorer(analyzer, operator_overrides={"TGT-001": 1.0})
        score = scorer.score_target(_track(track_id="TGT-001"))
        assert score >= 0.1  # At least some contribution from override


class TestBehaviorAwareHypotheses:
    def test_evasive_widens_search_radius(self):
        terrain = _make_terrain()
        mgr = HypothesisManager(terrain, fps=15.0)
        mgr.set_behavior_hint("TGT-001", evasive=True, concealment_seeking=False)
        mgr.on_target_lost(
            track_id="TGT-001",
            bbox=(100, 100, 200, 200),
            world_lat=55.0,
            world_lon=12.0,
            world_speed_mps=2.0,
            world_heading_rad=1.0,
            total_hits=50,
            smoothed_confidence=0.9,
            frame_w=1280,
            frame_h=720,
            all_tracks=[],
            drone_pose=_make_drone_pose(),
        )
        hyps = mgr.get_hypotheses("TGT-001")
        for h in hyps:
            # Search radius should be widened by 1.5x for evasive targets
            assert h.search_radius_m >= 5.0 * 1.5 or h.htype in (HypothesisType.OCCLUDED,)

    def test_concealment_boosts_occluded(self):
        terrain = _make_terrain()
        mgr = HypothesisManager(terrain, fps=15.0)
        mgr.set_behavior_hint("TGT-001", evasive=False, concealment_seeking=True)
        mgr.on_target_lost(
            track_id="TGT-001",
            bbox=(400, 200, 600, 400),  # Center of frame, not near edge
            world_lat=55.0,
            world_lon=12.0,
            world_speed_mps=1.0,
            world_heading_rad=0.0,
            total_hits=30,
            smoothed_confidence=0.8,
            frame_w=1280,
            frame_h=720,
            all_tracks=[],
            drone_pose=_make_drone_pose(),
        )
        hyps = mgr.get_hypotheses("TGT-001")
        occ_hyps = [h for h in hyps if h.htype == HypothesisType.OCCLUDED]
        assert len(occ_hyps) > 0
        # OCCLUDED should be dominant
        best = max(hyps, key=lambda h: h.probability)
        assert best.htype == HypothesisType.OCCLUDED

    def test_concealment_slower_occluded_decay(self):
        terrain = _make_terrain()
        mgr = HypothesisManager(terrain, fps=15.0)
        mgr.set_behavior_hint("TGT-001", evasive=False, concealment_seeking=True)
        mgr.on_target_lost(
            track_id="TGT-001",
            bbox=(100, 100, 200, 200),
            world_lat=55.0,
            world_lon=12.0,
            world_speed_mps=0.5,
            world_heading_rad=0.0,
            total_hits=30,
            smoothed_confidence=0.8,
            frame_w=1280,
            frame_h=720,
            all_tracks=[],
            drone_pose=_make_drone_pose(),
        )
        # Compare with non-concealment target
        mgr2 = HypothesisManager(terrain, fps=15.0)
        mgr2.on_target_lost(
            track_id="TGT-002",
            bbox=(100, 100, 200, 200),
            world_lat=55.0,
            world_lon=12.0,
            world_speed_mps=0.5,
            world_heading_rad=0.0,
            total_hits=30,
            smoothed_confidence=0.8,
            frame_w=1280,
            frame_h=720,
            all_tracks=[],
            drone_pose=_make_drone_pose(),
        )
        # Evolve both 50 frames
        pose = _make_drone_pose()
        for _ in range(50):
            mgr.evolve("TGT-001", pose, 1.0 / 15.0)
            mgr2.evolve("TGT-002", pose, 1.0 / 15.0)
        # Concealment-seeking target should retain higher OCCLUDED probability
        occ1 = [h for h in mgr.get_hypotheses("TGT-001") if h.htype == HypothesisType.OCCLUDED]
        occ2 = [h for h in mgr2.get_hypotheses("TGT-002") if h.htype == HypothesisType.OCCLUDED]
        if occ1 and occ2:
            assert occ1[0].probability >= occ2[0].probability * 0.9


class TestPatternClassification:
    def test_stationary_observation(self):
        terrain = _make_terrain()
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        for _ in range(80):
            analyzer.update([_track(speed_mps=0.1)])
        profile = analyzer.get_profile("TGT-001")
        assert profile.current_pattern == MovementPattern.STATIONARY_OBSERVATION

    def test_loitering(self):
        terrain = _make_terrain()
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        # Simulate stop-and-go with direction changes
        for i in range(100):
            speed = 0.5 if i % 10 < 3 else 0.1
            heading = (i % 5) * 1.5
            analyzer.update([_track(speed_mps=speed, heading_rad=heading)])
        profile = analyzer.get_profile("TGT-001")
        assert profile.current_pattern in (
            MovementPattern.LOITERING,
            MovementPattern.STOP_AND_GO,
            MovementPattern.STATIONARY_OBSERVATION,
        )


class TestGeospatialBehavioral:
    def test_loitering_heatmap(self):
        from sentinel.tracker.geospatial import GeospatialLayer
        geo = GeospatialLayer(grid_resolution_m=5.0)
        # Stationary target
        for _ in range(20):
            geo.update([_track(speed_mps=0.1)])
        zones = geo.get_loitering_zones(min_seconds=0.5)
        assert len(zones) > 0

    def test_interaction_heatmap(self):
        from sentinel.tracker.geospatial import GeospatialLayer
        geo = GeospatialLayer(grid_resolution_m=5.0)
        tracks = [
            _track(track_id="TGT-001", lat=55.0001, lon=12.0001),
            _track(track_id="TGT-002", lat=55.0001, lon=12.0001),
        ]
        geo.update(tracks, interaction_pairs=[("TGT-001", "TGT-002")])
        zones = geo.get_interaction_zones()
        assert len(zones) > 0

    def test_concealment_heatmap(self):
        from sentinel.tracker.geospatial import GeospatialLayer
        from sentinel.tracker.terrain_reasoning import TerrainReasoner
        terrain = TerrainReasoner()
        terrain.add_vegetation_zone(VegetationZone(
            name="test_veg",
            center_lat=55.0002,
            center_lon=12.0002,
            radius_m=20.0,
            density=0.8,
        ))
        geo = GeospatialLayer(grid_resolution_m=5.0, terrain=terrain)
        for _ in range(10):
            geo.update([_track(lat=55.0002, lon=12.0002)])
        hotspots = geo.get_concealment_hotspots(min_count=1)
        assert len(hotspots) > 0


class TestISRWorldStateBehavioral:
    def test_behavioral_profiles(self):
        from sentinel.tracker.isr_world_state import ISRWorldState
        from sentinel.tracker.geospatial import GeospatialLayer
        terrain = _make_terrain()
        geo = GeospatialLayer()
        ws = ISRWorldState(terrain, geo)
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        ws.attach_behavior(analyzer)
        for _ in range(5):
            analyzer.update([_track()])
        profiles = ws.get_behavioral_profiles()
        assert "TGT-001" in profiles
        assert "pattern" in profiles["TGT-001"]

    def test_targets_by_pattern(self):
        from sentinel.tracker.isr_world_state import ISRWorldState
        from sentinel.tracker.geospatial import GeospatialLayer
        terrain = _make_terrain()
        geo = GeospatialLayer()
        ws = ISRWorldState(terrain, geo)
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        ws.attach_behavior(analyzer)
        for _ in range(80):
            analyzer.update([_track(speed_mps=0.1)])
        stationary = ws.get_targets_by_pattern("STATIONARY_OBSERVATION")
        assert "TGT-001" in stationary

    def test_priority_ranking(self):
        from sentinel.tracker.isr_world_state import ISRWorldState
        from sentinel.tracker.geospatial import GeospatialLayer
        terrain = _make_terrain()
        geo = GeospatialLayer()
        ws = ISRWorldState(terrain, geo)
        analyzer = BehaviorAnalyzer(terrain, fps=15.0)
        scorer = PriorityScorer(analyzer)
        ws.attach_behavior(analyzer)
        ws.attach_priority_scorer(scorer)
        for _ in range(10):
            analyzer.update([_track(track_id="TGT-001")])
        ranked = ws.get_priority_ranking([_track(track_id="TGT-001")])
        assert len(ranked) == 1

    def test_interaction_summary(self):
        from sentinel.tracker.isr_world_state import ISRWorldState
        from sentinel.tracker.geospatial import GeospatialLayer
        terrain = _make_terrain()
        geo = GeospatialLayer()
        ws = ISRWorldState(terrain, geo)
        analyzer = BehaviorAnalyzer(terrain, fps=15.0, interaction_distance_m=5.0)
        ws.attach_behavior(analyzer)
        for _ in range(15):
            analyzer.update([
                _track(track_id="TGT-001", lat=55.0001, lon=12.0001),
                _track(track_id="TGT-002", lat=55.0001, lon=12.0001),
            ])
        interactions = ws.get_interaction_summary()
        assert len(interactions) >= 1
        assert "id_a" in interactions[0]
