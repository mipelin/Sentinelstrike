"""Tests for shared Pydantic models."""

import pytest
from pydantic import ValidationError

from sentinel.common.types import (
    BoundingBox,
    Detection,
    GeoPoint,
    MissionRequest,
)


class TestGeoPoint:
    def test_valid(self):
        p = GeoPoint(lat=38.0, lon=-8.0)
        assert p.lat == 38.0

    def test_invalid_lat(self):
        with pytest.raises(ValidationError):
            GeoPoint(lat=91, lon=0)

    def test_invalid_lon(self):
        with pytest.raises(ValidationError):
            GeoPoint(lat=0, lon=-181)


class TestBoundingBox:
    def test_valid(self):
        bb = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert bb.x2 == 100

    def test_rejects_x2_le_x1(self):
        with pytest.raises(ValidationError):
            BoundingBox(x1=100, y1=0, x2=50, y2=100)

    def test_rejects_y2_le_y1(self):
        with pytest.raises(ValidationError):
            BoundingBox(x1=0, y1=100, x2=100, y2=50)


class TestDetection:
    def test_valid(self):
        d = Detection(
            frame_id=1,
            timestamp_utc="2025-01-01T00:00:00Z",
            class_name="person",
            confidence=0.95,
            bbox_xyxy=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        )
        assert d.confidence == 0.95

    def test_rejects_confidence_over_1(self):
        with pytest.raises(ValidationError):
            Detection(
                frame_id=1,
                timestamp_utc="2025-01-01T00:00:00Z",
                class_name="person",
                confidence=1.5,
                bbox_xyxy=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            )


class TestMissionRequest:
    def _make_request(self, n_points: int):
        points = [GeoPoint(lat=38 + i * 0.001, lon=-8) for i in range(n_points)]
        return MissionRequest(
            mission_id="test",
            launch_point=GeoPoint(lat=38, lon=-8),
            area_of_interest=points,
        )

    def test_valid_with_3_points(self):
        req = self._make_request(3)
        assert len(req.area_of_interest) == 3

    def test_rejects_under_3_points(self):
        with pytest.raises(ValidationError):
            self._make_request(2)
