"""Tests for pipeline report."""

from sentinel.pipeline.report import write_markdown_report


def _summary():
    return {
        "mission_id": "m1",
        "run_dir": "/tmp/r",
        "planner": {"version": "v1", "waypoint_count": 5, "total_distance_m": 1000.0, "estimated_duration_s": 100.0},
        "perception": {"frame_count": 30, "detection_count": 3},
        "tracking": {"total_tracks": 1, "active_tracks": 0, "lost_tracks": 1, "terminated_tracks": 0},
        "geolocalization": {"observation_count": 1},
        "tak": {"message_count": 5, "sent_count": 5, "failed_count": 0},
        "mavlink": {"command_count": 6, "success_count": 6, "failed_count": 0},
        "artifacts": {"events": "events.jsonl", "report": "report.md"},
    }


def test_report_creates_file(tmp_path):
    out = tmp_path / "report.md"
    write_markdown_report(_summary(), out)
    assert out.exists()


def test_report_contains_title(tmp_path):
    out = tmp_path / "report.md"
    write_markdown_report(_summary(), out)
    content = out.read_text()
    assert "Integrated Mission Report" in content


def test_report_contains_limitations(tmp_path):
    out = tmp_path / "report.md"
    write_markdown_report(_summary(), out)
    content = out.read_text()
    assert "Limitations" in content
    assert "Laboratory/simulation" in content


def test_report_contains_artifacts(tmp_path):
    out = tmp_path / "report.md"
    write_markdown_report(_summary(), out)
    content = out.read_text()
    assert "events.jsonl" in content
    assert "report.md" in content
