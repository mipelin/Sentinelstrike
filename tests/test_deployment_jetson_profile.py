"""Tests for Jetson preflight profiling."""

from __future__ import annotations

from sentinel.deployment.jetson_profile import (
    CheckStatus,
    PreflightCheck,
    PreflightReport,
    _check_camera,
    _check_disk,
    _check_gpu,
    _check_memory,
    _check_yolo_model,
    run_preflight,
)
from sentinel.deployment.resources import ResourceProfile, detect_resources


# --- Unit tests for individual checks ---


def test_check_gpu_returns_preflight_check() -> None:
    profile = detect_resources()
    result = _check_gpu(profile)
    assert isinstance(result, PreflightCheck)
    assert result.name == "gpu_cuda"
    assert isinstance(result.status, CheckStatus)


def test_check_yolo_model_missing_returns_warning() -> None:
    result = _check_yolo_model("nonexistent_model.pt")
    assert isinstance(result, PreflightCheck)
    assert result.name == "yolo_model"
    assert result.status == CheckStatus.WARN


def test_check_yolo_model_real_file() -> None:
    from pathlib import Path

    test_path = Path("/tmp/test_model.pt")
    test_path.write_bytes(b"\x00" * 100)
    try:
        result = _check_yolo_model(str(test_path))
        assert result.status == CheckStatus.PASS
    finally:
        test_path.unlink(missing_ok=True)


def test_check_memory_returns_check() -> None:
    profile = detect_resources()
    result = _check_memory(profile, min_mb=1)
    assert isinstance(result, PreflightCheck)
    assert result.name == "memory"


def test_check_memory_low_threshold() -> None:
    profile = detect_resources()
    result = _check_memory(profile, min_mb=9999999)
    assert result.status == CheckStatus.FAIL


def test_check_disk_returns_check() -> None:
    profile = detect_resources()
    result = _check_disk(profile, budget_mb=1)
    assert isinstance(result, PreflightCheck)
    assert result.name == "disk_space"


def test_check_disk_huge_budget() -> None:
    profile = detect_resources()
    result = _check_disk(profile, budget_mb=99999999)
    assert result.status == CheckStatus.FAIL


def test_check_camera_skips_when_not_requested() -> None:
    result = _check_camera(source_type="file", file_path="nonexistent.mp4")
    assert isinstance(result, PreflightCheck)
    assert result.name == "camera"


# --- PreflightReport integration tests ---


def test_run_preflight_returns_report() -> None:
    report = run_preflight(
        model_path="yolov8n.pt",
        source_type="file",
        file_path="data/videos/demo.mp4",
        disk_budget_mb=2048,
        min_memory_mb=1,
        check_mavlink=False,
        check_camera=False,
        allow_non_jetson=True,
    )
    assert isinstance(report, PreflightReport)
    assert len(report.checks) >= 4
    assert isinstance(report.profile, ResourceProfile)


def test_run_preflight_with_camera_check() -> None:
    report = run_preflight(
        model_path="yolov8n.pt",
        source_type="file",
        file_path="nonexistent.mp4",
        disk_budget_mb=2048,
        min_memory_mb=1,
        check_mavlink=False,
        check_camera=True,
        allow_non_jetson=True,
    )
    camera_checks = [c for c in report.checks if c.name == "camera"]
    assert len(camera_checks) > 0


def test_run_preflight_has_platform_check() -> None:
    report = run_preflight(
        disk_budget_mb=2048,
        min_memory_mb=1,
        check_mavlink=False,
        check_camera=False,
        allow_non_jetson=True,
    )
    platform = [c for c in report.checks if c.name == "platform"]
    assert len(platform) == 1


# --- CheckStatus enum and property tests ---


def test_check_status_enum_values() -> None:
    assert CheckStatus.PASS == "PASS"
    assert CheckStatus.WARN == "WARN"
    assert CheckStatus.SKIP == "SKIP"
    assert CheckStatus.FAIL == "FAIL"


def test_preflight_check_passed_property() -> None:
    assert PreflightCheck(name="t", message="m", status=CheckStatus.PASS).passed is True
    assert PreflightCheck(name="t", message="m", status=CheckStatus.WARN).passed is True
    assert PreflightCheck(name="t", message="m", status=CheckStatus.SKIP).passed is True
    assert PreflightCheck(name="t", message="m", status=CheckStatus.FAIL).passed is False


# --- Strict / non-strict mode tests ---


def test_run_preflight_strict_mode_platform_fail() -> None:
    """On non-Jetson, strict mode without allow_non_jetson produces FAIL."""
    report = run_preflight(strict=True, allow_non_jetson=False, min_memory_mb=1)
    platform = [c for c in report.checks if c.name == "platform"][0]
    if not report.profile.is_jetson:
        assert platform.status == CheckStatus.FAIL
        assert report.passed is False
    else:
        assert platform.status == CheckStatus.PASS


def test_run_preflight_non_strict_platform_warn() -> None:
    """Non-strict mode on non-Jetson produces WARN, report still passes."""
    report = run_preflight(strict=False, allow_non_jetson=False, min_memory_mb=1)
    platform = [c for c in report.checks if c.name == "platform"][0]
    if not report.profile.is_jetson:
        assert platform.status == CheckStatus.WARN
        assert report.passed is True
    else:
        assert platform.status == CheckStatus.PASS


def test_run_preflight_skip_camera() -> None:
    report = run_preflight(skip_camera=True, check_camera=False, min_memory_mb=1, allow_non_jetson=True)
    camera = [c for c in report.checks if c.name == "camera"][0]
    assert camera.status == CheckStatus.SKIP


def test_run_preflight_skip_mavlink() -> None:
    report = run_preflight(skip_mavlink=True, check_mavlink=False, min_memory_mb=1, allow_non_jetson=True)
    mavlink = [c for c in report.checks if c.name == "mavlink"][0]
    assert mavlink.status == CheckStatus.SKIP


def test_run_preflight_non_strict_no_fail_means_passed() -> None:
    report = run_preflight(
        strict=False,
        allow_non_jetson=True,
        skip_camera=True,
        skip_mavlink=True,
        min_memory_mb=1,
    )
    assert report.passed is True
    assert not any(c.status == CheckStatus.FAIL for c in report.checks)


def test_run_preflight_strict_camera_check_elevates_warn_to_fail() -> None:
    """In strict mode, a camera WARN (e.g. can't open source) is elevated to FAIL."""
    report = run_preflight(
        strict=True,
        check_camera=True,
        source_type="file",
        file_path="nonexistent_camera_file.mp4",
        min_memory_mb=1,
        allow_non_jetson=True,
    )
    camera = [c for c in report.checks if c.name == "camera"][0]
    assert camera.status == CheckStatus.FAIL
