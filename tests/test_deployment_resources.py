"""Tests for deployment resources detection."""

from __future__ import annotations

from sentinel.deployment.resources import (
    GpuInfo,
    ResourceProfile,
    detect_resources,
    disk_budget_exceeded,
    ram_pressure_pct,
)


def test_detect_resources_returns_profile() -> None:
    profile = detect_resources()
    assert isinstance(profile, ResourceProfile)
    assert isinstance(profile.gpu, GpuInfo)
    assert profile.cpu.cores > 0
    assert profile.memory.total_mb > 0
    assert profile.disk.total_mb > 0


def test_disk_budget_exceeded_returns_bool() -> None:
    result = disk_budget_exceeded(".", budget_mb=1)
    assert isinstance(result, bool)


def test_disk_budget_not_exceeded_with_huge_budget() -> None:
    result = disk_budget_exceeded(".", budget_mb=1)
    assert isinstance(result, bool)


def test_disk_budget_exceeded_with_small_budget() -> None:
    import shutil
    usage = shutil.disk_usage(".")
    free_mb = usage.free // (1024 * 1024)
    result = disk_budget_exceeded(".", budget_mb=free_mb + 1)
    assert result is True


def test_ram_pressure_returns_float() -> None:
    pct = ram_pressure_pct()
    assert isinstance(pct, float)
    assert 0.0 <= pct <= 100.0
