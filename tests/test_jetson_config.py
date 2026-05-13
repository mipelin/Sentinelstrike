"""Tests for Jetson config and RuntimeConfig schema."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sentinel.config.schema import AppConfig, RuntimeConfig


def test_runtime_config_defaults() -> None:
    cfg = RuntimeConfig()
    assert cfg.target_fps == 10.0
    assert cfg.save_frames is False
    assert cfg.save_latest_frame is True
    assert cfg.save_annotated_every_n_frames == 10
    assert cfg.disk_budget_mb == 2048
    assert cfg.watchdog_sensor_stale_s == 5.0
    assert cfg.watchdog_telemetry_stale_s == 10.0
    assert cfg.watchdog_min_fps == 2.0
    assert cfg.watchdog_ram_threshold_pct == 90.0
    assert cfg.watchdog_disk_threshold_pct == 95.0
    assert cfg.frame_timeout_s == 5.0
    assert cfg.mavlink_telemetry_timeout_s == 3.0
    assert cfg.log_rotation_enabled is True
    assert cfg.log_max_size_mb == 512


def test_app_config_includes_runtime() -> None:
    cfg = AppConfig()
    assert hasattr(cfg, "runtime")
    assert isinstance(cfg.runtime, RuntimeConfig)


def test_jetson_yaml_loads() -> None:
    config_path = Path("configs/jetson.yaml")
    if not config_path.exists():
        pytest.skip("configs/jetson.yaml not found")

    raw = yaml.safe_load(config_path.read_text())
    cfg = AppConfig(**raw)
    assert cfg.runtime.save_frames is False
    assert cfg.runtime.save_latest_frame is True
    assert cfg.runtime.target_fps >= 1.0
    assert cfg.perception.device == "cuda"


def test_runtime_config_validation() -> None:
    cfg = RuntimeConfig(target_fps=30.0, save_annotated_every_n_frames=5)
    assert cfg.target_fps == 30.0

    cfg2 = RuntimeConfig(target_fps=10.0, max_frames=None)
    assert cfg2.max_frames is None


def test_runtime_config_disk_budget_minimum() -> None:
    cfg = RuntimeConfig(disk_budget_mb=64)
    assert cfg.disk_budget_mb == 64
