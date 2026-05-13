"""Tests for PX4 SITL config schema."""

from pathlib import Path

from sentinel.config.loader import load_config


def test_load_sim_px4_config():
    cfg = load_config(Path("configs/sim_px4.yaml"))
    assert cfg.system.mode == "SIMULATION_MODE"
    assert cfg.mavlink.backend == "mavsdk"
    assert cfg.mavlink.allow_real_backend is True
    assert cfg.mavlink.allow_arm is True
    assert cfg.mavlink.allow_takeoff is True
    assert cfg.mavlink.connect_timeout_s == 20
    assert cfg.mavlink.default_waypoint_altitude_m == 50
    assert cfg.mavlink.default_speed_mps == 10
    assert cfg.mavlink.connection_url.startswith("udpin://")


def test_sim_config_still_uses_mock():
    cfg = load_config(Path("configs/sim.yaml"))
    assert cfg.mavlink.backend == "mock"
    assert cfg.mavlink.allow_arm is False
    assert cfg.mavlink.allow_takeoff is False
    assert cfg.mavlink.allow_real_backend is False


def test_mavlink_config_defaults():
    from sentinel.config.schema import MavlinkConfig

    cfg = MavlinkConfig()
    assert cfg.connect_timeout_s == 15
    assert cfg.default_waypoint_altitude_m == 50
    assert cfg.default_speed_mps == 10
    assert cfg.backend == "mock"
    assert cfg.connection_url.startswith("udpin://")
