"""Tests for config loading."""

from pathlib import Path

from sentinel.config.loader import load_config


def test_load_sim_config():
    cfg = load_config(Path("configs/sim.yaml"))
    assert cfg.system.mode == "SIMULATION_MODE"
    assert cfg.system.vehicle_id == "uav_001"
    assert cfg.recorder.enabled is True
    assert cfg.simulation.link_loss_enabled is False
