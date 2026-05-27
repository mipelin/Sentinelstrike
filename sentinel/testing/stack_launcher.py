"""Simulation stack launcher — manages PX4+Gazebo lifecycle."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


class SimStackLauncher:
    """Launch and stop the Sentinel simulation stack (PX4 + Gazebo)."""

    WORLD_ALIASES = {
        "dynamic_v3": "1779343687303_isr_rural_dynamic_v3",
        "simple_single_person": "1779343687303_isr_rural_simple_single_person",
        "simple_single_vehicle": "1779343687303_isr_rural_simple_single_vehicle",
        "simple_person_vehicle_separated": "1779343687303_isr_rural_simple_person_vehicle_separated",
        "isr_dynamic_v3": "1779343687303_isr_rural_dynamic_v3",
        "1779343687303_isr_rural_dynamic_v3": "1779343687303_isr_rural_dynamic_v3",
        "1779343687303_isr_rural_simple_single_person": "1779343687303_isr_rural_simple_single_person",
        "1779343687303_isr_rural_simple_single_vehicle": "1779343687303_isr_rural_simple_single_vehicle",
        "1779343687303_isr_rural_simple_person_vehicle_separated": "1779343687303_isr_rural_simple_person_vehicle_separated",
        "isr-realistic-lite-v2": "1779343687303_isr_rural_realistic_lite_v2",
        "1779343687303_isr_rural_realistic_lite_v2": "1779343687303_isr_rural_realistic_lite_v2",
        "default": "default",
    }

    def __init__(
        self,
        sentinel_dir: str | Path | None = None,
        px4_dir: str | Path | None = None,
    ) -> None:
        self.sentinel_dir = Path(sentinel_dir or os.environ.get(
            "SENTINEL_DIR", "/home/mipelin/projects/DRON/ons-sentinel-core",
        ))
        self.px4_dir = Path(px4_dir or os.environ.get(
            "PX4_DIR", "/home/mipelin/projects/DRON/PX4-Autopilot",
        ))
        self._started = False

    def normalize_world_name(self, world: str) -> str:
        return self.WORLD_ALIASES.get(world, world)

    def camera_topic(self, world: str, model: str = "x500_mono_cam") -> str:
        world_name = self.normalize_world_name(world)
        return f"/world/{world_name}/model/{model}_0/link/camera_link/sensor/camera/image"

    def build_env(self, spawn_pose: str | None = None) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("GZ_IP", "127.0.0.1")
        env.setdefault("DISPLAY", ":0")
        env.setdefault("XAUTHORITY", "/home/mipelin/.Xauthority")
        # Prepend Sentinel model override path so project-local models shadow PX4 defaults
        sentinel_model_path = str(self.sentinel_dir / "configs" / "gz" / "models")
        existing = env.get("GZ_SIM_RESOURCE_PATH", "")
        if sentinel_model_path not in existing:
            env["GZ_SIM_RESOURCE_PATH"] = f"{sentinel_model_path}:{existing}" if existing else sentinel_model_path
        if spawn_pose:
            env["PX4_GZ_MODEL_POSE"] = spawn_pose
        return env

    def start(
        self,
        world: str = "default",
        spawn_pose: str | None = None,
        no_qgc: bool = True,
    ) -> bool:
        """Launch the simulation stack and wait for readiness.

        Returns True if PX4 is confirmed running after startup.
        """
        cmd = [str(self.sentinel_dir / "scripts/start_sentinel_stack.sh")]
        world = self.normalize_world_name(world)

        # Map world names to script flags
        world_flag_map = {
            "1779343687303_isr_rural_light": "--isr-world",
            "1779343687303_isr_rural_camera_lite": "--isr-lite",
            "1779343687303_isr_rural_realistic_lite": "--isr-realistic-lite",
            "1779343687303_isr_rural_realistic_lite_v2": "--isr-realistic-lite-v2",
            "1779343687303_isr_rural_dynamic_v3": "--isr-dynamic-v3",
            "1779343687303_isr_rural_simple_single_person": "--simple-single-person",
            "1779343687303_isr_rural_simple_single_vehicle": "--simple-single-vehicle",
            "1779343687303_isr_rural_simple_person_vehicle_separated": "--simple-person-vehicle-separated",
        }
        if world != "default":
            flag = world_flag_map.get(world)
            if flag:
                cmd.append(flag)
            else:
                # Unknown world — pass as-is and hope the script handles it
                cmd.append(f"--world={world}")

        if no_qgc:
            cmd.append("--no-qgc")

        env = self.build_env(spawn_pose=spawn_pose)

        # Verify PX4 camera model pitch is correct before launching
        verify_script = self.sentinel_dir / "scripts" / "verify_px4_camera_pitch.py"
        if verify_script.exists():
            subprocess.run([sys.executable, str(verify_script)], cwd=self.sentinel_dir)

        print(f"[STACK] Starting stack: {' '.join(cmd)}")
        if spawn_pose:
            print(f"[STACK] Spawn pose: {spawn_pose}")

        result = subprocess.run(cmd, cwd=self.sentinel_dir, env=env)
        if result.returncode != 0:
            print(f"[STACK] Start failed with exit code {result.returncode}")
            return False

        self._started = True
        print("[STACK] Stack started successfully")
        return True

    def stop(self) -> None:
        """Kill all simulation processes cleanly."""
        cmd = [str(self.sentinel_dir / "scripts/stop_sentinel_stack.sh")]
        print(f"[STACK] Stopping stack...")
        subprocess.run(cmd, cwd=self.sentinel_dir)
        self._started = False
        # Give processes time to die
        time.sleep(2)

    def wait_for_clock(self, world: str, timeout_s: float = 20.0) -> bool:
        """Wait until Gazebo world stats confirm the sim is advancing."""
        world = self.normalize_world_name(world)
        deadline = time.monotonic() + timeout_s
        cmd = [
            "python3",
            "-m",
            "apps.tools.probe_gazebo_topic_rate",
            "--stats",
            "--world",
            world,
            "--seconds",
            "3",
        ]
        while time.monotonic() < deadline:
            result = subprocess.run(
                cmd,
                cwd=self.sentinel_dir,
                capture_output=True,
                text=True,
            )
            output = (result.stdout or "") + (result.stderr or "")
            if "OK: simulation running" in output:
                print("[STACK] /clock validation passed")
                return True
            time.sleep(2)
        print("[STACK] /clock validation failed")
        return False

    def validate_camera(
        self,
        topic: str,
        timeout_s: float = 10.0,
        save_frame: str | None = None,
    ) -> bool:
        """Validate camera frames publish and optionally save one frame."""
        cmd = [
            "python3",
            "-m",
            "apps.tools.run_gazebo_camera_probe",
            "--topic",
            topic,
            "--timeout",
            str(timeout_s),
        ]
        if save_frame:
            cmd.extend(["--save-frame", save_frame])
        result = subprocess.run(cmd, cwd=self.sentinel_dir)
        if result.returncode == 0:
            print(f"[STACK] Camera validation passed: {topic}")
            return True
        print(f"[STACK] Camera validation failed: {topic}")
        return False

    def is_running(self) -> bool:
        """Check if PX4 process is alive."""
        result = subprocess.run(
            ["pgrep", "-f", "px4"],
            capture_output=True,
        )
        return result.returncode == 0
