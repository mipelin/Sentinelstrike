"""Auto-pilot helpers for simulation tests — arm, takeoff, land."""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _ensure_local_venv_site_packages() -> None:
    """Make repo-local .venv packages importable when launched via system python."""
    repo_root = Path(__file__).resolve().parents[2]
    lib_root = repo_root / ".venv" / "lib"
    if not lib_root.exists():
        return
    for py_dir in sorted(lib_root.glob("python*/site-packages")):
        site_path = str(py_dir)
        if site_path not in sys.path:
            sys.path.insert(0, site_path)


class AutoPilot:
    """MAVSDK-based auto-pilot for SITL simulation.

    Creates a temporary MAVSDK connection to perform arm/takeoff/land
    operations without interfering with the FlightBridgeWorker.
    """

    def __init__(self, url: str = "udpin://0.0.0.0:14540") -> None:
        self.url = url

    def _get_backend(self):
        """Lazy-import and instantiate MavsdkBackend."""
        _ensure_local_venv_site_packages()
        from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend

        return MavsdkBackend(connection_url=self.url)

    def connect(self):
        """Connect and return the backend instance, or None on failure."""
        try:
            backend = self._get_backend()
        except ImportError as exc:
            print(f"[AUTO-PILOT] ERROR: mavsdk not installed ({exc})")
            return None
        result = backend.connect()
        if not result.success:
            print(f"[AUTO-PILOT] ERROR: connection failed: {result.message}")
            return None
        return backend

    def wait_until_ready(self, timeout_s: float = 30.0) -> bool:
        """Wait until telemetry is available from SITL."""
        backend = self.connect()
        if backend is None:
            return False
        try:
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout_s:
                try:
                    tele = backend.get_telemetry()
                    if tele.connected and tele.position is not None:
                        print("[AUTO-PILOT] MAVLink ready")
                        return True
                except Exception as exc:
                    print(f"[AUTO-PILOT] Telemetry read error: {exc}")
                time.sleep(0.5)
            print(f"[AUTO-PILOT] ERROR: timeout waiting for MAVLink telemetry ({timeout_s}s)")
            return False
        finally:
            backend.close()

    def wait_until_altitude(self, altitude_m: float, timeout_s: float = 60.0) -> bool:
        """Wait until vehicle reaches a relative altitude threshold."""
        backend = self.connect()
        if backend is None:
            return False
        threshold = altitude_m * 0.9
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            try:
                tele = backend.get_telemetry()
                alt = tele.position.alt_m if tele.position else 0.0
                if alt >= threshold:
                    print(f"[AUTO-PILOT] Altitude reached: {alt:.1f}m")
                    return True
            except Exception as exc:
                print(f"[AUTO-PILOT] Telemetry read error: {exc}")
            time.sleep(0.5)
        print(f"[AUTO-PILOT] ERROR: timeout waiting for altitude >= {threshold:.1f}m")
        return False

    def arm_and_takeoff(self, altitude_m: float = 15.0, timeout_s: float = 60.0) -> bool:
        """Arm and takeoff. Wait until altitude reached.

        Returns True if airborne at target altitude.
        """
        try:
            backend = self._get_backend()
        except ImportError as exc:
            print(f"[AUTO-PILOT] ERROR: mavsdk not installed ({exc})")
            return False

        print(f"[AUTO-PILOT] Connecting to {self.url}...")
        result = backend.connect()
        if not result.success:
            print(f"[AUTO-PILOT] ERROR: connection failed: {result.message}")
            return False

        try:
            print("[AUTO-PILOT] Connected. Arming...")
            result = backend.arm()
            if not result.success:
                print(f"[AUTO-PILOT] ERROR: arm failed: {result.message}")
                return False
            print("[AUTO-PILOT] Armed.")

            # PX4 SITL: use offboard velocity climb instead of action.takeoff()
            # which times out in simulation
            print("[AUTO-PILOT] Starting offboard mode for climb...")
            result = backend.offboard_start()
            if not result.success:
                print(f"[AUTO-PILOT] ERROR: offboard start failed: {result.message}")
                return False
            print("[AUTO-PILOT] Offboard engaged.")

            print(f"[AUTO-PILOT] Climbing to {altitude_m}m...")
            t0 = time.monotonic()
            last_log = 0.0
            while time.monotonic() - t0 < timeout_s:
                try:
                    tele = backend.get_telemetry()
                    alt = tele.position.alt_m if tele.position else 0.0
                    armed = getattr(tele, 'armed', '?')
                    mode = getattr(tele, 'mode', '?')
                    if time.monotonic() - last_log >= 5.0:
                        print(f"[AUTO-PILOT]  alt={alt:.1f}m armed={armed} mode={mode}")
                        last_log = time.monotonic()
                    if alt >= altitude_m * 0.9:
                        print(f"[AUTO-PILOT] Airborne at {alt:.1f}m. Ready.")
                        return True
                    # Send upward velocity (vz negative = up in body frame)
                    backend.offboard_set_velocity_body(0.0, 0.0, -1.5, 0.0)
                except Exception as exc:
                    print(f"[AUTO-PILOT] Telemetry read error: {exc}")
                time.sleep(0.2)

            print(f"[AUTO-PILOT] ERROR: timeout waiting for altitude ({timeout_s}s)")
            return False
        finally:
            backend.close()

    def land(self, timeout_s: float = 30.0) -> bool:
        """Command land. Returns True if command accepted."""
        try:
            backend = self._get_backend()
        except ImportError:
            return False

        result = backend.connect()
        if not result.success:
            return False

        try:
            print("[AUTO-PILOT] Landing...")
            result = backend.land()
            if result.success:
                print("[AUTO-PILOT] Land command accepted")
                # Wait a few seconds for descent to start
                time.sleep(3)
                return True
            print(f"[AUTO-PILOT] Land failed: {result.message}")
            return False
        finally:
            backend.close()

    def hold(self) -> bool:
        """Command hold (Loiter)."""
        try:
            backend = self._get_backend()
        except ImportError:
            return False

        result = backend.connect()
        if not result.success:
            return False

        try:
            result = backend.hold()
            return result.success
        finally:
            backend.close()
