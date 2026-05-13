"""Jetson preflight profile — validates system readiness before mission start."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .resources import ResourceProfile, detect_resources


class CheckStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    SKIP = "SKIP"
    FAIL = "FAIL"


@dataclass
class PreflightCheck:
    name: str
    message: str
    status: CheckStatus = CheckStatus.PASS
    severity: str = "info"

    @property
    def passed(self) -> bool:
        return self.status in (CheckStatus.PASS, CheckStatus.WARN, CheckStatus.SKIP)


@dataclass
class PreflightReport:
    passed: bool
    checks: list[PreflightCheck]
    profile: ResourceProfile
    recommendations: list[str] = field(default_factory=list)


def _check_gpu(profile: ResourceProfile) -> PreflightCheck:
    if profile.gpu.cuda_available:
        return PreflightCheck(
            name="gpu_cuda",
            message=f"CUDA available: {profile.gpu.name} (CC {profile.gpu.compute_capability})",
            status=CheckStatus.PASS,
            severity="info",
        )
    if profile.is_jetson:
        return PreflightCheck(
            name="gpu_cuda",
            message="Jetson detected but CUDA not available. Install PyTorch for Jetson.",
            status=CheckStatus.FAIL,
            severity="error",
        )
    return PreflightCheck(
        name="gpu_cuda",
        message="No GPU/CUDA detected. YOLO inference will run on CPU (slow).",
        status=CheckStatus.WARN,
        severity="warning",
    )


def _check_yolo_model(model_path: str) -> PreflightCheck:
    p = Path(model_path)
    if p.exists():
        size_mb = p.stat().st_size / (1024 * 1024)
        return PreflightCheck(
            name="yolo_model",
            message=f"Model found: {model_path} ({size_mb:.1f} MB)",
            status=CheckStatus.PASS,
            severity="info",
        )
    return PreflightCheck(
        name="yolo_model",
        message=f"Model not found locally: {model_path}. Ultralytics will auto-download on first use.",
        status=CheckStatus.WARN,
        severity="warning",
    )


def _check_memory(profile: ResourceProfile, min_mb: int = 512) -> PreflightCheck:
    if profile.memory.available_mb >= min_mb:
        return PreflightCheck(
            name="memory",
            message=f"Available memory: {profile.memory.available_mb} MB (of {profile.memory.total_mb} MB)",
            status=CheckStatus.PASS,
            severity="info",
        )
    return PreflightCheck(
        name="memory",
        message=f"Low memory: {profile.memory.available_mb} MB available (min {min_mb} MB)",
        status=CheckStatus.FAIL,
        severity="error",
    )


def _check_disk(profile: ResourceProfile, budget_mb: int) -> PreflightCheck:
    if profile.disk.free_mb >= budget_mb:
        return PreflightCheck(
            name="disk_space",
            message=f"Free disk: {profile.disk.free_mb} MB (budget {budget_mb} MB)",
            status=CheckStatus.PASS,
            severity="info",
        )
    return PreflightCheck(
        name="disk_space",
        message=f"Low disk: {profile.disk.free_mb} MB free (budget {budget_mb} MB)",
        status=CheckStatus.FAIL,
        severity="error",
    )


def _check_camera(source_type: str, rtsp_url: str = "", webcam_index: int = 0, file_path: str = "") -> PreflightCheck:
    import cv2
    cap = None
    try:
        if source_type == "rtsp":
            if not rtsp_url:
                return PreflightCheck(
                    name="camera",
                    message="RTSP URL is empty",
                    status=CheckStatus.FAIL,
                    severity="error",
                )
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        elif source_type == "webcam":
            cap = cv2.VideoCapture(webcam_index)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        elif source_type == "file":
            cap = cv2.VideoCapture(file_path)

        if cap is None or not cap.isOpened():
            return PreflightCheck(
                name="camera",
                message=f"Cannot open {source_type} source",
                status=CheckStatus.WARN,
                severity="warning",
            )

        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            return PreflightCheck(
                name="camera",
                message=f"Camera OK: {source_type}, {w}x{h}",
                status=CheckStatus.PASS,
                severity="info",
            )
        return PreflightCheck(
            name="camera",
            message="Camera opened but cannot read frame",
            status=CheckStatus.WARN,
            severity="warning",
        )
    except Exception as exc:
        return PreflightCheck(
            name="camera",
            message=f"Camera error: {exc}",
            status=CheckStatus.WARN,
            severity="warning",
        )
    finally:
        if cap is not None:
            cap.release()


def _check_mavlink_connection(connection_url: str, timeout_s: int = 5) -> PreflightCheck:
    try:
        import threading
        import time

        from sentinel.mavlink_bridge.mavsdk_backend import MavsdkBackend

        _ = time  # used implicitly by threading.join timeout logic

        result_container: dict = {"success": False, "message": ""}

        def _try_connect():
            try:
                backend = MavsdkBackend(
                    vehicle_id="preflight",
                    connection_url=connection_url,
                    connect_timeout_s=timeout_s,
                )
                r = backend.connect()
                result_container["success"] = r.success
                result_container["message"] = r.message
                backend.close()
            except Exception as exc:
                result_container["message"] = str(exc)

        thread = threading.Thread(target=_try_connect, daemon=True)
        thread.start()
        thread.join(timeout=timeout_s + 3)

        if result_container["success"]:
            return PreflightCheck(
                name="mavlink",
                message=f"MAVLink OK: {connection_url}",
                status=CheckStatus.PASS,
                severity="info",
            )
        msg = result_container.get("message", "timeout")
        return PreflightCheck(
            name="mavlink",
            message=f"MAVLink check failed: {msg}",
            status=CheckStatus.WARN,
            severity="warning",
        )
    except ImportError:
        return PreflightCheck(
            name="mavlink",
            message="MAVSDK not installed (mock/offline mode)",
            status=CheckStatus.WARN,
            severity="warning",
        )
    except Exception as exc:
        return PreflightCheck(
            name="mavlink",
            message=f"MAVLink check error: {exc}",
            status=CheckStatus.WARN,
            severity="warning",
        )


def run_preflight(
    *,
    model_path: str = "yolov8n.pt",
    source_type: str = "file",
    rtsp_url: str = "",
    webcam_index: int = 0,
    file_path: str = "data/videos/demo.mp4",
    mavlink_connection_url: str = "",
    disk_budget_mb: int = 2048,
    min_memory_mb: int = 512,
    check_mavlink: bool = False,
    check_camera: bool = False,
    strict: bool = True,
    allow_non_jetson: bool = False,
    skip_camera: bool = False,
    skip_mavlink: bool = False,
) -> PreflightReport:
    profile = detect_resources()
    checks: list[PreflightCheck] = []

    # Platform check
    if profile.is_jetson:
        platform_status = CheckStatus.PASS
        platform_severity = "info"
    elif strict and not allow_non_jetson:
        platform_status = CheckStatus.FAIL
        platform_severity = "error"
    else:
        platform_status = CheckStatus.WARN
        platform_severity = "warning"
    checks.append(PreflightCheck(
        name="platform",
        message=f"Jetson: {profile.jetson_model}" if profile.is_jetson else f"Standard Linux: {profile.cpu.model}",
        status=platform_status,
        severity=platform_severity,
    ))

    checks.append(_check_gpu(profile))
    checks.append(_check_yolo_model(model_path))
    checks.append(_check_memory(profile, min_mb=min_memory_mb))
    checks.append(_check_disk(profile, budget_mb=disk_budget_mb))

    # Camera check
    if skip_camera:
        checks.append(PreflightCheck(
            name="camera",
            message="Camera check skipped (--skip-camera)",
            status=CheckStatus.SKIP,
            severity="info",
        ))
    elif check_camera:
        cam_check = _check_camera(source_type, rtsp_url, webcam_index, file_path)
        if strict and cam_check.status == CheckStatus.WARN:
            cam_check = PreflightCheck(
                name=cam_check.name,
                message=cam_check.message,
                status=CheckStatus.FAIL,
                severity="error",
            )
        checks.append(cam_check)
    else:
        checks.append(PreflightCheck(
            name="camera",
            message="Camera check skipped (use --check-camera)",
            status=CheckStatus.SKIP,
            severity="info",
        ))

    # MAVLink check
    if skip_mavlink:
        checks.append(PreflightCheck(
            name="mavlink",
            message="MAVLink check skipped (--skip-mavlink)",
            status=CheckStatus.SKIP,
            severity="info",
        ))
    elif check_mavlink and mavlink_connection_url:
        mav_check = _check_mavlink_connection(mavlink_connection_url)
        if strict and mav_check.status == CheckStatus.WARN:
            mav_check = PreflightCheck(
                name=mav_check.name,
                message=mav_check.message,
                status=CheckStatus.FAIL,
                severity="error",
            )
        checks.append(mav_check)
    else:
        checks.append(PreflightCheck(
            name="mavlink",
            message="MAVLink check skipped (use --check-mavlink)",
            status=CheckStatus.SKIP,
            severity="info",
        ))

    all_passed = not any(c.status == CheckStatus.FAIL for c in checks)

    report = PreflightReport(passed=all_passed, checks=checks, profile=profile)

    if profile.memory.percent_used > 85:
        report.recommendations.append("High memory usage — consider closing other processes")
    if profile.disk.percent_used > 90:
        report.recommendations.append("Disk nearly full — expect frame recording to be disabled")
    if not profile.gpu.cuda_available:
        report.recommendations.append("Install PyTorch with CUDA support for GPU inference")
    if profile.thermal.cpu_temp_c and profile.thermal.cpu_temp_c > 80:
        report.recommendations.append(f"CPU temperature high ({profile.thermal.cpu_temp_c:.0f}°C) — check cooling")

    return report
