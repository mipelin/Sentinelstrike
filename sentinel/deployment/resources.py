"""System resource detection for Jetson and general Linux platforms."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GpuInfo:
    available: bool
    name: str
    memory_total_mb: int
    memory_free_mb: int
    cuda_available: bool
    compute_capability: str


@dataclass
class CpuInfo:
    model: str
    cores: int
    freq_mhz: float


@dataclass
class MemoryInfo:
    total_mb: int
    available_mb: int
    used_mb: int
    percent_used: float


@dataclass
class DiskInfo:
    path: str
    total_mb: int
    free_mb: int
    used_mb: int
    percent_used: float


@dataclass
class ThermalInfo:
    cpu_temp_c: float | None
    gpu_temp_c: float | None
    throttle_active: bool


@dataclass
class ResourceProfile:
    is_jetson: bool
    jetson_model: str
    gpu: GpuInfo
    cpu: CpuInfo
    memory: MemoryInfo
    disk: DiskInfo
    thermal: ThermalInfo


def _parse_jetson_model() -> str:
    try:
        model_path = Path("/proc/device-tree/model")
        if model_path.exists():
            return model_path.read_text().strip("\x00").strip()
    except (OSError, PermissionError):
        pass

    try:
        result = os.popen("cat /sys/firmware/devicetree/base/model 2>/dev/null").read().strip()
        if result:
            return result
    except OSError:
        pass

    return ""


def _detect_is_jetson() -> bool:
    model = _parse_jetson_model()
    return "nvidia" in model.lower() and any(
        kw in model.lower() for kw in ("jetson", "orin", "xavier", "nano", "tx2", "tx1")
    )


def _detect_gpu() -> GpuInfo:
    cuda_available = False
    compute = ""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            compute = f"{props.major}.{props.minor}"
    except ImportError:
        pass

    total_mb = 0
    free_mb = 0
    name = "unknown"

    gpu_info_path = "/proc/driver/nvidia/gpus"
    if Path(gpu_info_path).exists():
        name = "NVIDIA GPU"
        try:
            result = os.popen(
                "nvidia-smi --query-gpu=memory.total,memory.free "
                "--format=csv,noheader 2>/dev/null"
            ).read().strip()
            if result:
                parts = result.split(",")
                if len(parts) >= 2:
                    total_mb = int(parts[0].strip().split()[0])
                    free_mb = int(parts[1].strip().split()[0])
        except (OSError, ValueError):
            pass

    if total_mb == 0:
        try:
            # Check if we can read memblock (informational only)
            _ = Path("/sys/kernel/debug/memblock/memory").exists()
        except OSError:
            pass

        jetson_ram_path = Path("/sys/devices/platform/tegra-mc/tegra_mc_stats/meminfo")
        if jetson_ram_path.exists():
            name = "NVIDIA Jetson iGPU"

    return GpuInfo(
        available=cuda_available or _detect_is_jetson(),
        name=name,
        memory_total_mb=total_mb,
        memory_free_mb=free_mb,
        cuda_available=cuda_available,
        compute_capability=compute,
    )


def _detect_cpu() -> CpuInfo:
    model = "unknown"
    cores = os.cpu_count() or 0
    freq_mhz = 0.0

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
                if line.startswith("Model") and model == "unknown":
                    model = line.split(":", 1)[1].strip()
    except OSError:
        pass

    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq") as f:
            freq_khz = int(f.read().strip())
            freq_mhz = freq_khz / 1000.0
    except (OSError, ValueError):
        pass

    return CpuInfo(model=model, cores=cores, freq_mhz=freq_mhz)


def _detect_memory() -> MemoryInfo:
    total_mb = 0
    available_mb = 0
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_mb = int(mem.total / (1024 * 1024))
        available_mb = int(mem.available / (1024 * 1024))
        return MemoryInfo(
            total_mb=total_mb,
            available_mb=available_mb,
            used_mb=int(mem.used / (1024 * 1024)),
            percent_used=mem.percent,
        )
    except ImportError:
        pass

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_mb = int(line.split()[1]) // 1024
                if line.startswith("MemAvailable:"):
                    available_mb = int(line.split()[1]) // 1024
    except OSError:
        pass

    used_mb = total_mb - available_mb
    percent = (used_mb / total_mb * 100) if total_mb > 0 else 0.0
    return MemoryInfo(total_mb=total_mb, available_mb=available_mb, used_mb=used_mb, percent_used=percent)


def _detect_disk(path: str = ".") -> DiskInfo:
    usage = shutil.disk_usage(path)
    return DiskInfo(
        path=path,
        total_mb=usage.total // (1024 * 1024),
        free_mb=usage.free // (1024 * 1024),
        used_mb=usage.used // (1024 * 1024),
        percent_used=round(usage.used / usage.total * 100, 1),
    )


def _detect_thermal() -> ThermalInfo:
    cpu_temp: float | None = None
    gpu_temp: float | None = None
    throttle = False

    temp_paths = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone1/temp",
        "/sys/class/thermal/thermal_zone2/temp",
    ]
    for tp in temp_paths:
        try:
            with open(tp) as f:
                val = int(f.read().strip()) / 1000.0
            if "cpu" in tp.lower() or cpu_temp is None:
                cpu_temp = val
        except (OSError, ValueError):
            continue

    try:
        p = Path("/sys/devices/virtual/thermal/thermal_zone0/temp")
        if p.exists():
            raw = int(p.read_text().strip()) / 1000.0
            gpu_temp = raw
    except (OSError, ValueError):
        pass

    try:
        throttle_path = Path("/sys/devices/platform/tegra-oc/oc_freq_cap_odm")
        if throttle_path.exists():
            throttle = "1" in throttle_path.read_text().strip()
    except (OSError, PermissionError):
        pass

    return ThermalInfo(cpu_temp_c=cpu_temp, gpu_temp_c=gpu_temp, throttle_active=throttle)


def detect_resources() -> ResourceProfile:
    is_jetson = _detect_is_jetson()
    return ResourceProfile(
        is_jetson=is_jetson,
        jetson_model=_parse_jetson_model() if is_jetson else "",
        gpu=_detect_gpu(),
        cpu=_detect_cpu(),
        memory=_detect_memory(),
        disk=_detect_disk(),
        thermal=_detect_thermal(),
    )


def disk_budget_exceeded(path: str, budget_mb: int) -> bool:
    usage = shutil.disk_usage(path)
    free_mb = usage.free // (1024 * 1024)
    return free_mb < budget_mb


def ram_pressure_pct() -> float:
    info = _detect_memory()
    return info.percent_used
