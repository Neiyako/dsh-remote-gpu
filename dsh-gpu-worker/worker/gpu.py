"""GPU / CUDA detection that works without PyTorch installed.

Detection strategy:
1. Try ``torch.cuda`` (authoritative when torch is available).
2. Fall back to parsing ``nvidia-smi`` output (gives device info without torch).
3. Otherwise report ``cuda_available: false`` with a human readable ``error``.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import threading
import time
from typing import Any

_CACHE_TTL_SECONDS = 10.0
_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"at": 0.0, "payload": None}


def _import_torch():
    try:
        return importlib.import_module("torch")
    except Exception:  # ImportError or a broken CUDA install
        return None


def _import_pynvml():
    try:
        return importlib.import_module("pynvml")
    except Exception:
        return None


def _nvidia_smi_path() -> str | None:
    return shutil.which("nvidia-smi")


def _run_nvidia_smi(query: str) -> list[list[str]] | None:
    """Run ``nvidia-smi --query-gpu=...`` and return CSV rows."""
    executable = _nvidia_smi_path()
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    rows: list[list[str]] = []
    for line in result.stdout.strip().splitlines():
        rows.append([part.strip() for part in line.split(",")])
    return rows


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _device_from_nvidia_smi() -> list[dict[str, Any]]:
    rows = _run_nvidia_smi(
        "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,driver_version"
    )
    if rows is None:
        return []
    driver_rows = _run_nvidia_smi("driver_version")
    driver = (driver_rows[0][0] if driver_rows else None) or None

    devices: list[dict[str, Any]] = []
    for row in rows:
        if len(row) < 7:
            continue
        total = _float_or_none(row[2])
        used = _float_or_none(row[3])
        free = (total - used) if (total is not None and used is not None) else None
        devices.append(
            {
                "index": int(_float_or_none(row[0]) or 0),
                "name": row[1],
                "capability": "unknown",
                "total_memory_mb": int(total) if total is not None else None,
                "free_memory_mb": int(free) if free is not None else None,
                "used_memory_mb": int(used) if used is not None else None,
                "utilization_percent": int(_float_or_none(row[4] or "")) if row[4] not in {"", "N/A", "[N/A]"} else None,
                "temperature_c": int(_float_or_none(row[5] or "")) if row[5] not in {"", "N/A", "[N/A]"} else None,
                "torch_version": None,
                "driver_version": row[6] or driver,
            }
        )
    return devices


def _device_from_torch(torch) -> list[dict[str, Any]]:
    pynvml = _import_pynvml()
    nvml_handles = None
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            nvml_handles = True
        except Exception:
            nvml_handles = None

    devices: list[dict[str, Any]] = []
    driver = None
    driver_rows = _run_nvidia_smi("driver_version")
    if driver_rows:
        driver = driver_rows[0][0]

    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        capability = f"{properties.major}.{properties.minor}"
        total = int(properties.total_memory / (1024 * 1024))
        free: int | None = None
        used: int | None = None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            free = int(free_bytes / (1024 * 1024))
            total = int(total_bytes / (1024 * 1024))
            used = total - free
        except Exception:
            pass

        utilization = None
        temperature = None
        try:
            utilization = int(torch.cuda.utilization(index))
        except Exception:
            pass
        if pynvml is not None and nvml_handles is not None:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                if utilization is None:
                    utilization = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                temperature = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                if driver is None:
                    driver = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver, bytes):
                        driver = driver.decode()
            except Exception:
                pass

        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": capability,
                "total_memory_mb": total,
                "free_memory_mb": free,
                "used_memory_mb": used,
                "utilization_percent": utilization,
                "temperature_c": temperature,
                "torch_version": getattr(torch, "__version__", None),
                "driver_version": driver,
            }
        )

    if pynvml is not None and nvml_handles is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return devices


def gpu_info(*, refresh: bool = False) -> dict[str, Any]:
    """Return the ``GET /gpu`` payload, cached for a few seconds."""
    now = time.monotonic()
    with _cache_lock:
        cached = _cache["payload"]
        if not refresh and cached is not None and now - _cache["at"] < _CACHE_TTL_SECONDS:
            return cached

    torch = _import_torch()
    devices: list[dict[str, Any]] = []
    error: str | None = None
    smi_available = _nvidia_smi_path() is not None

    if torch is not None and getattr(torch.cuda, "is_available", lambda: False)():
        try:
            devices = _device_from_torch(torch)
        except Exception as exc:  # pragma: no cover - hardware dependent
            error = f"torch.cuda device enumeration failed: {exc}"
            devices = []
        if not devices and error is None:
            error = "torch reports CUDA available but no devices were enumerated"
    else:
        devices = _device_from_nvidia_smi()
        if not devices:
            if torch is None:
                error = (
                    "CUDA unavailable: PyTorch is not installed in this environment "
                    "and no nvidia-smi device was detected."
                )
            else:
                error = "CUDA unavailable: torch.cuda.is_available() is false and nvidia-smi found no device."

    payload = {
        "cuda_available": bool(devices),
        "device_count": len(devices),
        "devices": devices,
        "nvidia_smi_available": smi_available,
        "error": error,
        "torch_installed": torch is not None,
        "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
        "time": None,
    }
    with _cache_lock:
        _cache["at"] = now
        _cache["payload"] = payload
    return payload


def cuda_available(*, refresh: bool = False) -> bool:
    return bool(gpu_info(refresh=refresh)["cuda_available"])


def cuda_unavailable_reason() -> str:
    info = gpu_info()
    return info.get("error") or "CUDA is not available on this worker"
