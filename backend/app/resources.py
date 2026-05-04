from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import psutil


def collect_resources() -> dict[str, Any]:
    return {
        "cpu": collect_cpu(),
        "memory": collect_memory(),
        "gpu": collect_gpu(),
        "workers": {"available": True},
    }


def collect_cpu() -> dict[str, Any]:
    return {
        "available": True,
        "source": "psutil",
        "usage_percent": psutil.cpu_percent(interval=0.05),
        "logical_count": psutil.cpu_count(logical=True),
        "physical_count": psutil.cpu_count(logical=False),
    }


def collect_memory() -> dict[str, Any]:
    mem = psutil.virtual_memory()
    return {
        "available": True,
        "source": "psutil",
        "total": mem.total,
        "used": mem.used,
        "usage_percent": mem.percent,
    }


def collect_gpu() -> dict[str, Any]:
    nvml = collect_gpu_nvml()
    if nvml["available"]:
        return nvml
    smi = collect_gpu_nvidia_smi()
    if smi["available"]:
        return smi
    return nvml


def collect_gpu_nvml() -> dict[str, Any]:
    try:
        import pynvml

        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8", errors="replace")
        gpus = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpus.append(
                {
                    "index": index,
                    "name": name,
                    "usage_percent": float(util.gpu),
                    "memory_used": int(memory.used / 1024 / 1024),
                    "memory_total": int(memory.total / 1024 / 1024),
                    "memory_usage_percent": round((memory.used / memory.total) * 100, 2) if memory.total else 0,
                }
            )
        totals = {
            "memory_used": sum(item["memory_used"] for item in gpus),
            "memory_total": sum(item["memory_total"] for item in gpus),
        }
        return {
            "available": bool(gpus),
            "source": "pynvml",
            "driver_version": driver,
            "cuda_runtime": None,
            "gpus": gpus,
            "usage_percent": max((item["usage_percent"] for item in gpus), default=0),
            **totals,
            "memory_usage_percent": round((totals["memory_used"] / totals["memory_total"]) * 100, 2) if totals["memory_total"] else 0,
        }
    except Exception as exc:
        return {"available": False, "source": "pynvml", "message": str(exc)}


def collect_gpu_nvidia_smi() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        gpus = []
        driver = None
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            index, name, util, mem_used, mem_total, driver_version = [part.strip() for part in line.split(",", 5)]
            driver = driver_version
            used = int(float(mem_used))
            total = int(float(mem_total))
            gpus.append(
                {
                    "index": int(index),
                    "name": name,
                    "usage_percent": float(util),
                    "memory_used": used,
                    "memory_total": total,
                    "memory_usage_percent": round((used / total) * 100, 2) if total else 0,
                }
            )
        return {
            "available": bool(gpus),
            "source": "nvidia-smi",
            "driver_version": driver,
            "cuda_runtime": query_cuda_runtime(),
            "gpus": gpus,
            "usage_percent": max((item["usage_percent"] for item in gpus), default=0),
            "memory_used": sum(item["memory_used"] for item in gpus),
            "memory_total": sum(item["memory_total"] for item in gpus),
        }
    except Exception as exc:
        return {"available": False, "source": "nvidia-smi", "message": str(exc)}


def query_cuda_runtime() -> str | None:
    try:
        result = subprocess.run(["nvidia-smi", "-q", "-x"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5)
        if "<cuda_version>" in result.stdout:
            return result.stdout.split("<cuda_version>", 1)[1].split("</cuda_version>", 1)[0].strip()
    except Exception:
        return None
    return None


def python_info() -> dict[str, Any]:
    return {"version": sys.version.split()[0], "executable": sys.executable}


def torch_info() -> dict[str, Any]:
    try:
        import torch

        return {
            "available": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception as exc:
        return {"available": False, "cuda_available": False, "error": str(exc)}


def to_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))

