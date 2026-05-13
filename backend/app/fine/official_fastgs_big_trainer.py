from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.fine.option_utils import read_float, read_int
from app.fine.types import FineFailure


Progress = Callable[[str, int, str], None]


@dataclass(slots=True)
class OfficialFastGSTrainResult:
    ply_path: Path
    iterations: int
    metrics: dict[str, Any]


def train_official_fastgs_big(
    *,
    scene_dir: Path,
    output_dir: Path,
    iterations: int,
    options: dict[str, Any],
    progress: Progress,
) -> OfficialFastGSTrainResult:
    vendor_root = _fastgs_vendor_root()
    _require_fastgs_vendor(vendor_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    iterations = read_int((options or {}).get("fine_iterations"), iterations, minimum=5_000, maximum=60_000)
    densification_interval = read_int(
        (options or {}).get("fine_densification_interval"),
        100,
        minimum=1,
        maximum=10_000,
    )
    data_device = str((options or {}).get("fine_data_device") or "cpu").strip().lower()
    if data_device not in {"cpu", "cuda"}:
        raise FineFailure("UNSUPPORTED_FASTGS_DATA_DEVICE", f"Unsupported FastGS data_device: {data_device}")
    grad_abs_thresh = read_float((options or {}).get("fine_grad_abs_thresh"), 0.0004, minimum=1e-7, maximum=0.1)
    dense = read_float((options or {}).get("fine_dense"), 0.005, minimum=0.0, maximum=1.0)
    mult = read_float((options or {}).get("fine_mult"), 0.7, minimum=0.01, maximum=10.0)
    lambda_dssim = read_float((options or {}).get("fine_lambda_dssim"), 0.2, minimum=0.0, maximum=1.0)
    highfeature_lr = _optional_float(options or {}, "fine_highfeature_lr", fallback=0.02, minimum=1e-7, maximum=1.0)
    lowfeature_lr = _optional_float(options or {}, "fine_lowfeature_lr", fallback=None, minimum=1e-7, maximum=1.0)
    resolution = _optional_int(options or {}, "fine_train_resolution", minimum=1, maximum=16_384)

    command = [
        sys.executable,
        str(vendor_root / "train.py"),
        "-s",
        str(scene_dir.resolve()),
        "-i",
        "images",
        "-m",
        str(output_dir.resolve()),
        "--iterations",
        str(iterations),
        "--save_iterations",
        str(iterations),
        "--test_iterations",
        str(iterations),
        "--checkpoint_iterations",
        str(iterations),
        "--densification_interval",
        str(densification_interval),
        "--optimizer_type",
        "default",
        "--data_device",
        data_device,
        "--grad_abs_thresh",
        str(grad_abs_thresh),
        "--dense",
        str(dense),
        "--mult",
        str(mult),
        "--lambda_dssim",
        str(lambda_dssim),
    ]
    if highfeature_lr is not None:
        command.extend(["--highfeature_lr", str(highfeature_lr)])
    if lowfeature_lr is not None:
        command.extend(["--lowfeature_lr", str(lowfeature_lr)])
    if resolution is not None:
        command.extend(["-r", str(resolution)])

    log_path = output_dir / "fastgs_train.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(vendor_root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )

    progress("fine_fastgs_train_start", 44, f"training official FastGS-Big for {iterations} iterations")
    tail: deque[str] = deque(maxlen=80)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(" ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=str(vendor_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line_number, line in enumerate(process.stdout, start=1):
            log_file.write(line)
            tail.append(line.rstrip())
            if _is_progress_line(line):
                progress("fine_fastgs_training", 70, line.strip()[:220] or "FastGS-Big training")
        return_code = process.wait()

    if return_code != 0:
        raise FineFailure("FASTGS_TRAIN_FAILED", "\n".join(tail) or f"FastGS train.py exited with {return_code}")

    ply_path = _find_final_ply(output_dir, iterations)
    if ply_path is None:
        raise FineFailure("FASTGS_PLY_NOT_FOUND", f"FastGS did not create point_cloud.ply under {output_dir / 'point_cloud'}")

    metrics = {
        "training_backend": "official_fastgs_big",
        "fastgs_vendor_root": str(vendor_root),
        "fastgs_mode": "big",
        "iterations": iterations,
        "data_device": data_device,
        "densification_interval": densification_interval,
        "grad_abs_thresh": grad_abs_thresh,
        "dense": dense,
        "mult": mult,
        "lambda_dssim": lambda_dssim,
        "final_ply_bytes": ply_path.stat().st_size,
        "fastgs_log_path": str(log_path),
    }
    if highfeature_lr is not None:
        metrics["highfeature_lr"] = highfeature_lr
    if lowfeature_lr is not None:
        metrics["lowfeature_lr"] = lowfeature_lr
    if resolution is not None:
        metrics["resolution"] = resolution
    return OfficialFastGSTrainResult(ply_path=ply_path, iterations=iterations, metrics=metrics)


def _fastgs_vendor_root() -> Path:
    configured = os.getenv("FASTGS_VENDOR_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parent / "vendor" / "fastgs"


def _require_fastgs_vendor(vendor_root: Path) -> None:
    missing = [
        str(path)
        for path in (
            vendor_root / "train.py",
            vendor_root / "gaussian_renderer",
            vendor_root / "scene",
            vendor_root / "utils",
        )
        if not path.exists()
    ]
    if missing:
        raise FineFailure("FASTGS_VENDOR_MISSING", f"FastGS vendor source is incomplete: {', '.join(missing)}")


def _optional_float(
    options: dict[str, Any],
    key: str,
    *,
    fallback: float | None,
    minimum: float,
    maximum: float,
) -> float | None:
    if key not in options and fallback is None:
        return None
    return read_float(options.get(key), fallback if fallback is not None else minimum, minimum=minimum, maximum=maximum)


def _optional_int(options: dict[str, Any], key: str, *, minimum: int, maximum: int) -> int | None:
    if key not in options or options.get(key) in {None, ""}:
        return None
    return read_int(options.get(key), minimum, minimum=minimum, maximum=maximum)


def _is_progress_line(line: str) -> bool:
    lowered = line.lower()
    return any(token in lowered for token in ("training progress", "[iter", "saving gaussians", "training complete"))


def _find_final_ply(output_dir: Path, iterations: int) -> Path | None:
    exact = output_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if exact.exists() and exact.stat().st_size > 0:
        return exact

    point_cloud_dir = output_dir / "point_cloud"
    if not point_cloud_dir.exists():
        return None
    candidates = []
    for path in point_cloud_dir.glob("iteration_*/point_cloud.ply"):
        match = re.search(r"iteration_(\d+)", str(path.parent.name))
        if path.stat().st_size > 0 and match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]
