from __future__ import annotations

import contextlib
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.preview.types import PreviewFailure


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"


def image_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def require_file(path: Path, code: str, label: str) -> Path:
    if not path.exists() or not path.is_file():
        raise PreviewFailure(code, f"{label} not found: {path}")
    return path


def require_dir(path: Path, code: str, label: str) -> Path:
    if not path.exists() or not path.is_dir():
        raise PreviewFailure(code, f"{label} not found: {path}")
    return path


def require_cuda() -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on worker image
        raise PreviewFailure("TORCH_UNAVAILABLE", f"PyTorch import failed: {exc}") from exc
    if not torch.cuda.is_available():
        raise PreviewFailure("GPU_RESOURCE_UNAVAILABLE", "CUDA GPU is required for preview reconstruction")


@contextlib.contextmanager
def prepend_sys_path(*paths: Path) -> Iterator[None]:
    """临时把内置 vendor 路径放到 sys.path 前面，避免依赖外部仓库。"""

    resolved = [str(path.resolve()) for path in paths]
    old_path = list(sys.path)
    old_pythonpath = os.environ.get("PYTHONPATH")
    sys.path[:0] = resolved
    os.environ["PYTHONPATH"] = os.pathsep.join(resolved + ([old_pythonpath] if old_pythonpath else []))
    try:
        yield
    finally:
        sys.path = old_path
        if old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pythonpath


class StageTimer:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        self.marks[name] = round(time.monotonic() - self.started, 3)

    def metrics(self) -> dict[str, Any]:
        return {"stage_durations": dict(self.marks)}
