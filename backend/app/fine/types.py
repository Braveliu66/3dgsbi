from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class FineFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class FineContext:
    task_id: str
    project_id: str
    pipeline: str
    input_dir: Path
    input_video: Path | None
    work_dir: Path
    model_cache_dir: Path
    final_ply: Path
    final_spz: Path | None
    metrics_json: Path
    viewer_meta_json: Path | None
    lod_rad: Path | None
    source_version: int
    options: dict[str, Any]
    progress: Callable[[str, int, str | None, dict[str, Any] | None], None] | None = None


@dataclass
class FineResult:
    final_ply: Path
    final_spz: Path
    metrics_json: Path
    viewer_meta_json: Path | None
    lod_rad: Path | None
    splat_count: int | None
    source_commits: dict[str, str]
    metrics: dict[str, Any]
