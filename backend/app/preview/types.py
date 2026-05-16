from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[[str, int, str | None, dict[str, Any] | None], None]


class PreviewFailure(Exception):
    """算法预览任务的可预期失败，worker 会把 code 写入 task.error_code。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True)
class PreviewContext:
    task_id: str
    project_id: str
    pipeline: str
    input_dir: Path
    work_dir: Path
    output_spz: Path
    model_cache_dir: Path
    source_version: int
    options: dict[str, Any]
    progress: ProgressCallback

    def model_path(self, *parts: str) -> Path:
        return self.model_cache_dir.joinpath(*parts)

    def report(self, stage: str, progress: int, message: str | None = None, metrics: dict[str, Any] | None = None) -> None:
        self.progress(stage, progress, message, metrics)


@dataclass(slots=True)
class PreviewArtifactResult:
    path: Path
    kind: str
    file_name: str


@dataclass(slots=True)
class PreviewResult:
    output_spz: Path
    intermediate_ply: Path | None
    splat_count: int | None
    metrics: dict[str, Any] = field(default_factory=dict)
    source_commits: dict[str, str] = field(default_factory=dict)
    primary_artifact: Path | None = None
    primary_artifact_kind: str = "preview_spz"
    primary_artifact_file_name: str = "preview.spz"
    primary_artifact_format: str = "spz"
    extra_artifacts: tuple[PreviewArtifactResult, ...] = ()


SOURCE_COMMITS: dict[str, str] = {
    "LiteVGGT": "4767c17f8b6f176bb751566e92f60eb885040033",
    "LingBot-Map": "4cd986009b9adeded8a4e740919221940dedeffe",
    "Spark": "3cf9fa15adb7ac7c47a1e962740db97b9e8a9fdf",
}
