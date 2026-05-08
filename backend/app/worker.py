from __future__ import annotations

import os
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.algorithms import normalize_preview_pipeline
from app.config import get_settings
from app.database import SessionLocal, initialize_database_schema
from app.models import Artifact, MediaAsset, Project, Task, TaskEvent, WorkerHeartbeat, utc_now
from app.preview.image_preprocess import normalize_image_directory
from app.preview.runner import run_preview_pipeline
from app.preview.types import PreviewContext, PreviewFailure
from app.preview.weights import ModelDownloadError, download_model_weights, weights_for_pipeline
from app.resources import collect_gpu
from app.storage import Storage, sha256_path, storage_key

settings = get_settings()
storage = Storage(settings)


def main() -> None:
    worker_id = os.getenv("WORKER_ID") or f"preview-{socket.gethostname()}-{os.getpid()}"
    initialize_database_schema()
    redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    recover_interrupted_preview_tasks(redis_client)
    print(f"[worker] {worker_id} listening on {settings.preview_queue_name}", flush=True)
    while True:
        heartbeat(worker_id)
        item = redis_client.blpop(settings.preview_queue_name, timeout=5)
        if not item:
            continue
        _, task_id = item
        try:
            run_preview_task(task_id, worker_id)
        except Exception as exc:
            print(f"[worker] unexpected failure for {task_id}: {exc}", flush=True)


def recover_interrupted_preview_tasks(redis_client: redis.Redis) -> None:
    queued_ids = set(redis_client.lrange(settings.preview_queue_name, 0, -1))
    recovered = []
    with SessionLocal() as db:
        tasks = db.scalars(
            select(Task)
            .where(Task.type == "preview")
            .where(Task.status.in_(["queued", "running"]))
            .order_by(Task.created_at)
        ).all()
        for task in tasks:
            if task.id in queued_ids:
                continue
            task.status = "queued"
            task.current_stage = "queued"
            task.worker_id = None
            task.logs = append_log(task.logs, "worker startup recovered interrupted preview task")
            redis_client.rpush(settings.preview_queue_name, task.id)
            recovered.append(task.id)
        db.commit()
    if recovered:
        print(f"[worker] recovered {len(recovered)} preview task(s): {', '.join(recovered)}", flush=True)


def heartbeat(worker_id: str, current_task_id: str | None = None) -> None:
    gpu = collect_gpu()
    first = (gpu.get("gpus") or [{}])[0] if gpu.get("available") else {}
    with SessionLocal() as db:
        item = db.get(WorkerHeartbeat, worker_id) or WorkerHeartbeat(worker_id=worker_id, hostname=socket.gethostname())
        item.gpu_index = first.get("index")
        item.gpu_name = first.get("name")
        item.gpu_memory_total = first.get("memory_total")
        item.gpu_memory_used = first.get("memory_used")
        item.gpu_utilization = first.get("usage_percent")
        item.current_task_id = current_task_id
        item.last_seen_at = utc_now()
        db.merge(item)
        db.commit()


def run_preview_task(task_id: str, worker_id: str) -> None:
    started = time.monotonic()
    work_dir = settings.local_work_root / task_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as db:
        task = db.get(Task, task_id)
        if not task:
            return
        project = db.scalar(
            select(Project)
            .where(Project.id == task.project_id)
            .options(selectinload(Project.media), selectinload(Project.artifacts))
        )
        if not project:
            fail_task(db, task, None, "PROJECT_NOT_FOUND", "Project not found")
            return
        task.status = "running"
        task.worker_id = worker_id
        task.started_at = utc_now()
        task.current_stage = "input_preparing"
        task.progress = 4
        task.eta_seconds = estimate_eta(task, project, started)
        task.logs = append_log(task.logs, f"worker {worker_id} started task")
        project.status = "PREVIEW_RUNNING"
        emit(db, project.id, "task_started", task_payload(task), task.id)
        db.commit()

    heartbeat(worker_id, task_id)
    try:
        with SessionLocal() as db:
            task = db.get(Task, task_id)
            project = db.scalar(
                select(Project)
                .where(Project.id == task.project_id)
                .options(selectinload(Project.media), selectinload(Project.artifacts))
            )
            if not task or not project:
                return

            pipeline = normalize_preview_pipeline((task.options or {}).get("preview_pipeline"), project.input_type)
            ensure_pipeline_weights(db, task, project, pipeline, started)
            input_dir, input_video = download_media(project, work_dir)
            update_task(db, task, project, "input_downloaded", 13, started, f"downloaded {len(project.media)} media files")
            input_metrics: dict[str, Any] = {}
            if any(media.kind == "image" for media in project.media):
                max_side = read_positive_int((task.options or {}).get("preview_image_max_side"), settings.preview_image_max_side)
                jpeg_quality = read_positive_int((task.options or {}).get("preview_image_jpeg_quality"), settings.preview_image_jpeg_quality)
                normalized = normalize_image_directory(
                    input_dir,
                    work_dir / "input_normalized",
                    max_side=max_side,
                    jpeg_quality=jpeg_quality,
                )
                input_dir = normalized.output_dir
                input_metrics = normalized.metrics()
                update_task(
                    db,
                    task,
                    project,
                    "input_ready",
                    14,
                    started,
                    f"normalized {normalized.output_count} images to RGB JPEG, max side {normalized.max_side}px",
                )
            else:
                update_task(db, task, project, "input_ready", 14, started, f"downloaded {len(project.media)} media files")

            output_spz = work_dir / "preview.spz"
            source_version = int((task.options or {}).get("source_version") or project.source_version)
            ctx = PreviewContext(
                task_id=task.id,
                project_id=project.id,
                pipeline=pipeline,
                input_dir=input_dir,
                input_video=input_video,
                work_dir=work_dir,
                output_spz=output_spz,
                model_cache_dir=Path(settings.model_cache_dir).resolve(),
                source_version=source_version,
                options=task.options or {},
                progress=lambda stage, progress, message=None, metrics=None: progress_task(
                    task.id, project.id, stage, progress, started, message, metrics
                ),
            )
            update_task(db, task, project, stage_for_pipeline(pipeline), 20, started, f"running bundled preview adapter: {pipeline}")
            result = run_preview_pipeline(ctx)

            if not output_spz.exists() or output_spz.stat().st_size <= 0:
                raise PreviewFailure("ARTIFACT_NOT_FOUND", f"Algorithm finished but did not create non-empty SPZ: {output_spz}")

            with SessionLocal() as upload_db:
                task = upload_db.get(Task, task_id)
                project = upload_db.scalar(
                    select(Project)
                    .where(Project.id == task.project_id)
                    .options(selectinload(Project.media), selectinload(Project.artifacts))
                )
                update_task(upload_db, task, project, "uploading_artifact", 92, started, "validated non-empty preview.spz")
                key = storage_key("users", project.owner_id, "projects", project.id, "preview", "preview.spz")
                checksum = sha256_path(output_spz)
                uri = storage.upload_path(output_spz, key)
                artifact = Artifact(
                    project_id=project.id,
                    task_id=task.id,
                    kind="preview_spz",
                    object_uri=uri,
                    file_name="preview.spz",
                    file_size=output_spz.stat().st_size,
                    checksum=checksum,
                    source_version=source_version,
                    metadata_json={
                        "pipeline": pipeline,
                        "source_version": source_version,
                        "generated_by": worker_id,
                        "adapter": result.metrics.get("adapter"),
                        "source_commits": result.source_commits,
                        "splat_count": result.splat_count,
                        "intermediate_ply": str(result.intermediate_ply) if result.intermediate_ply else None,
                        "intermediate_ply_size": result.metrics.get("intermediate_ply_size"),
                    },
                )
                upload_db.add(artifact)
                ply_artifact = None
                if result.intermediate_ply and result.intermediate_ply.exists() and result.intermediate_ply.stat().st_size > 0:
                    ply_key = storage_key("users", project.owner_id, "projects", project.id, "preview", task.id, "original.ply")
                    ply_uri = storage.upload_path(result.intermediate_ply, ply_key)
                    ply_artifact = Artifact(
                        project_id=project.id,
                        task_id=task.id,
                        kind="original_ply",
                        object_uri=ply_uri,
                        file_name="original.ply",
                        file_size=result.intermediate_ply.stat().st_size,
                        checksum=sha256_path(result.intermediate_ply),
                        source_version=source_version,
                        metadata_json={
                            "pipeline": pipeline,
                            "source_version": source_version,
                            "generated_by": worker_id,
                            "source_commits": result.source_commits,
                        },
                    )
                    upload_db.add(ply_artifact)
                task.status = "succeeded"
                task.progress = 100
                task.current_stage = "preview_ready"
                task.eta_seconds = 0
                task.finished_at = utc_now()
                task.metrics = {
                    **(task.metrics or {}),
                    "duration_seconds": round(time.monotonic() - started, 2),
                    "output_bytes": output_spz.stat().st_size,
                    "pipeline": pipeline,
                    "splat_count": result.splat_count,
                    "source_commits": result.source_commits,
                    **input_metrics,
                    **result.metrics,
                }
                task.logs = append_log(task.logs, "uploaded preview.spz")
                if ply_artifact:
                    task.logs = append_log(task.logs, "uploaded original.ply")
                project.status = "PREVIEW_READY"
                project.preview_source_version = artifact.source_version
                project.error_message = None
                emit(upload_db, project.id, "artifact_created", {"artifact_id": artifact.id, "kind": artifact.kind}, task.id)
                if ply_artifact:
                    emit(upload_db, project.id, "artifact_created", {"artifact_id": ply_artifact.id, "kind": ply_artifact.kind}, task.id)
                emit(upload_db, project.id, "task_succeeded", task_payload(task), task.id)
                upload_db.commit()
    except PreviewFailure as exc:
        with SessionLocal() as db:
            task = db.get(Task, task_id)
            project = db.get(Project, task.project_id) if task else None
            fail_task(db, task, project, exc.code, exc.message)
    except Exception as exc:
        with SessionLocal() as db:
            task = db.get(Task, task_id)
            project = db.get(Project, task.project_id) if task else None
            fail_task(db, task, project, "ALGORITHM_EXECUTION_FAILED", str(exc))
    finally:
        heartbeat(worker_id)


def ensure_gpu_available() -> None:
    gpu = collect_gpu()
    if not gpu.get("available"):
        raise PreviewFailure("GPU_RESOURCE_UNAVAILABLE", str(gpu.get("message") or "GPU is unavailable"))


def ensure_pipeline_weights(db, task: Task, project: Project, pipeline: str, started: float) -> None:
    specs = weights_for_pipeline(pipeline)
    if not specs:
        return
    if not settings.model_auto_download:
        return
    update_task(db, task, project, "weights_checking", 8, started, f"checking {len(specs)} model weights for {pipeline}")
    try:
        download_model_weights(
            Path(settings.model_cache_dir),
            specs,
            prefer_hf_mirror=settings.model_download_prefer_hf_mirror,
            lock_timeout_seconds=settings.model_download_lock_timeout_seconds,
            log=lambda line: progress_task(task.id, project.id, "weights_downloading", 10, started, line),
        )
    except ModelDownloadError as exc:
        raise PreviewFailure("MODEL_WEIGHT_DOWNLOAD_FAILED", str(exc)) from exc
    update_task(db, task, project, "weights_ready", 12, started, f"model weights ready for {pipeline}")


def download_media(project: Project, work_dir: Path) -> tuple[Path, Path | None]:
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_video: Path | None = None
    for index, media in enumerate(project.media):
        suffix = Path(media.file_name).suffix or (".jpg" if media.kind == "image" else ".mp4")
        target = input_dir / f"{index:06d}-{media.id}{suffix}"
        storage.download_to_path(media.object_uri, target)
        if media.kind == "video" and input_video is None:
            input_video = target
    return input_dir, input_video


def read_positive_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def update_task(db, task: Task, project: Project, stage: str, progress: int, started: float, *logs: str) -> None:
    task.current_stage = stage
    task.progress = max(task.progress or 0, progress)
    task.eta_seconds = estimate_eta(task, project, started)
    if logs:
        task.logs = append_log(task.logs, *logs)
    emit(db, project.id, "task_progress", task_payload(task), task.id)
    db.commit()


def progress_task(
    task_id: str,
    project_id: str,
    stage: str,
    progress: int,
    started: float,
    message: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        project = db.get(Project, project_id)
        if not task or not project:
            return
        if metrics:
            task.metrics = {**(task.metrics or {}), **metrics}
        update_task(db, task, project, stage, progress, started, *(message,) if message else ())


def fail_task(db, task: Task | None, project: Project | None, code: str, message: str) -> None:
    if task:
        task.status = "failed"
        task.error_code = code
        task.error_message = message
        task.finished_at = utc_now()
        task.current_stage = "failed"
        task.logs = append_log(task.logs, f"{code}: {message}")
    if project:
        project.status = "FAILED"
        project.error_message = message
    if task and project:
        emit(db, project.id, "task_failed", task_payload(task), task.id)
    db.commit()


def append_log(existing: list[str] | None, *lines: str) -> list[str]:
    logs = list(existing or [])
    for line in lines:
        if line and (not logs or logs[-1] != line):
            logs.append(line)
    return logs[-240:]


def emit(db, project_id: str, event: str, payload: dict[str, Any], task_id: str | None = None) -> None:
    db.add(TaskEvent(project_id=project_id, task_id=task_id, event=event, payload=payload))


def task_payload(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "project_id": task.project_id,
        "project_name": task.project.name if task.project else None,
        "type": task.type,
        "status": task.status,
        "progress": task.progress,
        "worker_id": task.worker_id,
        "options": task.options or {},
        "current_stage": task.current_stage,
        "eta_seconds": task.eta_seconds,
        "error_code": task.error_code,
        "error_message": task.error_message,
        "metrics": task.metrics or {},
        "logs": task.logs or [],
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "finished_at": task.finished_at.isoformat() if task.finished_at else None,
    }


def estimate_eta(task: Task, project: Project, started: float) -> int | None:
    progress = max(1, min(99, int(task.progress or 0)))
    if progress >= 100:
        return 0
    elapsed = max(1.0, time.monotonic() - started)
    by_progress = elapsed * (100 - progress) / progress
    expected = expected_seconds_for_task(task, project)
    by_expected = max(0, expected - elapsed)
    if progress < 20:
        return int(by_expected)
    return int(max(0, by_progress * 0.55 + by_expected * 0.45))


def expected_seconds_for_task(task: Task, project: Project) -> int:
    pipeline = str((task.options or {}).get("preview_pipeline") or "")
    if pipeline == "litevggt_edgs":
        steps = int((task.options or {}).get("edgs_preview_steps") or 200)
        frame_count = max(1, len(project.media))
        return int(max(25, 24 + steps * 0.035 + max(0, frame_count - 6) * 0.8))
    return expected_seconds_for_pipeline(pipeline)


def expected_seconds_for_pipeline(pipeline: str | None) -> int:
    if pipeline == "litevggt_spz":
        return settings.preview_expected_seconds_litevggt_spz
    if pipeline == "lingbot_spz":
        return settings.preview_expected_seconds_video
    return settings.preview_expected_seconds_litevggt_edgs


def stage_for_pipeline(pipeline: str) -> str:
    if pipeline == "litevggt_spz":
        return "litevggt_direct_spz"
    if pipeline == "lingbot_spz":
        return "lingbot_map_video_spz"
    return "litevggt_edgs_training"


if __name__ == "__main__":
    main()
