from __future__ import annotations

import os
import shutil
import socket
import time
import traceback
from pathlib import Path
from typing import Any

import redis
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.database import SessionLocal, initialize_database_schema
from app.fine.amb3r_sfm import ensure_amb3r_weight
from app.fine.runner import run_fine_pipeline
from app.fine.types import FineContext, FineFailure
from app.models import Artifact, Project, Task, utc_now
from app.preview.image_preprocess import normalize_image_directory
from app.preview.weights import ModelDownloadError, download_model_weights, weights_for_pipeline
from app.storage import Storage, sha256_path, storage_key
from app.worker import (
    TaskLogCapture,
    append_log,
    download_media,
    emit,
    fail_task,
    heartbeat,
    read_positive_int,
    run_task_in_subprocess,
    task_log_path,
    task_source_version,
    task_payload,
    upload_task_log,
)

settings = get_settings()
storage = Storage(settings)


def main() -> None:
    worker_id = os.getenv("WORKER_ID") or f"fine-{socket.gethostname()}-{os.getpid()}"
    if os.getenv("FINE_RECONSTRUCTION_ENABLED", "true").lower() not in {"1", "true", "yes", "on"}:
        print("[fine-worker] FINE_RECONSTRUCTION_ENABLED is false; worker will not start", flush=True)
        return
    initialize_database_schema()
    redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    recover_interrupted_fine_tasks(redis_client)
    print(f"[fine-worker] {worker_id} listening on {settings.fine_queue_name}", flush=True)
    while True:
        heartbeat(worker_id)
        item = redis_client.blpop(settings.fine_queue_name, timeout=5)
        if not item:
            continue
        _, task_id = item
        try:
            run_task_in_subprocess(task_id, worker_id, redis_client, run_fine_task, "fine-worker")
        except Exception as exc:
            print(f"[fine-worker] unexpected failure for {task_id}: {exc}", flush=True)


def recover_interrupted_fine_tasks(redis_client: redis.Redis) -> None:
    queued_ids = set(redis_client.lrange(settings.fine_queue_name, 0, -1))
    recovered = []
    with SessionLocal() as db:
        tasks = db.scalars(
            select(Task)
            .where(Task.type == "fine")
            .where(Task.status.in_(["queued", "running"]))
            .order_by(Task.created_at)
        ).all()
        for task in tasks:
            if task.id in queued_ids:
                continue
            task.status = "queued"
            task.current_stage = "queued"
            task.worker_id = None
            task.logs = append_log(task.logs, "worker startup recovered interrupted fine task")
            redis_client.rpush(settings.fine_queue_name, task.id)
            recovered.append(task.id)
        db.commit()
    if recovered:
        print(f"[fine-worker] recovered {len(recovered)} fine task(s): {', '.join(recovered)}", flush=True)


def run_fine_task(task_id: str, worker_id: str) -> None:
    started = time.monotonic()
    work_dir = settings.local_work_root / task_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_capture = TaskLogCapture(task_log_path(task_id))
    source_version = 0

    with SessionLocal() as db:
        task = db.get(Task, task_id)
        if not task:
            return
        if task.status == "canceled":
            return
        if task.status not in {"queued", "running"}:
            return
        project = db.scalar(
            select(Project)
            .where(Project.id == task.project_id)
            .options(selectinload(Project.media), selectinload(Project.artifacts))
        )
        if not project:
            fail_task(db, task, None, "PROJECT_NOT_FOUND", "Project not found")
            return
        source_version = task_source_version(task, project)
        task.status = "running"
        task.worker_id = worker_id
        task.started_at = utc_now()
        task.current_stage = "input_preparing"
        task.progress = 4
        task.eta_seconds = estimate_fine_eta(task, started)
        task.logs = append_log(task.logs, f"worker {worker_id} started fine reconstruction")
        project.status = "FINE_RUNNING"
        project.error_message = None
        emit(db, project.id, "task_started", task_payload(task), task.id)
        db.commit()

    log_capture.start()
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
            image_count = sum(1 for media in project.media if media.kind == "image")
            if project.input_type != "images" or image_count < 3:
                raise FineFailure("INSUFFICIENT_IMAGES", "Fine reconstruction requires an image project with at least 3 images")

            input_dir, _ = download_media(project, work_dir)
            update_task(db, task, project, "input_downloaded", 12, started, f"downloaded {len(project.media)} media files")
            max_side = read_positive_int((task.options or {}).get("fine_image_max_side"), settings.fine_image_max_side)
            normalized = normalize_image_directory(input_dir, work_dir / "input_normalized", max_side=max_side, jpeg_quality=92)
            input_dir = normalized.output_dir
            update_task(
                db,
                task,
                project,
                "input_ready",
                16,
                started,
                f"normalized {normalized.output_count} images to RGB JPEG, max side {normalized.max_side}px",
            )
            ensure_fine_weights(db, task, project, started)

            source_version = task_source_version(task, project)
            ctx = FineContext(
                task_id=task.id,
                project_id=project.id,
                pipeline=str((task.options or {}).get("fine_pipeline") or "mobilegs_lmrs"),
                input_dir=input_dir,
                work_dir=work_dir,
                model_cache_dir=Path(settings.model_cache_dir).resolve(),
                final_ply=work_dir / "final.ply",
                final_spz=work_dir / "final_web.spz",
                metrics_json=work_dir / "metrics.json",
                lod_rad=work_dir / "final_lod.rad",
                source_version=source_version,
                options=task.options or {},
                progress=lambda stage, progress, message=None, metrics=None: progress_fine_task(
                    task.id, project.id, stage, progress, started, message, metrics
                ),
            )
            update_task(db, task, project, "fine_preflight", 18, started, "checking MobileGS AMB3R-SfM + LM-RS runtime")
            result = run_fine_pipeline(ctx)

            if not result.final_ply.exists() or result.final_ply.stat().st_size <= 0:
                raise FineFailure("ARTIFACT_NOT_FOUND", f"Missing non-empty final.ply: {result.final_ply}")
            if not result.final_spz.exists() or result.final_spz.stat().st_size <= 0:
                raise FineFailure("ARTIFACT_NOT_FOUND", f"Missing non-empty final_web.spz: {result.final_spz}")

            with SessionLocal() as upload_db:
                task = upload_db.get(Task, task_id)
                if not task or task.status == "canceled":
                    return
                project = upload_db.scalar(
                    select(Project)
                    .where(Project.id == task.project_id)
                    .options(selectinload(Project.media), selectinload(Project.artifacts))
                )
                update_task(upload_db, task, project, "uploading_artifacts", 94, started, "uploading fine reconstruction artifacts")
                artifacts = upload_fine_artifacts(upload_db, project, task, worker_id, source_version, result)
                task.status = "succeeded"
                task.progress = 100
                task.current_stage = "fine_ready"
                task.eta_seconds = 0
                task.finished_at = utc_now()
                task.metrics = {
                    **(task.metrics or {}),
                    "duration_seconds": round(time.monotonic() - started, 2),
                    "pipeline": ctx.pipeline,
                    "splat_count": result.splat_count,
                    "source_commits": result.source_commits,
                    **normalized.metrics(),
                    **result.metrics,
                }
                task.logs = append_log(task.logs, "uploaded final.ply", "uploaded final_web.spz", "uploaded metrics.json")
                if result.lod_rad:
                    task.logs = append_log(task.logs, "uploaded final_lod.rad")
                log_capture.flush()
                log_artifact = upload_task_log(upload_db, project, task, worker_id, source_version)
                if log_artifact:
                    task.logs = append_log(task.logs, f"uploaded {log_artifact.file_name}")
                project.status = "COMPLETED"
                project.error_message = None
                for artifact in artifacts:
                    emit(upload_db, project.id, "artifact_created", {"artifact_id": artifact.id, "kind": artifact.kind}, task.id)
                if log_artifact:
                    emit(upload_db, project.id, "artifact_created", {"artifact_id": log_artifact.id, "kind": log_artifact.kind}, task.id)
                emit(upload_db, project.id, "task_succeeded", task_payload(task), task.id)
                upload_db.commit()
    except FineFailure as exc:
        traceback.print_exc()
        with SessionLocal() as db:
            task = db.get(Task, task_id)
            project = db.get(Project, task.project_id) if task else None
            log_capture.flush()
            if task and project:
                log_artifact = upload_task_log(db, project, task, worker_id, source_version or task_source_version(task, project))
                if log_artifact:
                    task.logs = append_log(task.logs, f"uploaded {log_artifact.file_name}")
                    emit(db, project.id, "artifact_created", {"artifact_id": log_artifact.id, "kind": log_artifact.kind}, task.id)
            fail_task(db, task, project, exc.code, exc.message)
    except Exception as exc:
        traceback.print_exc()
        with SessionLocal() as db:
            task = db.get(Task, task_id)
            project = db.get(Project, task.project_id) if task else None
            log_capture.flush()
            if task and project:
                log_artifact = upload_task_log(db, project, task, worker_id, source_version or task_source_version(task, project))
                if log_artifact:
                    task.logs = append_log(task.logs, f"uploaded {log_artifact.file_name}")
                    emit(db, project.id, "artifact_created", {"artifact_id": log_artifact.id, "kind": log_artifact.kind}, task.id)
            fail_task(db, task, project, "FINE_RECONSTRUCTION_FAILED", str(exc))
    finally:
        heartbeat(worker_id)
        log_capture.stop()


def upload_fine_artifacts(
    db,
    project: Project,
    task: Task,
    worker_id: str,
    source_version: int,
    result,
) -> list[Artifact]:
    common_metadata = {
        "pipeline": "mobilegs_lmrs",
        "source_version": source_version,
        "generated_by": worker_id,
        "source_commits": result.source_commits,
        "splat_count": result.splat_count,
    }
    specs = [
        ("final_ply", result.final_ply, storage_key("users", project.owner_id, "projects", project.id, "final", "final.ply"), "final.ply"),
        ("final_spz", result.final_spz, storage_key("users", project.owner_id, "projects", project.id, "final", "final_web.spz"), "final_web.spz"),
        ("metrics_json", result.metrics_json, storage_key("users", project.owner_id, "projects", project.id, "final", "metrics.json"), "metrics.json"),
    ]
    if result.lod_rad:
        specs.append(
            ("lod_rad", result.lod_rad, storage_key("users", project.owner_id, "projects", project.id, "final", "lod", "final_lod.rad"), "final_lod.rad")
        )
    artifacts = []
    for kind, path, key, file_name in specs:
        uri = storage.upload_path(path, key)
        artifact = Artifact(
            project_id=project.id,
            task_id=task.id,
            kind=kind,
            object_uri=uri,
            file_name=file_name,
            file_size=Path(path).stat().st_size,
            checksum=sha256_path(path),
            source_version=source_version,
            metadata_json={**common_metadata, "artifact": kind},
        )
        db.add(artifact)
        artifacts.append(artifact)
    return artifacts


def ensure_fine_weights(db, task: Task, project: Project, started: float) -> None:
    update_task(db, task, project, "weights_checking", 8, started, "checking AMB3R weight for fine reconstruction")
    if settings.model_auto_download:
        try:
            download_model_weights(
                Path(settings.model_cache_dir),
                weights_for_pipeline("mobilegs_lmrs"),
                prefer_hf_mirror=settings.model_download_prefer_hf_mirror,
                lock_timeout_seconds=settings.model_download_lock_timeout_seconds,
                log=lambda line: progress_fine_task(task.id, project.id, "weights_downloading", 10, started, line),
            )
        except ModelDownloadError as exc:
            raise FineFailure("MODEL_WEIGHT_DOWNLOAD_FAILED", str(exc)) from exc
    ensure_amb3r_weight(Path(settings.model_cache_dir))
    update_task(db, task, project, "weights_ready", 12, started, "AMB3R fine reconstruction weight is ready")


def update_task(db, task: Task, project: Project, stage: str, progress: int, started: float, *logs: str) -> None:
    task.current_stage = stage
    task.progress = max(task.progress or 0, progress)
    task.eta_seconds = estimate_fine_eta(task, started)
    if logs:
        task.logs = append_log(task.logs, *logs)
    emit(db, project.id, "task_progress", task_payload(task), task.id)
    db.commit()


def progress_fine_task(
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


def estimate_fine_eta(task: Task, started: float) -> int | None:
    progress = max(1, min(99, int(task.progress or 0)))
    if progress >= 100:
        return 0
    elapsed = max(1.0, time.monotonic() - started)
    expected = int((task.options or {}).get("fine_expected_seconds") or settings.fine_expected_seconds_images)
    by_progress = elapsed * (100 - progress) / progress
    by_expected = max(0, expected - elapsed)
    if progress < 20:
        return int(by_expected)
    return int(max(0, by_progress * 0.45 + by_expected * 0.55))


if __name__ == "__main__":
    main()
