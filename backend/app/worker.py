from __future__ import annotations

import os
import signal
import shutil
import socket
import sys
import time
import multiprocessing as mp
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
from app.task_control import task_cancel_requested

settings = get_settings()
storage = Storage(settings)
TASK_PROCESS_POLL_SECONDS = 2.0
TASK_PROCESS_TERM_TIMEOUT_SECONDS = 10.0


class TaskLogCapture:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._log = None
        self._saved_stdout_fd: int | None = None
        self._saved_stderr_fd: int | None = None
        self._stdout = None
        self._stderr = None
        self._old_stdout = None
        self._old_stderr = None

    def start(self) -> None:
        if self._log is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.path.open("ab", buffering=0)
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)
        os.dup2(self._log.fileno(), 1)
        os.dup2(self._log.fileno(), 2)
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        self._stdout = os.fdopen(os.dup(1), "w", encoding="utf-8", errors="replace", buffering=1)
        self._stderr = os.fdopen(os.dup(2), "w", encoding="utf-8", errors="replace", buffering=1)
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        print(f"[task-log] capturing stdout/stderr to {self.path}", flush=True)

    def flush(self) -> None:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        if self._log is not None:
            try:
                os.fsync(self._log.fileno())
            except Exception:
                pass

    def stop(self) -> None:
        if self._log is None:
            return
        self.flush()
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
        if self._old_stderr is not None:
            sys.stderr = self._old_stderr
        for stream in (self._stdout, self._stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        if self._saved_stdout_fd is not None:
            os.dup2(self._saved_stdout_fd, 1)
            os.close(self._saved_stdout_fd)
        if self._saved_stderr_fd is not None:
            os.dup2(self._saved_stderr_fd, 2)
            os.close(self._saved_stderr_fd)
        self._log.close()
        self._log = None


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
            run_task_in_subprocess(task_id, worker_id, redis_client, run_preview_task, "worker")
        except Exception as exc:
            print(f"[worker] unexpected failure for {task_id}: {exc}", flush=True)


def run_task_in_subprocess(
    task_id: str,
    worker_id: str,
    redis_client: redis.Redis,
    target: Callable[[str, str], None],
    label: str,
) -> None:
    if should_stop_task(redis_client, task_id):
        mark_task_canceled(task_id)
        return

    context = mp.get_context(os.getenv("WORKER_TASK_START_METHOD", "spawn"))
    process = context.Process(target=target, args=(task_id, worker_id))
    process.start()
    canceled = False
    try:
        while process.is_alive():
            process.join(TASK_PROCESS_POLL_SECONDS)
            heartbeat(worker_id, task_id)
            if process.is_alive() and should_stop_task(redis_client, task_id):
                canceled = True
                terminate_task_process(process, label, task_id)
                mark_task_canceled(task_id)
                break
        process.join()
        if not canceled and process.exitcode not in (0, None):
            print(f"[{label}] task process exited with code {process.exitcode} for {task_id}", flush=True)
            mark_task_process_failed(task_id, process.exitcode, label)
    finally:
        heartbeat(worker_id)


def terminate_task_process(process: mp.Process, label: str, task_id: str) -> None:
    print(f"[{label}] terminating canceled task process for {task_id}", flush=True)
    process.terminate()
    process.join(TASK_PROCESS_TERM_TIMEOUT_SECONDS)
    if process.is_alive():
        print(f"[{label}] killing unresponsive task process for {task_id}", flush=True)
        process.kill()
        process.join(TASK_PROCESS_TERM_TIMEOUT_SECONDS)


def should_stop_task(redis_client: redis.Redis, task_id: str) -> bool:
    try:
        if task_cancel_requested(redis_client, task_id):
            return True
    except Exception:
        pass
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        return task is None or task.status == "canceled"


def mark_task_canceled(task_id: str) -> None:
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        if not task:
            return
        project = db.get(Project, task.project_id)
        changed = task.status != "canceled"
        task.status = "canceled"
        task.finished_at = task.finished_at or utc_now()
        task.current_stage = "canceled"
        task.error_message = task.error_message or "用户取消任务"
        log_artifact = upload_task_log(db, project, task, task.worker_id or "unknown", task_source_version(task, project)) if project else None
        task.worker_id = None
        if log_artifact:
            task.logs = append_log(task.logs, f"uploaded {log_artifact.file_name}")
        task.logs = append_log(task.logs, "task process terminated after cancel request")
        if project:
            project.status = "CANCELED"
            project.error_message = task.error_message
        if changed and project:
            if log_artifact:
                emit(db, project.id, "artifact_created", {"artifact_id": log_artifact.id, "kind": log_artifact.kind}, task.id)
            emit(db, project.id, "task_failed", task_payload(task), task.id)
        db.commit()


def mark_task_process_failed(task_id: str, exitcode: int | None, label: str) -> None:
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        if not task or task.status not in {"queued", "running"}:
            return
        project = db.get(Project, task.project_id)
        message = process_exit_message(exitcode, label)
        if project:
            log_artifact = upload_task_log(db, project, task, task.worker_id or "unknown", task_source_version(task, project))
            if log_artifact:
                task.logs = append_log(task.logs, f"uploaded {log_artifact.file_name}")
                emit(db, project.id, "artifact_created", {"artifact_id": log_artifact.id, "kind": log_artifact.kind}, task.id)
        fail_task(db, task, project, "WORKER_PROCESS_EXITED", message)


def process_exit_message(exitcode: int | None, label: str) -> str:
    if exitcode is None:
        return f"{label} task process exited unexpectedly"
    details = f"{label} task process exited unexpectedly with code {exitcode}"
    signum = -exitcode if exitcode < 0 else exitcode - 128 if exitcode >= 128 else None
    if signum and signum > 0:
        if signum == 9:
            signal_name = "SIGKILL"
        else:
            try:
                signal_name = signal.Signals(signum).name
            except ValueError:
                signal_name = f"signal {signum}"
        details = f"{details} ({signal_name})"
        if signum == 9:
            details = f"{details}; possible OOM, GPU driver kill, or external termination"
    return details


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
        task.eta_seconds = estimate_eta(task, project, started)
        task.logs = append_log(task.logs, f"worker {worker_id} started task")
        project.status = "PREVIEW_RUNNING"
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

            pipeline = normalize_preview_pipeline((task.options or {}).get("preview_pipeline"), project.input_type)
            ensure_pipeline_weights(db, task, project, pipeline, started)
            input_dir = download_media(project, work_dir)
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
            source_version = task_source_version(task, project)
            ctx = PreviewContext(
                task_id=task.id,
                project_id=project.id,
                pipeline=pipeline,
                input_dir=input_dir,
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
                if not task or task.status == "canceled":
                    return
                project = upload_db.scalar(
                    select(Project)
                    .where(Project.id == task.project_id)
                    .options(selectinload(Project.media), selectinload(Project.artifacts))
                )
                update_task(upload_db, task, project, "uploading_artifact", 92, started, "validated non-empty preview.spz")
                preview_file_name = "preview.spz"
                key = storage_key("users", project.owner_id, "projects", project.id, "preview", preview_file_name)
                checksum = sha256_path(output_spz)
                uri = storage.upload_path(output_spz, key)
                artifact = Artifact(
                    project_id=project.id,
                    task_id=task.id,
                    kind="preview_spz",
                    object_uri=uri,
                    file_name=preview_file_name,
                    file_size=output_spz.stat().st_size,
                    checksum=checksum,
                    source_version=source_version,
                    metadata_json={
                        "pipeline": pipeline,
                        "source_version": source_version,
                        "generated_by": worker_id,
                        "adapter": result.metrics.get("adapter"),
                        "input_mode": project.input_type,
                        "source_commits": result.source_commits,
                        "splat_count": result.splat_count,
                        "intermediate_ply": str(result.intermediate_ply) if result.intermediate_ply else None,
                        "intermediate_ply_size": result.metrics.get("intermediate_ply_size"),
                        **preview_artifact_metrics(result.metrics),
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
                task.logs = append_log(task.logs, f"uploaded {preview_file_name}")
                if ply_artifact:
                    task.logs = append_log(task.logs, "uploaded original.ply")
                log_capture.flush()
                log_artifact = upload_task_log(upload_db, project, task, worker_id, source_version)
                if log_artifact:
                    task.logs = append_log(task.logs, f"uploaded {log_artifact.file_name}")
                project.status = "PREVIEW_READY"
                project.preview_source_version = artifact.source_version
                project.error_message = None
                emit(upload_db, project.id, "artifact_created", {"artifact_id": artifact.id, "kind": artifact.kind}, task.id)
                if ply_artifact:
                    emit(upload_db, project.id, "artifact_created", {"artifact_id": ply_artifact.id, "kind": ply_artifact.kind}, task.id)
                if log_artifact:
                    emit(upload_db, project.id, "artifact_created", {"artifact_id": log_artifact.id, "kind": log_artifact.kind}, task.id)
                emit(upload_db, project.id, "task_succeeded", task_payload(task), task.id)
                upload_db.commit()
    except PreviewFailure as exc:
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
            fail_task(db, task, project, "ALGORITHM_EXECUTION_FAILED", str(exc))
    finally:
        heartbeat(worker_id)
        log_capture.stop()


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


def download_media(project: Project, work_dir: Path) -> Path:
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    media_items = list(project.media)
    for index, media in enumerate(media_items):
        suffix = Path(media.file_name).suffix or (".jpg" if media.kind == "image" else ".mp4")
        target = input_dir / f"{index:06d}-{media.id}{suffix}"
        storage.download_to_path(media.object_uri, target)
    return input_dir


def task_log_path(task_id: str) -> Path:
    return settings.local_work_root / task_id / "task.log"


def task_log_file_name(task: Task) -> str:
    return "fine-task.log" if task.type == "fine" else "preview-task.log"


def task_source_version(task: Task, project: Project | None) -> int:
    try:
        return int((task.options or {}).get("source_version") or (project.source_version if project else 0) or 0)
    except (TypeError, ValueError):
        return int(project.source_version if project else 0)


def upload_task_log(
    db,
    project: Project,
    task: Task,
    worker_id: str,
    source_version: int,
    log_path: Path | None = None,
) -> Artifact | None:
    path = log_path or task_log_path(task.id)
    if not path.exists() or path.stat().st_size <= 0:
        return None
    existing = db.scalar(select(Artifact).where(Artifact.task_id == task.id, Artifact.kind == "task_log"))
    if existing:
        return None
    file_name = task_log_file_name(task)
    key = storage_key("users", project.owner_id, "projects", project.id, "tasks", task.id, file_name)
    uri = storage.upload_path(path, key)
    artifact = Artifact(
        project_id=project.id,
        task_id=task.id,
        kind="task_log",
        object_uri=uri,
        file_name=file_name,
        file_size=path.stat().st_size,
        checksum=sha256_path(path),
        source_version=source_version,
        metadata_json={
            "artifact": "task_log",
            "task_type": task.type,
            "source_version": source_version,
            "generated_by": worker_id,
        },
    )
    db.add(artifact)
    db.flush()
    return artifact


def read_positive_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def read_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def preview_artifact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "lingbot_commit",
        "lingbot_model",
        "lingbot_sampled_frames",
        "lingbot_source_fps",
        "lingbot_sampled_fps",
        "lingbot_frame_width",
        "lingbot_frame_height",
        "lingbot_image_size",
        "lingbot_inference_mode",
        "lingbot_keyframe_interval",
        "lingbot_camera_iterations",
        "lingbot_num_scale_frames",
        "lingbot_window_size",
        "lingbot_overlap_keyframes",
        "lingbot_use_sdpa",
        "lingbot_compile",
        "lingbot_compile_requested",
        "lingbot_compile_cudagraphs",
        "lingbot_compile_fallback",
        "lingbot_max_frames",
        "lingbot_frame_stride",
        "lingbot_pixel_stride",
        "lingbot_conf_percentile",
        "lingbot_min_conf",
        "lingbot_max_points",
        "lingbot_save_predictions",
        "lingbot_predictions_dir",
        "lingbot_point_source",
        "lingbot_ply_format",
        "lingbot_points_before_downsample",
        "lingbot_points_after_downsample",
        "point_count",
        "cuda_memory_peak_mb",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


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
    return expected_seconds_for_pipeline(pipeline)


def expected_seconds_for_pipeline(pipeline: str | None) -> int:
    if pipeline == "litevggt_spz":
        return settings.preview_expected_seconds_litevggt_spz
    if pipeline == "lingbot_map_spz":
        return settings.preview_expected_seconds_lingbot_map_spz
    return settings.preview_expected_seconds_litevggt_spz


def stage_for_pipeline(pipeline: str) -> str:
    if pipeline == "litevggt_spz":
        return "litevggt_direct_spz"
    if pipeline == "lingbot_map_spz":
        return "lingbot_map_spz"
    return "unknown_preview_pipeline"


if __name__ == "__main__":
    main()
