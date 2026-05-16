from __future__ import annotations

import asyncio
import hashlib
import io
import json
import mimetypes
import os
import secrets
import shutil
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session, selectinload

from app.algorithms import license_notice_for, normalize_preview_pipeline, runtime_preflight, seed_algorithm_registry
from app.config import get_settings
from app.database import SessionLocal, engine, get_db, initialize_database_schema
from app.fine.fastgs_defaults import DEFAULT_FINE_SCENE_PROFILE, FINE_SCENE_PROFILE_MAX_SIDES
from app.models import AlgorithmRegistry, Artifact, Feedback, MediaAsset, Project, Task, TaskEvent, UploadSession, User, new_id, utc_now
from app.resources import collect_resources
from app.security import (
    create_access_token,
    create_artifact_token,
    get_current_user,
    hash_password,
    require_admin,
    verify_artifact_token,
    verify_password,
)
from app.storage import Storage, safe_filename, storage_key
from app.task_control import request_task_cancel

settings = get_settings()
storage = Storage(settings)
app = FastAPI(title=settings.app_name)
MAX_UPLOAD_CHUNK_SIZE = 64 * 1024 * 1024
UPLOAD_COMPLETE_LOCK_TIMEOUT_SECONDS = 30

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Range", "Accept-Ranges"],
)


class RegisterPayload(BaseModel):
    username: str
    password: str
    email: str | None = None


class LoginPayload(BaseModel):
    username: str
    password: str


class ProjectCreatePayload(BaseModel):
    name: str
    input_type: str
    tags: list[str] = []


class ProjectBulkDeletePayload(BaseModel):
    project_ids: list[str]


class TaskCreatePayload(BaseModel):
    options: dict[str, Any] = {}


DEFAULT_PREVIEW_SCENE_PROFILE = "mixed_balanced"
PREVIEW_SCENE_PROFILES = {"mixed_balanced", "indoor_full", "outdoor_fast_clean"}
FINE_SCENE_PROFILES = set(FINE_SCENE_PROFILE_MAX_SIDES)


class FeedbackPayload(BaseModel):
    title: str
    content: str
    project_id: str | None = None


class UploadCheckPayload(BaseModel):
    file_name: str
    file_size: int
    chunk_size: int
    total_chunks: int
    client_order: int | None = None
    file_hash: str | None = None
    file_signature: str | None = None
    content_type: str | None = None


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def user_dict(user: User) -> dict[str, Any]:
    return {"id": user.id, "username": user.username, "email": user.email, "role": user.role, "created_at": iso(user.created_at)}


def auth_response(user: User) -> dict[str, Any]:
    return {"access_token": create_access_token(user.id), "token_type": "bearer", "user": user_dict(user)}


def media_dict(item: MediaAsset) -> dict[str, Any]:
    return {
        "id": item.id,
        "project_id": item.project_id,
        "kind": item.kind,
        "object_uri": item.object_uri,
        "thumbnail_uri": item.thumbnail_uri,
        "file_name": item.file_name,
        "file_size": item.file_size,
        "width": item.width,
        "height": item.height,
        "duration_seconds": item.duration_seconds,
        "quality_flags": item.quality_flags or {},
        "source_version": item.source_version,
        "client_order": item.client_order,
        "created_at": iso(item.created_at),
    }


def task_dict(task: Task, project_name: str | None = None) -> dict[str, Any]:
    project = task.__dict__.get("project")
    resolved_project_name = project_name if project_name is not None else project.name if project else None
    return {
        "id": task.id,
        "project_id": task.project_id,
        "project_name": resolved_project_name,
        "type": task.type,
        "status": task.status,
        "priority": task.priority,
        "progress": task.progress,
        "worker_id": task.worker_id,
        "options": task.options or {},
        "current_stage": task.current_stage,
        "eta_seconds": task.eta_seconds,
        "error_code": task.error_code,
        "error_message": task.error_message,
        "metrics": task.metrics or {},
        "logs": task.logs or [],
        "created_at": iso(task.created_at),
        "started_at": iso(task.started_at),
        "finished_at": iso(task.finished_at),
    }


def artifact_dict(item: Artifact) -> dict[str, Any]:
    return {
        "id": item.id,
        "project_id": item.project_id,
        "task_id": item.task_id,
        "kind": item.kind,
        "object_uri": item.object_uri,
        "file_name": item.file_name,
        "file_size": item.file_size,
        "checksum": item.checksum,
        "metadata": item.metadata_json or {},
        "source_version": item.source_version,
        "created_at": iso(item.created_at),
    }


def is_ply_artifact(item: Artifact) -> bool:
    kind = (item.kind or "").lower()
    file_name = (item.file_name or "").lower()
    object_uri = (item.object_uri or "").lower()
    return kind == "ply" or kind.endswith("_ply") or file_name.endswith(".ply") or object_uri.endswith(".ply")


def intermediate_ply_path(item: Artifact) -> Path | None:
    path_value = (item.metadata_json or {}).get("intermediate_ply")
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value).expanduser().resolve()
    if path.suffix.lower() != ".ply" or not path.is_file():
        return None
    return path


def project_dict(project: Project, include_children: bool = False) -> dict[str, Any]:
    payload = {
        "id": project.id,
        "owner_id": project.owner_id,
        "name": project.name,
        "input_type": project.input_type,
        "status": project.status,
        "tags": project.tags or [],
        "total_size_bytes": project.total_size_bytes,
        "preview_image_uri": project.preview_image_uri,
        "error_message": project.error_message,
        "source_version": project.source_version,
        "preview_source_version": project.preview_source_version,
        "created_at": iso(project.created_at),
        "updated_at": iso(project.updated_at),
    }
    if include_children:
        payload["media"] = [media_dict(item) for item in project.media]
        payload["tasks"] = [task_dict(item, project.name) for item in project.tasks]
        payload["artifacts"] = [artifact_dict(item) for item in project.artifacts]
    return payload


def emit_event(db: Session, project_id: str, event: str, payload: dict[str, Any], task_id: str | None = None) -> None:
    db.add(TaskEvent(project_id=project_id, task_id=task_id, event=event, payload=payload))


def owned_project(db: Session, project_id: str, user: User, include_children: bool = False) -> Project:
    statement = select(Project).where(Project.id == project_id)
    if include_children:
        statement = statement.options(selectinload(Project.media), selectinload(Project.tasks), selectinload(Project.artifacts))
    project = db.scalar(statement)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if user.role != "admin" and project.owner_id != user.id:
        raise HTTPException(status_code=403, detail="Project access denied")
    return project


def delete_project_record(db: Session, project: Project) -> None:
    for task in project.tasks:
        if task.status in {"queued", "running"}:
            request_worker_cancel(task)
    for media in project.media:
        storage.delete(media.object_uri)
        storage.delete(media.thumbnail_uri)
    for artifact in project.artifacts:
        storage.delete(artifact.object_uri)
    db.delete(project)


def media_key(project: Project, media_id: str, file_name: str, kind: str) -> str:
    folder = "images" if kind == "image" else "video"
    return storage_key("users", project.owner_id, "projects", project.id, "raw", folder, f"{media_id}-{safe_filename(file_name)}")


def thumbnail_key(project: Project, media_id: str) -> str:
    return storage_key("users", project.owner_id, "projects", project.id, "thumbs", f"{media_id}.jpg")


def create_thumbnail(uri: str, project: Project, media_id: str) -> tuple[str | None, int | None, int | None]:
    try:
        from PIL import Image, ImageOps
        from app.preview.image_preprocess import convert_to_rgb, register_optional_heif_support

        register_optional_heif_support()
        with storage.open_file(uri) as handle:
            image = Image.open(handle)
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            image.thumbnail((420, 420))
            image = convert_to_rgb(image)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=86)
            thumb_uri = storage.write_bytes(thumbnail_key(project, media_id), output.getvalue())
            return thumb_uri, width, height
    except Exception:
        return None, None, None


def create_video_thumbnail(uri: str, project: Project, media_id: str) -> tuple[str | None, int | None, int | None, float | None]:
    temp_path: Path | None = None
    cap = None
    try:
        import cv2
        from PIL import Image

        resolved_uri = storage.resolve_existing_uri(uri)
        if resolved_uri.startswith("local://"):
            video_path = storage.local_path(resolved_uri)
        else:
            suffix = Path(storage.key_from_uri(resolved_uri)).suffix or ".video"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = Path(tmp.name)
            storage.download_to_path(resolved_uri, temp_path)
            video_path = temp_path

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None, None, None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        duration = frame_count / fps if frame_count > 0 and fps > 0 else None
        if frame_count > 3:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_count - 1, max(1, frame_count // 3)))

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None, None, None, duration

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        width, height = image.size
        image.thumbnail((420, 420), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=86)
        thumb_uri = storage.write_bytes(thumbnail_key(project, media_id), output.getvalue())
        return thumb_uri, width, height, duration
    except Exception:
        return None, None, None, None
    finally:
        if cap is not None:
            cap.release()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def media_kind_for_upload(file_name: str, content_type: str | None) -> str:
    content_type = content_type or mimetypes.guess_type(file_name)[0] or ""
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("video/"):
        return "video"
    suffix = Path(file_name).suffix.lower()
    return "image" if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"} else "video"


def validate_media_kind(project: Project, kind: str) -> None:
    if project.input_type == "images" and kind != "image":
        raise HTTPException(status_code=400, detail="图片项目只能上传图片")
    if project.input_type == "video" and kind != "video":
        raise HTTPException(status_code=400, detail="视频项目只能上传视频")


def add_media_asset(
    db: Session,
    project: Project,
    file_name: str,
    kind: str,
    uri: str,
    size: int,
    media_id: str | None = None,
    client_order: int = 0,
) -> MediaAsset:
    media = MediaAsset(project_id=project.id, id=media_id or new_id(), kind=kind, object_uri=uri, file_name=file_name, file_size=size, client_order=client_order)
    project.total_size_bytes += size
    project.source_version += 1
    media.source_version = project.source_version
    project.status = "UPLOADING"
    project.updated_at = utc_now()
    if kind == "image":
        thumb_uri, width, height = create_thumbnail(uri, project, media.id)
        media.thumbnail_uri = thumb_uri
        media.width = width
        media.height = height
        if thumb_uri and not project.preview_image_uri:
            project.preview_image_uri = media.thumbnail_uri
    else:
        thumb_uri, width, height, duration = create_video_thumbnail(uri, project, media.id)
        media.thumbnail_uri = thumb_uri
        media.width = width
        media.height = height
        media.duration_seconds = duration
        if thumb_uri and not project.preview_image_uri:
            project.preview_image_uri = media.thumbnail_uri
    db.add(media)
    return media


def validate_upload_payload(payload: UploadCheckPayload) -> None:
    if payload.file_size <= 0:
        raise HTTPException(status_code=400, detail="file_size must be positive")
    if payload.chunk_size <= 0:
        raise HTTPException(status_code=400, detail="chunk_size must be positive")
    if payload.client_order is not None and payload.client_order < 0:
        raise HTTPException(status_code=400, detail="client_order must be non-negative")
    if payload.chunk_size > MAX_UPLOAD_CHUNK_SIZE:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="chunk_size exceeds 64MB limit")
    expected_chunks = (payload.file_size + payload.chunk_size - 1) // payload.chunk_size
    if payload.total_chunks != expected_chunks:
        raise HTTPException(status_code=400, detail="total_chunks does not match file_size and chunk_size")
    if payload.file_hash is not None:
        if len(payload.file_hash) != 64 or any(char not in "0123456789abcdef" for char in payload.file_hash.lower()):
            raise HTTPException(status_code=400, detail="file_hash must be a SHA-256 hex digest")
        return
    if not payload.file_signature or len(payload.file_signature) > 512:
        raise HTTPException(status_code=400, detail="file_hash or file_signature is required")


def normalized_client_order(value: int | None) -> int:
    if value is None:
        return 0
    if value < 0:
        raise HTTPException(status_code=400, detail="client_order must be non-negative")
    return value


def upload_session_key(payload: UploadCheckPayload) -> str:
    if payload.file_hash:
        return payload.file_hash.lower()
    signature = payload.file_signature or ""
    return "sig-" + hashlib.sha256(signature.encode("utf-8")).hexdigest()


def upload_has_strong_hash(payload: UploadCheckPayload) -> bool:
    return payload.file_hash is not None


def is_sha256_hex(value: str | None) -> bool:
    return bool(value) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def upload_session_dir(session: UploadSession) -> Path:
    return settings.local_work_root / "uploads" / session.user_id / session.project_id / session.file_hash


def chunk_path(session: UploadSession, chunk_index: int) -> Path:
    return upload_session_dir(session) / f"{chunk_index:08d}.part"


def expected_chunk_size(session: UploadSession, chunk_index: int) -> int:
    if chunk_index < 0 or chunk_index >= session.total_chunks:
        raise HTTPException(status_code=404, detail="Chunk index out of range")
    if chunk_index == session.total_chunks - 1:
        return session.file_size - session.chunk_size * (session.total_chunks - 1)
    return session.chunk_size


def uploaded_chunk_indexes(session: UploadSession) -> list[int]:
    root = upload_session_dir(session)
    if not root.exists():
        return []
    indexes: list[int] = []
    for index in range(session.total_chunks):
        path = chunk_path(session, index)
        if path.exists() and path.stat().st_size == expected_chunk_size(session, index):
            indexes.append(index)
    return indexes


def owned_upload_session(db: Session, upload_id: str, user: User) -> UploadSession:
    session = db.get(UploadSession, upload_id)
    if not session:
        raise HTTPException(status_code=404, detail="Upload session not found")
    if user.role != "admin" and session.user_id != user.id:
        raise HTTPException(status_code=403, detail="Upload session access denied")
    return session


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def upload_complete_lock(session: UploadSession):
    root = upload_session_dir(session)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".complete.lock"
    deadline = time.monotonic() + UPLOAD_COMPLETE_LOCK_TIMEOUT_SECONDS
    handle: int | None = None
    while handle is None:
        try:
            handle = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise HTTPException(status_code=409, detail="Upload is already being completed")
            time.sleep(0.2)
    try:
        os.write(handle, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(handle)
        lock_path.unlink(missing_ok=True)


def first_media_thumbnail(project: Project, exclude_media_id: str | None = None) -> str | None:
    for item in project.media:
        if item.id != exclude_media_id and item.thumbnail_uri:
            return item.thumbnail_uri
    return None


def get_redis() -> redis.Redis:
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def enqueue_preview_task(task_id: str) -> None:
    get_redis().rpush(settings.preview_queue_name, task_id)


def enqueue_fine_task(task_id: str) -> None:
    get_redis().rpush(settings.fine_queue_name, task_id)


def normalize_preview_scene_profile(value: Any) -> str:
    profile = str(value or DEFAULT_PREVIEW_SCENE_PROFILE).strip().lower()
    if profile not in PREVIEW_SCENE_PROFILES:
        raise HTTPException(status_code=400, detail=f"Unsupported preview scene profile: {profile}")
    return profile


def normalize_fine_scene_profile(value: Any) -> str:
    profile = str(value or DEFAULT_FINE_SCENE_PROFILE).strip().lower()
    if profile not in FINE_SCENE_PROFILES:
        raise HTTPException(status_code=400, detail=f"Unsupported fine scene profile: {profile}")
    return profile


def request_worker_cancel(task: Task) -> None:
    try:
        request_task_cancel(get_redis(), task.id, task.type)
    except Exception:
        pass


def seed_database(db: Session) -> None:
    admin = db.scalar(select(User).where(User.username == settings.admin_username))
    if not admin:
        db.add(
            User(
                username=settings.admin_username,
                email=settings.admin_email,
                password_hash=hash_password(settings.admin_password),
                role="admin",
            )
        )
    seed_algorithm_registry(db, settings)
    db.commit()


def ensure_upload_session_schema() -> None:
    inspector = inspect(engine)
    if "upload_sessions" not in inspector.get_table_names():
        return
    columns = {item["name"] for item in inspector.get_columns("upload_sessions")}
    with engine.begin() as connection:
        if "error_message" not in columns:
            connection.execute(text("ALTER TABLE upload_sessions ADD COLUMN error_message TEXT"))
        if "client_order" not in columns:
            connection.execute(text("ALTER TABLE upload_sessions ADD COLUMN client_order INTEGER DEFAULT 0"))


def ensure_media_asset_order_schema() -> None:
    inspector = inspect(engine)
    if "media_assets" not in inspector.get_table_names():
        return
    columns = {item["name"] for item in inspector.get_columns("media_assets")}
    if "client_order" in columns:
        return
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE media_assets ADD COLUMN client_order INTEGER DEFAULT 0"))


def ensure_algorithm_registry_schema() -> None:
    if engine.dialect.name != "postgresql":
        return
    inspector = inspect(engine)
    if "algorithm_registry" not in inspector.get_table_names():
        return
    columns = {item["name"]: item for item in inspector.get_columns("algorithm_registry")}
    license_column = columns.get("license")
    if not license_column or getattr(license_column["type"], "length", None) is None:
        return
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE algorithm_registry ALTER COLUMN license TYPE TEXT"))


def ensure_project_share_schema() -> None:
    inspector = inspect(engine)
    if "projects" not in inspector.get_table_names():
        return
    columns = {item["name"] for item in inspector.get_columns("projects")}
    if "share_token" not in columns:
        with engine.begin() as connection:
            column_type = "VARCHAR(64)" if engine.dialect.name == "postgresql" else "VARCHAR(64)"
            connection.execute(text(f"ALTER TABLE projects ADD COLUMN share_token {column_type}"))
    index_names = {item["name"] for item in inspector.get_indexes("projects")}
    if "ix_projects_share_token" not in index_names:
        with engine.begin() as connection:
            connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_projects_share_token ON projects (share_token)"))


@app.on_event("startup")
def startup() -> None:
    initialize_database_schema()
    ensure_upload_session_schema()
    ensure_media_asset_order_schema()
    ensure_algorithm_registry_schema()
    ensure_project_share_schema()
    storage.ensure_bucket()
    with SessionLocal() as db:
        seed_database(db)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": settings.app_name}


@app.post("/api/auth/register")
def register(payload: RegisterPayload, db: Session = Depends(get_db)) -> dict[str, Any]:
    username = payload.username.strip()
    if len(username) < 2 or len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="用户名至少 2 位，密码至少 6 位")
    if db.scalar(select(User).where(User.username == username)):
        raise HTTPException(status_code=409, detail="用户名已存在")
    user = User(username=username, email=payload.email, password_hash=hash_password(payload.password), role="user")
    db.add(user)
    db.commit()
    db.refresh(user)
    return auth_response(user)


@app.post("/api/auth/login")
def login(payload: LoginPayload, db: Session = Depends(get_db)) -> dict[str, Any]:
    user = db.scalar(select(User).where(User.username == payload.username.strip()))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return auth_response(user)


@app.post("/api/auth/logout")
def logout() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/me")
def me(user: User = Depends(get_current_user)) -> dict[str, Any]:
    return user_dict(user)


@app.get("/api/system/resources")
def system_resources(_: User = Depends(get_current_user)) -> dict[str, Any]:
    return collect_resources()


@app.get("/api/admin/system/resources")
def admin_resources(_: User = Depends(require_admin)) -> dict[str, Any]:
    return collect_resources()


@app.get("/api/projects/summary")
def project_summary(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, int]:
    statement = select(Project).where(Project.owner_id == user.id)
    projects = db.scalars(statement).all()
    active = {"PREVIEW_RUNNING", "FINE_QUEUED", "FINE_RUNNING", "PREPROCESSING", "UPLOADING"}
    return {
        "total": len(projects),
        "running": sum(1 for item in projects if item.status in active),
        "completed": sum(1 for item in projects if item.status == "COMPLETED"),
        "failed": sum(1 for item in projects if item.status == "FAILED"),
        "total_size_bytes": sum(item.total_size_bytes for item in projects),
    }


@app.get("/api/projects")
def list_projects(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    projects = db.scalars(
        select(Project)
        .where(Project.owner_id == user.id)
        .options(selectinload(Project.media), selectinload(Project.tasks), selectinload(Project.artifacts))
        .order_by(Project.updated_at.desc())
    ).all()
    return {"projects": [project_dict(item, include_children=True) for item in projects]}


@app.post("/api/projects")
def create_project(payload: ProjectCreatePayload, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    if payload.input_type not in {"images", "video"}:
        raise HTTPException(status_code=400, detail="Unsupported input_type")
    project = Project(
        owner_id=user.id,
        name=payload.name.strip() or "新建重建项目",
        input_type=payload.input_type,
        tags=payload.tags,
        status="CREATED",
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project_dict(project, include_children=True)


@app.get("/api/projects/{project_id}")
def get_project(project_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    return project_dict(owned_project(db, project_id, user, include_children=True), include_children=True)


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, bool]:
    project = owned_project(db, project_id, user, include_children=True)
    delete_project_record(db, project)
    db.commit()
    return {"deleted": True}


@app.post("/api/projects/bulk-delete")
def bulk_delete_projects(payload: ProjectBulkDeletePayload, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    project_ids = list(dict.fromkeys(item for item in payload.project_ids if item))
    if not project_ids:
        raise HTTPException(status_code=400, detail="project_ids is required")
    projects = [owned_project(db, project_id, user, include_children=True) for project_id in project_ids]
    for project in projects:
        delete_project_record(db, project)
    db.commit()
    return {"deleted": len(projects), "project_ids": [project.id for project in projects]}


@app.post("/api/projects/{project_id}/media")
async def upload_media(
    project_id: str,
    client_order: int | None = Query(None),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    project = owned_project(db, project_id, user)
    content_type = file.content_type or mimetypes.guess_type(file.filename or "")[0] or ""
    kind = "image" if content_type.startswith("image/") else "video" if content_type.startswith("video/") else None
    if kind is None:
        suffix = Path(file.filename or "").suffix.lower()
        kind = "image" if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"} else "video"
    if project.input_type == "images" and kind != "image":
        raise HTTPException(status_code=400, detail="图片项目只能上传图片")
    if project.input_type == "video" and kind != "video":
        raise HTTPException(status_code=400, detail="视频项目只能上传视频")

    requested_client_order = normalized_client_order(client_order)
    media_id = new_id()
    file_name = safe_filename(file.filename or "upload.bin")
    uri, size = await storage.save_upload(file, media_key(project, media_id, file_name, kind))
    media = MediaAsset(
        project_id=project.id,
        id=media_id,
        kind=kind,
        object_uri=uri,
        file_name=file_name,
        file_size=size,
        client_order=requested_client_order,
    )
    project.total_size_bytes += size
    project.source_version += 1
    media.source_version = project.source_version
    project.status = "UPLOADING"
    project.updated_at = utc_now()
    if kind == "image":
        thumb_uri, width, height = create_thumbnail(uri, project, media.id)
        media.thumbnail_uri = thumb_uri
        media.width = width
        media.height = height
        if thumb_uri and not project.preview_image_uri:
            project.preview_image_uri = media.thumbnail_uri
    else:
        thumb_uri, width, height, duration = create_video_thumbnail(uri, project, media.id)
        media.thumbnail_uri = thumb_uri
        media.width = width
        media.height = height
        media.duration_seconds = duration
        if thumb_uri and not project.preview_image_uri:
            project.preview_image_uri = media.thumbnail_uri
    db.add(media)
    db.commit()
    db.refresh(media)
    return media_dict(media)


@app.post("/api/projects/{project_id}/uploads/check")
def check_upload(
    project_id: str,
    payload: UploadCheckPayload,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    validate_upload_payload(payload)
    project = owned_project(db, project_id, user)
    file_name = safe_filename(payload.file_name or "upload.bin")
    kind = media_kind_for_upload(file_name, payload.content_type)
    validate_media_kind(project, kind)
    session_key = upload_session_key(payload)
    has_strong_hash = upload_has_strong_hash(payload)
    client_order = normalized_client_order(payload.client_order)

    if has_strong_hash:
        completed_session = db.scalar(
            select(UploadSession)
            .where(
                UploadSession.project_id == project.id,
                UploadSession.user_id == user.id,
                UploadSession.file_hash == session_key,
                UploadSession.status == "completed",
            )
            .order_by(UploadSession.updated_at.desc())
        )
        if completed_session and completed_session.media_id:
            media = db.get(MediaAsset, completed_session.media_id)
            if media:
                return {"upload_id": completed_session.id, "uploaded_chunks": list(range(completed_session.total_chunks)), "completed": True, "media": media_dict(media)}

    session = db.scalar(
        select(UploadSession)
        .where(
            UploadSession.project_id == project.id,
            UploadSession.user_id == user.id,
            UploadSession.file_hash == session_key,
            UploadSession.status == "uploading",
        )
        .order_by(UploadSession.updated_at.desc())
    )
    if not session:
        session = UploadSession(
            project_id=project.id,
            user_id=user.id,
            file_hash=session_key,
            file_name=file_name,
            file_size=payload.file_size,
            chunk_size=payload.chunk_size,
            total_chunks=payload.total_chunks,
            content_type=payload.content_type,
            kind=kind,
            client_order=client_order,
            status="uploading",
            error_message=None,
        )
        db.add(session)
        db.commit()
        db.refresh(session)
    elif session.file_size != payload.file_size or session.chunk_size != payload.chunk_size or session.total_chunks != payload.total_chunks:
        raise HTTPException(status_code=409, detail="Existing upload session metadata does not match")
    elif session.client_order != client_order:
        session.client_order = client_order
        session.updated_at = utc_now()
        db.commit()
    elif session.status == "failed":
        session.status = "uploading"
        session.error_message = None
        session.updated_at = utc_now()
        db.commit()
    return {"upload_id": session.id, "uploaded_chunks": uploaded_chunk_indexes(session), "completed": False}


@app.put("/api/uploads/{upload_id}/chunks/{chunk_index}")
async def upload_chunk(
    upload_id: str,
    chunk_index: int,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = owned_upload_session(db, upload_id, user)
    if session.status == "completed":
        return {"chunk_index": chunk_index, "uploaded_chunks": uploaded_chunk_indexes(session)}
    if session.status == "failed":
        raise HTTPException(status_code=409, detail=session.error_message or "Upload session failed")
    expected_size = expected_chunk_size(session, chunk_index)
    if expected_size > MAX_UPLOAD_CHUNK_SIZE:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Chunk exceeds 64MB limit")
    root = upload_session_dir(session)
    root.mkdir(parents=True, exist_ok=True)
    target = chunk_path(session, chunk_index)
    temp_path = root / f"{target.name}.{new_id()}.tmp"
    size = 0
    try:
        with temp_path.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                handle.write(chunk)
        if size != expected_size:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Chunk size does not match expected size")
        temp_path.replace(target)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    session.updated_at = utc_now()
    db.commit()
    return {"chunk_index": chunk_index, "uploaded_chunks": uploaded_chunk_indexes(session)}


@app.put("/api/uploads/{upload_id}/chunks/{chunk_index}/raw")
async def upload_chunk_raw(
    upload_id: str,
    chunk_index: int,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    session = owned_upload_session(db, upload_id, user)
    if session.status == "completed":
        return {"chunk_index": chunk_index, "uploaded_chunks": uploaded_chunk_indexes(session)}
    if session.status == "failed":
        raise HTTPException(status_code=409, detail=session.error_message or "Upload session failed")
    expected_size = expected_chunk_size(session, chunk_index)
    if expected_size > MAX_UPLOAD_CHUNK_SIZE:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Chunk exceeds 64MB limit")
    root = upload_session_dir(session)
    root.mkdir(parents=True, exist_ok=True)
    target = chunk_path(session, chunk_index)
    temp_path = root / f"{target.name}.{new_id()}.tmp"
    size = 0
    try:
        with temp_path.open("wb") as handle:
            async for chunk in request.stream():
                if not chunk:
                    continue
                size += len(chunk)
                if size > expected_size:
                    temp_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=400, detail="Chunk size does not match expected size")
                handle.write(chunk)
        if size != expected_size:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Chunk size does not match expected size")
        temp_path.replace(target)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    session.updated_at = utc_now()
    db.commit()
    return {"chunk_index": chunk_index, "uploaded_chunks": uploaded_chunk_indexes(session)}


@app.post("/api/uploads/{upload_id}/complete")
def complete_upload(upload_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    session = owned_upload_session(db, upload_id, user)
    if session.status == "completed" and session.media_id:
        media = db.get(MediaAsset, session.media_id)
        if media:
            return {"media": media_dict(media)}
    if session.status == "failed":
        raise HTTPException(status_code=409, detail=session.error_message or "Upload session failed")
    project = owned_project(db, session.project_id, user)
    indexes = uploaded_chunk_indexes(session)
    if len(indexes) != session.total_chunks:
        raise HTTPException(status_code=400, detail="Upload chunks are incomplete")

    merge_root = upload_session_dir(session)
    merged_path = merge_root / f"merged-{new_id()}-{session.file_name}"
    try:
        with upload_complete_lock(session):
            db.refresh(session)
            if session.status == "completed" and session.media_id:
                media = db.get(MediaAsset, session.media_id)
                if media:
                    return {"media": media_dict(media)}
            with merged_path.open("wb") as out:
                for index in range(session.total_chunks):
                    with chunk_path(session, index).open("rb") as handle:
                        shutil.copyfileobj(handle, out)
            if merged_path.stat().st_size != session.file_size:
                session.status = "failed"
                session.error_message = "Merged file size does not match expected size"
                session.updated_at = utc_now()
                db.commit()
                raise HTTPException(status_code=400, detail=session.error_message)
            merged_hash = sha256_file(merged_path)
            if is_sha256_hex(session.file_hash) and merged_hash != session.file_hash:
                session.status = "failed"
                session.error_message = "Merged file SHA-256 does not match expected hash"
                session.updated_at = utc_now()
                db.commit()
                raise HTTPException(status_code=400, detail=session.error_message)
            media_id = new_id()
            uri = storage.upload_path(merged_path, media_key(project, media_id, session.file_name, session.kind))
            media = add_media_asset(db, project, session.file_name, session.kind, uri, session.file_size, media_id, session.client_order)
            session.status = "completed"
            session.object_uri = uri
            session.media_id = media.id
            session.file_hash = merged_hash
            session.error_message = None
            session.updated_at = utc_now()
            db.commit()
            db.refresh(media)
            shutil.rmtree(merge_root, ignore_errors=True)
            return {"media": media_dict(media)}
    finally:
        if merged_path.exists():
            merged_path.unlink(missing_ok=True)


@app.get("/api/projects/{project_id}/media")
def list_media(project_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    project = owned_project(db, project_id, user, include_children=True)
    return {"media": [media_dict(item) for item in project.media]}


@app.get("/api/projects/{project_id}/media/stats")
def media_stats(project_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    project = owned_project(db, project_id, user, include_children=True)
    media = project.media
    return {
        "image_count": sum(1 for item in media if item.kind == "image"),
        "video_count": sum(1 for item in media if item.kind == "video"),
        "total_size_bytes": sum(item.file_size for item in media),
        "source_version": project.source_version,
    }


@app.delete("/api/projects/{project_id}/media/{media_id}")
def delete_media(project_id: str, media_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    project = owned_project(db, project_id, user, include_children=True)
    media = next((item for item in project.media if item.id == media_id), None)
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")
    storage.delete(media.object_uri)
    storage.delete(media.thumbnail_uri)
    project.total_size_bytes = max(0, project.total_size_bytes - media.file_size)
    project.source_version += 1
    project.updated_at = utc_now()
    if project.preview_image_uri == media.thumbnail_uri:
        project.preview_image_uri = first_media_thumbnail(project, exclude_media_id=media.id)
    if not [item for item in project.media if item.id != media_id]:
        project.status = "CREATED"
    elif project.preview_source_version is not None and project.preview_source_version != project.source_version:
        project.status = "PREPROCESSING"
    db.delete(media)
    db.commit()
    return {"deleted": True, "source_version": project.source_version}


@app.post("/api/projects/{project_id}/tasks/preview")
def create_preview_task(
    project_id: str,
    payload: TaskCreatePayload,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    project = owned_project(db, project_id, user, include_children=True)
    if not project.media:
        raise HTTPException(status_code=400, detail="请先上传真实素材")
    if project.input_type == "images" and not any(item.kind == "image" for item in project.media):
        raise HTTPException(status_code=400, detail="图片预览至少需要 1 张图片")
    payload_options = payload.options or {}
    pipeline = normalize_preview_pipeline(str(payload_options.get("preview_pipeline") or ""), project.input_type)
    options = {**payload_options, "preview_pipeline": pipeline, "source_version": project.source_version}
    if project.input_type == "images":
        if pipeline != "litevggt_spz":
            raise HTTPException(status_code=400, detail=f"Unsupported preview pipeline for image input: {pipeline}")
        options["preview_scene_profile"] = normalize_preview_scene_profile(payload_options.get("preview_scene_profile"))
    elif project.input_type == "video":
        video_count = sum(1 for item in project.media if item.kind == "video")
        if video_count != 1 or len(project.media) != 1:
            raise HTTPException(status_code=400, detail="Video preview requires exactly one video file")
        if pipeline not in {"lingbot_video_pointcloud_fast", "lingbot_map_spz"}:
            raise HTTPException(status_code=400, detail=f"Unsupported preview pipeline for video input: {pipeline}")
        options.pop("preview_scene_profile", None)
    else:
        raise HTTPException(status_code=400, detail="Preview input type is unsupported")
    task = Task(
        project_id=project.id,
        type="preview",
        status="queued",
        priority=90,
        progress=0,
        current_stage="queued",
        eta_seconds=None,
        options=options,
    )
    project.status = "PREVIEW_RUNNING"
    project.error_message = None
    db.add(task)
    db.flush()
    emit_event(db, project.id, "task_queued", task_dict(task, project.name), task.id)
    db.commit()
    try:
        enqueue_preview_task(task.id)
    except Exception as exc:
        task.status = "failed"
        task.error_code = "QUEUE_UNAVAILABLE"
        task.error_message = f"Redis queue unavailable: {exc}"
        task.finished_at = utc_now()
        project.status = "FAILED"
        project.error_message = task.error_message
        emit_event(db, project.id, "task_failed", task_dict(task, project.name), task.id)
        db.commit()
    db.refresh(task)
    return task_dict(task, project.name)


@app.post("/api/projects/{project_id}/tasks/fine")
def create_fine_task(
    project_id: str,
    payload: TaskCreatePayload,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    project = owned_project(db, project_id, user, include_children=True)
    active_task = next((item for item in project.tasks if item.status in {"queued", "running"}), None)
    if active_task:
        raise HTTPException(status_code=409, detail="Project already has an active task")
    image_count = sum(1 for item in project.media if item.kind == "image")
    video_count = sum(1 for item in project.media if item.kind == "video")
    if project.input_type == "images" and image_count < 8:
        raise HTTPException(status_code=400, detail="FastGS-Big fine reconstruction requires at least 8 images")
    if project.input_type == "video" and (video_count != 1 or len(project.media) != 1):
        raise HTTPException(status_code=400, detail="Video fine reconstruction requires exactly one video file")
    if project.input_type == "video":
        raise HTTPException(status_code=400, detail="Video fine reconstruction is disabled; use video preview instead")
    if project.input_type not in {"images", "video"}:
        raise HTTPException(status_code=400, detail="Fine reconstruction input type is unsupported")

    fine_pipeline = "official_fastgs_big"
    eta_seconds = settings.fine_expected_seconds_images
    payload_options = payload.options or {}
    if str(payload_options.get("fine_edgs_enabled", "")).strip().lower() in {"1", "true", "yes", "on"}:
        raise HTTPException(status_code=400, detail="EDGS/RoMA dense initialization has been removed")

    options = {
        **payload_options,
        "fine_pipeline": fine_pipeline,
        "fine_scene_profile": normalize_fine_scene_profile(
            payload_options.get("fine_scene_profile") or payload_options.get("preview_scene_profile")
        ),
        "source_version": project.source_version,
        "fine_iterations": int(payload_options.get("fine_iterations") or settings.fine_iterations),
    }
    task = Task(
        project_id=project.id,
        type="fine",
        status="queued",
        priority=40,
        progress=0,
        current_stage="queued",
        eta_seconds=eta_seconds,
        options=options,
    )
    project.status = "FINE_RUNNING"
    project.error_message = None
    db.add(task)
    db.flush()
    emit_event(db, project.id, "task_queued", task_dict(task, project.name), task.id)
    db.commit()
    try:
        enqueue_fine_task(task.id)
    except Exception as exc:
        task.status = "failed"
        task.error_code = "QUEUE_UNAVAILABLE"
        task.error_message = f"Redis queue unavailable: {exc}"
        task.finished_at = utc_now()
        project.status = "FAILED"
        project.error_message = task.error_message
        emit_event(db, project.id, "task_failed", task_dict(task, project.name), task.id)
        db.commit()
    db.refresh(task)
    return task_dict(task, project.name)


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    project = owned_project(db, task.project_id, user)
    return task_dict(task, project.name)


@app.post("/api/tasks/{task_id}/cancel")
def cancel_task(task_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    task = db.get(Task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    project = owned_project(db, task.project_id, user)
    if task.status in {"queued", "running"}:
        request_worker_cancel(task)
        task.status = "canceled"
        task.finished_at = utc_now()
        task.current_stage = "canceled"
        task.error_message = "用户取消任务"
        project.status = "CANCELED"
        emit_event(db, project.id, "task_failed", task_dict(task, project.name), task.id)
        db.commit()
    return task_dict(task, project.name)


@app.get("/api/projects/{project_id}/artifacts")
def list_artifacts(project_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    project = owned_project(db, project_id, user, include_children=True)
    return {"artifacts": [artifact_dict(item) for item in project.artifacts]}


def artifact_url(artifact: Artifact, *, download: bool = False) -> str:
    token = create_artifact_token(artifact.id)
    suffix = "&download=1" if download else ""
    return f"/api/artifacts/{artifact.id}/file?token={token}{suffix}"


def artifact_original_ply_url(artifact: Artifact, *, download: bool = False) -> str:
    token = create_artifact_token(artifact.id)
    suffix = "&download=1" if download else ""
    return f"/api/artifacts/{artifact.id}/original-ply/file?token={token}{suffix}"


@app.get("/api/artifacts/{artifact_id}/download-url")
def artifact_download_url(artifact_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    artifact = db.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    owned_project(db, artifact.project_id, user)
    return {"url": artifact_url(artifact, download=True), "expires_in_seconds": settings.artifact_token_expire_seconds}


@app.get("/api/artifacts/{artifact_id}/original-ply/download-url")
def artifact_original_ply_download_url(artifact_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    artifact = db.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    owned_project(db, artifact.project_id, user)
    if is_ply_artifact(artifact):
        return {"url": artifact_url(artifact, download=True), "expires_in_seconds": settings.artifact_token_expire_seconds}
    if not intermediate_ply_path(artifact):
        raise HTTPException(status_code=404, detail="Original PLY not found")
    return {"url": artifact_original_ply_url(artifact, download=True), "expires_in_seconds": settings.artifact_token_expire_seconds}


@app.get("/api/artifacts/{artifact_id}/file")
def artifact_file(
    artifact_id: str,
    token: str = Query(...),
    download: bool = Query(default=False),
    range_header: str | None = Header(default=None, alias="Range"),
    db: Session = Depends(get_db),
):
    verify_artifact_token(token, artifact_id)
    artifact = db.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    uri = storage.resolve_existing_uri(artifact.object_uri)
    if uri != artifact.object_uri and storage.exists(uri):
        artifact.object_uri = uri
        db.commit()
    return stored_file_response(
        uri,
        filename=artifact.file_name,
        media_type=mimetypes.guess_type(artifact.file_name)[0] or "application/octet-stream",
        attachment=download,
        range_header=range_header,
    )


@app.get("/api/artifacts/{artifact_id}/original-ply/file")
def artifact_original_ply_file(
    artifact_id: str,
    token: str = Query(...),
    download: bool = Query(default=False),
    range_header: str | None = Header(default=None, alias="Range"),
    db: Session = Depends(get_db),
):
    verify_artifact_token(token, artifact_id)
    artifact = db.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if is_ply_artifact(artifact):
        uri = storage.resolve_existing_uri(artifact.object_uri)
        return stored_file_response(
            uri,
            filename=artifact.file_name,
            media_type=mimetypes.guess_type(artifact.file_name)[0] or "application/octet-stream",
            attachment=download,
            range_header=range_header,
        )
    path = intermediate_ply_path(artifact)
    if not path:
        raise HTTPException(status_code=404, detail="Original PLY not found")
    return FileResponse(
        path,
        filename="original.ply" if download else None,
        media_type=mimetypes.guess_type("original.ply")[0] or "application/octet-stream",
    )


@app.get("/api/media/{media_id}/thumbnail")
def media_thumbnail(media_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    media = db.get(MediaAsset, media_id)
    if not media or not media.thumbnail_uri:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    owned_project(db, media.project_id, user)
    if media.thumbnail_uri.startswith("/api/media/"):
        key = thumbnail_key(media.project, media.id)
        uri = storage.uri_for_key(key)
    else:
        uri = media.thumbnail_uri
    resolved_uri = storage.resolve_existing_uri(uri)
    if resolved_uri != media.thumbnail_uri and storage.exists(resolved_uri):
        media.thumbnail_uri = resolved_uri
        if media.project.preview_image_uri == uri:
            media.project.preview_image_uri = resolved_uri
        db.commit()
    uri = resolved_uri
    return stored_file_response(uri, media_type="image/jpeg")


@app.get("/api/media/{media_id}/file")
def media_file(
    media_id: str,
    range_header: str | None = Header(default=None, alias="Range"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    media = db.get(MediaAsset, media_id)
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")
    owned_project(db, media.project_id, user)
    uri = storage.resolve_existing_uri(media.object_uri)
    if uri != media.object_uri and storage.exists(uri):
        media.object_uri = uri
        db.commit()
    return stored_file_response(
        uri,
        filename=media.file_name,
        media_type=mimetypes.guess_type(media.file_name)[0] or "application/octet-stream",
        range_header=range_header,
    )


def stored_file_response(
    uri: str,
    *,
    filename: str | None = None,
    media_type: str | None = None,
    attachment: bool = False,
    range_header: str | None = None,
):
    try:
        uri = storage.resolve_existing_uri(uri)
        size = storage.size(uri)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Stored file not found") from exc
    headers = {"Accept-Ranges": "bytes", "Content-Length": str(size)}
    if attachment and filename:
        headers["Content-Disposition"] = f'attachment; filename="{safe_filename(filename)}"'
    if range_header:
        start, end = parse_range_header(range_header, size)
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(end - start + 1)
        return StreamingResponse(
            storage.iter_range(uri, start, end),
            media_type=media_type or "application/octet-stream",
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            headers=headers,
        )
    if uri.startswith(("db://", "s3://")):
        return StreamingResponse(storage.iter_bytes(uri), media_type=media_type or "application/octet-stream", headers=headers)
    return FileResponse(storage.local_path(uri), filename=filename if attachment else None, media_type=media_type, headers=headers)


def parse_range_header(range_header: str, size: int) -> tuple[int, int]:
    if not range_header.startswith("bytes=") or "," in range_header:
        raise HTTPException(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            detail="Unsupported range",
            headers={"Content-Range": f"bytes */{size}"},
        )
    raw_start, _, raw_end = range_header.removeprefix("bytes=").partition("-")
    try:
        if raw_start == "":
            suffix_size = int(raw_end)
            if suffix_size <= 0:
                raise ValueError
            return max(size - suffix_size, 0), max(size - 1, 0)
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            detail="Invalid range",
            headers={"Content-Range": f"bytes */{size}"},
        ) from exc
    if start < 0 or start >= size or end < start:
        raise HTTPException(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            detail="Range not satisfiable",
            headers={"Content-Range": f"bytes */{size}"},
        )
    return start, min(end, size - 1)


def fresh_final_viewer_artifacts(project: Project) -> dict[str, Artifact | None]:
    final_artifacts = [
        item
        for item in project.artifacts
        if item.kind in {"final_spz", "lod_rad"} and item.source_version == project.source_version
    ]
    model = sorted(final_artifacts, key=lambda item: (item.kind == "final_spz", item.created_at), reverse=True)[0] if final_artifacts else None
    final_plys = [item for item in project.artifacts if item.kind == "final_ply" and item.source_version == project.source_version]
    metas = [item for item in project.artifacts if item.kind == "viewer_meta_json" and item.source_version == project.source_version]
    return {
        "model": model,
        "ply": sorted(final_plys, key=lambda item: item.created_at, reverse=True)[0] if final_plys else None,
        "meta": sorted(metas, key=lambda item: item.created_at, reverse=True)[0] if metas else None,
    }


def fresh_preview_viewer_artifacts(project: Project) -> dict[str, Artifact | None]:
    preview_artifacts = [
        item
        for item in project.artifacts
        if item.kind in {"preview_spz", "preview_pointcloud_ply"} and item.source_version == project.source_version
    ]
    model = sorted(preview_artifacts, key=lambda item: item.created_at, reverse=True)[0] if preview_artifacts else None
    if not model:
        return {"model": None, "ply": None, "full_ply": None, "debug_splats": None, "meta": None}
    task_artifacts = [item for item in project.artifacts if item.task_id == model.task_id]
    full_ply = next((item for item in task_artifacts if item.kind == "preview_full_ply"), None)
    return {
        "model": model,
        "ply": next((item for item in task_artifacts if item.kind == "original_ply"), None),
        "full_ply": full_ply,
        "debug_splats": next((item for item in task_artifacts if item.kind == "debug_splats_ply"), None),
        "meta": next((item for item in task_artifacts if item.kind == "preview_meta_json"), None),
    }


def project_viewer_payload(project: Project) -> dict[str, Any]:
    final = fresh_final_viewer_artifacts(project)
    final_model = final["model"]
    if final_model:
        final_ply = final["ply"]
        final_meta = final["meta"]
        return {
            "status": "ready",
            "mode": "single",
            "source": "final",
            "artifact_id": final_model.id,
            "model_url": artifact_url(final_model),
            "download_spz_url": artifact_url(final_model, download=True),
            "file_size": final_model.file_size,
            "gaussian_ply_url": artifact_url(final_ply) if final_ply else None,
            "download_ply_url": artifact_url(final_ply, download=True) if final_ply else None,
            "viewer_meta_url": artifact_url(final_meta) if final_meta else None,
            "format": "rad" if final_model.kind == "lod_rad" or final_model.file_name.lower().endswith(".rad") else "spz",
        }

    preview = fresh_preview_viewer_artifacts(project)
    preview_model = preview["model"]
    if preview_model:
        preview_ply = preview["ply"]
        preview_full_ply = preview["full_ply"]
        preview_meta = preview["meta"]
        if preview_model.kind == "preview_pointcloud_ply":
            point_ply = preview_ply or preview_model
            download_ply = preview_full_ply or point_ply
            return {
                "status": "ready",
                "mode": "single",
                "source": "preview",
                "artifact_id": preview_model.id,
                "model_url": None,
                "download_spz_url": None,
                "file_size": preview_model.file_size,
                "format": "ply",
                "debug_points_ply_url": artifact_url(point_ply) if point_ply else None,
                "debug_splats_ply_url": None,
                "download_ply_url": artifact_url(download_ply, download=True) if download_ply else None,
                "preview_meta_url": artifact_url(preview_meta) if preview_meta else None,
                "quality_warning": (preview_model.metadata_json or {}).get("quality_warning"),
                "point_source": (preview_model.metadata_json or {}).get("point_source")
                or (preview_model.metadata_json or {}).get("lingbot_point_source"),
            }
        return {
            "status": "ready",
            "mode": "single",
            "source": "preview",
            "artifact_id": preview_model.id,
            "model_url": artifact_url(preview_model),
            "download_spz_url": artifact_url(preview_model, download=True),
            "file_size": preview_model.file_size,
            "format": "spz",
            "debug_points_ply_url": artifact_url(preview_ply) if preview_ply else None,
            "debug_splats_ply_url": artifact_url(preview["debug_splats"]) if preview["debug_splats"] else None,
            "download_ply_url": artifact_url(preview_ply, download=True) if preview_ply else None,
            "preview_meta_url": artifact_url(preview_meta) if preview_meta else None,
            "quality_warning": (preview_model.metadata_json or {}).get("quality_warning"),
            "point_source": (preview_model.metadata_json or {}).get("point_source")
            or (preview_model.metadata_json or {}).get("lingbot_point_source"),
        }

    final_artifacts = [item for item in project.artifacts if item.kind in {"final_spz", "lod_rad"}]
    preview_artifacts = [item for item in project.artifacts if item.kind in {"preview_spz", "preview_pointcloud_ply"}]
    if final_artifacts:
        return {
            "status": "unavailable",
            "message": "Final artifact is stale because source media changed; start fine reconstruction again.",
            "stale": True,
        }
    if preview_artifacts:
        return {
            "status": "unavailable",
            "message": "Preview artifact is stale because source media changed; start preview again.",
            "stale": True,
        }
    return {"status": "unavailable", "message": "No preview or final 3D artifact is available."}


@app.get("/api/projects/{project_id}/viewer-config")
def viewer_config(project_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    project = owned_project(db, project_id, user, include_children=True)
    return project_viewer_payload(project)
    final_artifacts = [item for item in project.artifacts if item.kind in {"final_spz", "lod_rad"}]
    fresh_final = [item for item in final_artifacts if item.source_version == project.source_version]
    if fresh_final:
        artifact = sorted(fresh_final, key=lambda item: (item.kind == "final_spz", item.created_at), reverse=True)[0]
        fresh_final_plys = [
            item
            for item in project.artifacts
            if item.kind == "final_ply" and item.source_version == project.source_version
        ]
        gaussian_ply = sorted(fresh_final_plys, key=lambda item: item.created_at, reverse=True)[0] if fresh_final_plys else None
        return {
            "status": "ready",
            "mode": "single",
            "source": "final",
            "artifact_id": artifact.id,
            "model_url": artifact_url(artifact),
            "gaussian_ply_url": artifact_url(gaussian_ply) if gaussian_ply else None,
            "format": "rad" if artifact.kind == "lod_rad" or artifact.file_name.lower().endswith(".rad") else "spz",
        }
    preview_artifacts = [item for item in project.artifacts if item.kind == "preview_spz"]
    fresh = [item for item in preview_artifacts if item.source_version == project.source_version]
    if fresh:
        artifact = sorted(fresh, key=lambda item: item.created_at, reverse=True)[0]
        task_artifacts = [item for item in project.artifacts if item.task_id == artifact.task_id]
        debug_points = next((item for item in task_artifacts if item.kind == "original_ply"), None)
        debug_splats = next((item for item in task_artifacts if item.kind == "debug_splats_ply"), None)
        preview_meta = next((item for item in task_artifacts if item.kind == "preview_meta_json"), None)
        return {
            "status": "ready",
            "mode": "single",
            "source": "preview",
            "artifact_id": artifact.id,
            "model_url": artifact_url(artifact),
            "format": "spz",
            "debug_points_ply_url": artifact_url(debug_points) if debug_points else None,
            "debug_splats_ply_url": artifact_url(debug_splats) if debug_splats else None,
            "preview_meta_url": artifact_url(preview_meta) if preview_meta else None,
            "quality_warning": (artifact.metadata_json or {}).get("quality_warning"),
            "point_source": (artifact.metadata_json or {}).get("point_source")
            or (artifact.metadata_json or {}).get("lingbot_point_source"),
        }
    if final_artifacts:
        return {
            "status": "unavailable",
            "message": "Final artifact is stale because source media changed; start fine reconstruction again.",
            "stale": True,
        }
    if preview_artifacts:
        return {
            "status": "unavailable",
            "message": "素材已删除或补传，当前预览产物已过期，请重新启动预览。",
            "stale": True,
        }
    return {"status": "unavailable", "message": "暂无真实 preview.spz 产物。"}


@app.post("/api/projects/{project_id}/share")
def create_project_share(project_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    project = owned_project(db, project_id, user, include_children=True)
    if not project.share_token:
        project.share_token = unique_share_token(db)
        db.commit()
        db.refresh(project)
    return {"share_token": project.share_token, "share_url": f"/share/{project.share_token}", "project": shared_project_payload(project)}


@app.delete("/api/projects/{project_id}/share")
def delete_project_share(project_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    project = owned_project(db, project_id, user)
    project.share_token = None
    db.commit()
    return {"deleted": True}


@app.get("/api/shared-projects/{share_token}")
def shared_project(share_token: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    project = db.scalar(
        select(Project)
        .where(Project.share_token == share_token)
        .options(selectinload(Project.artifacts))
    )
    if not project:
        raise HTTPException(status_code=404, detail="Shared project not found")
    return shared_project_payload(project)


def unique_share_token(db: Session) -> str:
    for _ in range(8):
        token = secrets.token_urlsafe(24)
        if not db.scalar(select(Project.id).where(Project.share_token == token)):
            return token
    raise HTTPException(status_code=500, detail="Unable to allocate share token")


def shared_project_payload(project: Project) -> dict[str, Any]:
    return {
        "id": project.id,
        "name": project.name,
        "tags": project.tags or [],
        "total_size_bytes": project.total_size_bytes,
        "created_at": iso(project.created_at),
        "updated_at": iso(project.updated_at),
        "viewer": project_viewer_payload(project),
    }


@app.post("/api/feedback")
def create_feedback(payload: FeedbackPayload, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    feedback = Feedback(user_id=user.id, project_id=payload.project_id, title=payload.title, content=payload.content)
    db.add(feedback)
    db.commit()
    return {"ok": True, "id": feedback.id}


@app.get("/api/algorithms")
def public_algorithms(db: Session = Depends(get_db)) -> dict[str, Any]:
    items = db.scalars(select(AlgorithmRegistry).order_by(AlgorithmRegistry.name)).all()
    return {"algorithms": [algorithm_dict(item) for item in items]}


@app.get("/api/admin/algorithms")
def admin_algorithms(_: User = Depends(require_admin), db: Session = Depends(get_db)) -> dict[str, Any]:
    items = db.scalars(select(AlgorithmRegistry).order_by(AlgorithmRegistry.name)).all()
    return {"algorithms": [algorithm_dict(item) for item in items]}


def algorithm_dict(item: AlgorithmRegistry) -> dict[str, Any]:
    return {
        "name": item.name,
        "repo_url": item.repo_url,
        "license": item.license,
        "commit_hash": item.commit_hash,
        "weight_source": item.weight_source,
        "local_path": item.local_path,
        "enabled": item.enabled,
        "notes": item.notes,
        "commands": item.commands or {},
        "weight_paths": item.weight_paths or [],
        "source_type": item.source_type,
        "bundled": item.source_type == "bundled",
        "license_notice": license_notice_for(item.name),
    }


@app.get("/api/admin/runtime/preflight")
def preflight(_: User = Depends(require_admin), db: Session = Depends(get_db)) -> dict[str, Any]:
    return runtime_preflight(db, settings)


@app.get("/api/admin/tasks")
def admin_tasks(_: User = Depends(require_admin), db: Session = Depends(get_db)) -> dict[str, Any]:
    tasks = db.scalars(select(Task).options(selectinload(Task.project)).order_by(Task.created_at.desc()).limit(100)).all()
    return {"tasks": [task_dict(item) for item in tasks]}


@app.get("/api/admin/workers")
def admin_workers(_: User = Depends(require_admin), db: Session = Depends(get_db)) -> dict[str, Any]:
    from app.models import WorkerHeartbeat

    workers = db.scalars(select(WorkerHeartbeat).order_by(WorkerHeartbeat.last_seen_at.desc())).all()
    return {
        "workers": [
            {
                "worker_id": item.worker_id,
                "hostname": item.hostname,
                "gpu_index": item.gpu_index,
                "gpu_name": item.gpu_name,
                "gpu_memory_total": item.gpu_memory_total,
                "gpu_memory_used": item.gpu_memory_used,
                "gpu_utilization": item.gpu_utilization,
                "current_task_id": item.current_task_id,
                "last_seen_at": iso(item.last_seen_at),
            }
            for item in workers
        ]
    }


@app.get("/api/projects/{project_id}/events")
async def project_events(project_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    owned_project(db, project_id, user)

    async def stream():
        with SessionLocal() as snapshot_db:
            project = owned_project(snapshot_db, project_id, user, include_children=True)
            yield sse("project_snapshot", project_dict(project, include_children=True))
            last_id = snapshot_db.scalar(select(func.max(TaskEvent.id)).where(TaskEvent.project_id == project_id)) or 0
        while True:
            with SessionLocal() as event_db:
                events = event_db.scalars(
                    select(TaskEvent).where(TaskEvent.project_id == project_id, TaskEvent.id > last_id).order_by(TaskEvent.id)
                ).all()
                for event in events:
                    last_id = event.id
                    yield sse(event.event, event.payload)
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream")


def sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
