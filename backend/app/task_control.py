from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings


TASK_CANCEL_TTL_SECONDS = 24 * 60 * 60


def task_cancel_key(task_id: str) -> str:
    return f"task_cancel:{task_id}"


def queue_name_for_task_type(task_type: str | None, settings: Settings | None = None) -> str | None:
    settings = settings or get_settings()
    if task_type == "preview":
        return settings.preview_queue_name
    if task_type == "fine":
        return settings.fine_queue_name
    return None


def request_task_cancel(redis_client: Any, task_id: str, task_type: str | None = None) -> None:
    redis_client.set(task_cancel_key(task_id), "1", ex=TASK_CANCEL_TTL_SECONDS)
    queue_name = queue_name_for_task_type(task_type)
    if queue_name:
        redis_client.lrem(queue_name, 0, task_id)


def task_cancel_requested(redis_client: Any, task_id: str) -> bool:
    return bool(redis_client.exists(task_cancel_key(task_id)))
