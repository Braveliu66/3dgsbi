from __future__ import annotations

from datetime import datetime, timezone


def work_save_stem(username: str | None, mode: str | None, type_: str | None, value: datetime | None = None) -> str:
    timestamp = work_timestamp(value)
    parts = [
        clean_work_name_part(username or "user"),
        clean_work_name_part(mode or "work"),
        clean_work_name_part(type_ or "output"),
        timestamp,
    ]
    return "-".join(parts)


def work_timestamp(value: datetime | None = None) -> str:
    moment = value or datetime.now(timezone.utc)
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y%m%d%H%M%S%f")


def clean_work_name_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in str(value)).strip()
    return (cleaned or "work").replace(" ", "_")
