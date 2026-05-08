from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx


HF_BASE_URL = "https://huggingface.co"
HF_MIRROR_BASE_URL = "https://hf-mirror.com"


@dataclass(frozen=True, slots=True)
class ModelWeight:
    relative_path: str
    url: str
    alternate_urls: tuple[str, ...] = ()


LITEVGGT_WEIGHT = ModelWeight(
    "litevggt/te_dict.pt",
    "https://huggingface.co/ZhijianShu/LiteVGGT/resolve/main/te_dict.pt",
)
LINGBOT_WEIGHT = ModelWeight(
    "lingbot-map/lingbot-map-long.pt",
    "https://huggingface.co/robbyant/lingbot-map/resolve/main/lingbot-map-long.pt",
)
ROMA_WEIGHT = ModelWeight(
    "roma/roma_indoor.pth",
    "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
)
DINOV2_WEIGHT = ModelWeight(
    "roma/dinov2_vitl14_pretrain.pth",
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
)

MODEL_WEIGHTS: tuple[ModelWeight, ...] = (
    LITEVGGT_WEIGHT,
    LINGBOT_WEIGHT,
    ROMA_WEIGHT,
    DINOV2_WEIGHT,
)
WEIGHT_BY_RELATIVE_PATH = {item.relative_path: item for item in MODEL_WEIGHTS}

PIPELINE_WEIGHT_PATHS: dict[str, tuple[str, ...]] = {
    "litevggt_spz": ("litevggt/te_dict.pt",),
    "litevggt_edgs": ("roma/roma_indoor.pth", "roma/dinov2_vitl14_pretrain.pth"),
    "mobilegs_lmrs": ("litevggt/te_dict.pt",),
    "lingbot_spz": ("lingbot-map/lingbot-map-long.pt",),
}


class ModelDownloadError(RuntimeError):
    pass


class _RestartWithoutRange(Exception):
    pass


def download_model_weights(
    model_root: Path,
    specs: Iterable[ModelWeight] = MODEL_WEIGHTS,
    *,
    prefer_hf_mirror: bool = True,
    lock_timeout_seconds: int = 60 * 60,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    results = []
    for spec in specs:
        if log:
            log(f"[weights] {spec.relative_path}: checking")
        result = download_model_weight(
            model_root,
            spec,
            prefer_hf_mirror=prefer_hf_mirror,
            lock_timeout_seconds=lock_timeout_seconds,
        )
        results.append(result)
        if log:
            log(_format_result(result))
    return results


def weights_for_pipeline(pipeline: str) -> tuple[ModelWeight, ...]:
    return tuple(WEIGHT_BY_RELATIVE_PATH[path] for path in PIPELINE_WEIGHT_PATHS.get(pipeline, ()))


def download_model_weight(
    model_root: Path,
    spec: ModelWeight,
    *,
    prefer_hf_mirror: bool = True,
    lock_timeout_seconds: int = 60 * 60,
) -> dict[str, Any]:
    target = safe_weight_path(model_root, spec.relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = part_path(target)
    meta = meta_path(target)
    lock = lock_path(target)

    with file_lock(lock, timeout_seconds=lock_timeout_seconds):
        if target.exists() and target.is_file() and target.stat().st_size > 0:
            return {
                "relative_path": spec.relative_path,
                "path": str(target),
                "status": "exists",
                "size_bytes": target.stat().st_size,
                "partial_exists": part.exists(),
            }

        errors: list[str] = []
        for url in candidate_urls(spec, prefer_hf_mirror=prefer_hf_mirror):
            try:
                result = _download_url(url, target, part, meta, spec.relative_path)
                result["relative_path"] = spec.relative_path
                return result
            except Exception as exc:
                errors.append(f"{url}: {exc}")
        raise ModelDownloadError(f"failed to download {spec.relative_path}: {'; '.join(errors)}")


def safe_weight_path(model_root: Path, relative_path: str) -> Path:
    root = model_root.resolve()
    target = (root / relative_path).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"weight path escapes model cache: {relative_path}")
    return target


def candidate_urls(spec: ModelWeight, *, prefer_hf_mirror: bool) -> list[str]:
    urls: list[str] = []
    if prefer_hf_mirror and spec.url.startswith(HF_BASE_URL):
        urls.append(HF_MIRROR_BASE_URL + spec.url[len(HF_BASE_URL) :])
    urls.append(spec.url)
    urls.extend(spec.alternate_urls)

    seen: set[str] = set()
    deduped = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def _download_url(url: str, target: Path, part: Path, meta: Path, relative_path: str) -> dict[str, Any]:
    timeout = httpx.Timeout(connect=20.0, read=60.0, write=60.0, pool=20.0)
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        try:
            return _stream_url(client, url, target, part, meta, relative_path)
        except _RestartWithoutRange:
            part.unlink(missing_ok=True)
            return _stream_url(client, url, target, part, meta, relative_path)


def _stream_url(
    client: httpx.Client,
    url: str,
    target: Path,
    part: Path,
    meta: Path,
    relative_path: str,
) -> dict[str, Any]:
    existing_size = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}
    with client.stream("GET", url, headers=headers) as response:
        if response.status_code == 416 and existing_size > 0:
            raise _RestartWithoutRange()
        if response.status_code >= 400:
            raise ModelDownloadError(f"HTTP {response.status_code}")

        resumed = existing_size > 0 and response.status_code == 206
        mode = "ab" if resumed else "wb"
        if existing_size > 0 and not resumed:
            part.unlink(missing_ok=True)
            existing_size = 0

        with part.open(mode) as out:
            for chunk in response.iter_bytes():
                if chunk:
                    out.write(chunk)

        size_bytes = part.stat().st_size
        if size_bytes <= 0:
            raise ModelDownloadError("downloaded file is empty")

        os.replace(part, target)
        download_meta = {
            "relative_path": relative_path,
            "url": str(response.url),
            "requested_url": url,
            "etag": response.headers.get("etag"),
            "content_length": _content_length(response, existing_size),
            "size_bytes": target.stat().st_size,
            "resumed": resumed,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        meta.write_text(json.dumps(download_meta, ensure_ascii=True, indent=2), encoding="utf-8")
        return {
            "path": str(target),
            "status": "downloaded",
            "url": str(response.url),
            "requested_url": url,
            "size_bytes": target.stat().st_size,
            "resumed": resumed,
            "partial_exists": False,
        }


def _content_length(response: httpx.Response, existing_size: int) -> int | None:
    content_range = response.headers.get("content-range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[-1]
        if total.isdigit():
            return int(total)
    value = response.headers.get("content-length")
    if value and value.isdigit():
        return int(value) + existing_size
    return None


def weight_file_status(path: Path) -> dict[str, Any]:
    path = Path(path)
    part = part_path(path)
    meta = meta_path(path)
    return {
        "path": str(path),
        "exists": path.exists() and path.is_file(),
        "partial_exists": part.exists() and part.is_file(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "partial_size_bytes": part.stat().st_size if part.exists() and part.is_file() else None,
        "download_meta": read_download_meta(meta),
    }


def read_download_meta(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def part_path(target: Path) -> Path:
    return target.with_name(target.name + ".part")


def meta_path(target: Path) -> Path:
    return target.with_name(target.name + ".download.json")


def lock_path(target: Path) -> Path:
    return target.with_name(target.name + ".lock")


@contextmanager
def file_lock(path: Path, *, timeout_seconds: int):
    deadline = time.monotonic() + timeout_seconds
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
                lock_file.write(f"pid={os.getpid()}\n")
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise ModelDownloadError(f"timed out waiting for weight lock: {path}")
            time.sleep(1)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _format_result(result: dict[str, Any]) -> str:
    status = result.get("status")
    rel = result.get("relative_path")
    size = result.get("size_bytes")
    resumed = " resumed" if result.get("resumed") else ""
    return f"[weights] {rel}: {status}{resumed} ({size} bytes)"
