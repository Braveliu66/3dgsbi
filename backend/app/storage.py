from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import BinaryIO

import boto3
from fastapi import UploadFile

from app.config import Settings, get_settings


class Storage:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.backend = self.settings.storage_backend.lower()
        self.root = self.settings.local_storage_root
        self._s3 = None

    def s3(self):
        if self._s3 is None:
            self._s3 = boto3.client(
                "s3",
                endpoint_url=self.settings.s3_endpoint_url,
                aws_access_key_id=self.settings.s3_access_key,
                aws_secret_access_key=self.settings.s3_secret_key,
                region_name=self.settings.s3_region,
            )
        return self._s3

    def ensure_bucket(self) -> None:
        if self.backend != "s3":
            return
        client = self.s3()
        buckets = [item["Name"] for item in client.list_buckets().get("Buckets", [])]
        if self.settings.s3_bucket not in buckets:
            client.create_bucket(Bucket=self.settings.s3_bucket)

    def uri_for_key(self, key: str) -> str:
        key = key.replace("\\", "/").lstrip("/")
        if self.backend == "s3":
            return f"s3://{self.settings.s3_bucket}/{key}"
        return f"local://{key}"

    def key_from_uri(self, uri: str) -> str:
        if uri.startswith("local://"):
            return uri[len("local://") :]
        if uri.startswith("s3://"):
            _, rest = uri.split("s3://", 1)
            bucket, key = rest.split("/", 1)
            if bucket != self.settings.s3_bucket:
                raise ValueError("Unexpected bucket")
            return key
        raise ValueError(f"Unsupported storage uri: {uri}")

    def local_path(self, uri: str) -> Path:
        key = self.key_from_uri(uri)
        return (self.root / key).resolve()

    async def save_upload(self, file: UploadFile, key: str) -> tuple[str, int]:
        key = key.replace("\\", "/").lstrip("/")
        if self.backend == "s3":
            self.ensure_bucket()
            file.file.seek(0)
            self.s3().upload_fileobj(file.file, self.settings.s3_bucket, key)
            size = file.file.tell()
            return self.uri_for_key(key), size
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                out.write(chunk)
        return self.uri_for_key(key), size

    def write_bytes(self, key: str, data: bytes) -> str:
        key = key.replace("\\", "/").lstrip("/")
        if self.backend == "s3":
            self.ensure_bucket()
            self.s3().put_object(Bucket=self.settings.s3_bucket, Key=key, Body=data)
            return self.uri_for_key(key)
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return self.uri_for_key(key)

    def upload_path(self, source: Path, key: str) -> str:
        key = key.replace("\\", "/").lstrip("/")
        if self.backend == "s3":
            self.ensure_bucket()
            self.s3().upload_file(str(source), self.settings.s3_bucket, key)
            return self.uri_for_key(key)
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return self.uri_for_key(key)

    def download_to_path(self, uri: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        if uri.startswith("s3://"):
            self.ensure_bucket()
            self.s3().download_file(self.settings.s3_bucket, self.key_from_uri(uri), str(target))
            return target
        shutil.copy2(self.local_path(uri), target)
        return target

    def open_file(self, uri: str) -> BinaryIO:
        if uri.startswith("s3://"):
            import tempfile

            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.close()
            self.download_to_path(uri, Path(tmp.name))
            return open(tmp.name, "rb")
        return self.local_path(uri).open("rb")

    def exists(self, uri: str) -> bool:
        if uri.startswith("s3://"):
            try:
                self.s3().head_object(Bucket=self.settings.s3_bucket, Key=self.key_from_uri(uri))
                return True
            except Exception:
                return False
        return self.local_path(uri).exists()

    def size(self, uri: str) -> int:
        if uri.startswith("s3://"):
            return int(self.s3().head_object(Bucket=self.settings.s3_bucket, Key=self.key_from_uri(uri))["ContentLength"])
        return self.local_path(uri).stat().st_size

    def delete(self, uri: str | None) -> None:
        if not uri:
            return
        if uri.startswith("s3://"):
            self.s3().delete_object(Bucket=self.settings.s3_bucket, Key=self.key_from_uri(uri))
            return
        path = self.local_path(uri)
        if path.exists():
            path.unlink()

    def presigned_get_url(self, uri: str, expires: int) -> str | None:
        if not uri.startswith("s3://"):
            return None
        return self.s3().generate_presigned_url(
            "get_object",
            Params={"Bucket": self.settings.s3_bucket, "Key": self.key_from_uri(uri)},
            ExpiresIn=expires,
        )


def storage_key(*parts: str) -> str:
    return "/".join(str(part).strip("/").replace("\\", "/") for part in parts if str(part).strip("/"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in name).strip()
    return cleaned or "upload.bin"

