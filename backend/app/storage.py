from __future__ import annotations

import hashlib
import io
import shutil
from pathlib import Path
from typing import BinaryIO, Iterator

import boto3
from fastapi import UploadFile
from sqlalchemy import delete, select

from app.config import Settings, get_settings

DB_CHUNK_SIZE = 4 * 1024 * 1024
READ_CHUNK_SIZE = 1024 * 1024


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
        if self.is_database_backend:
            return f"db://{key}"
        return f"local://{key}"

    def key_from_uri(self, uri: str) -> str:
        if uri.startswith("local://"):
            return uri[len("local://") :]
        if uri.startswith("db://"):
            return uri[len("db://") :]
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
            file.file.seek(0, 2)
            size = file.file.tell()
            file.file.seek(0)
            self.s3().upload_fileobj(file.file, self.settings.s3_bucket, key)
            return self.uri_for_key(key), size
        if self.is_database_backend:
            size = await self._put_blob_from_upload(key, file)
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
        if self.is_database_backend:
            self._put_blob(key, data)
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
        if self.is_database_backend:
            self._put_blob_from_path(key, source)
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
        if uri.startswith("db://"):
            with target.open("wb") as handle:
                for chunk in self.iter_bytes(uri):
                    handle.write(chunk)
            return target
        shutil.copy2(self.local_path(uri), target)
        return target

    def open_file(self, uri: str) -> BinaryIO:
        if uri.startswith("db://"):
            return io.BytesIO(self._get_blob_data(self.key_from_uri(uri)))
        if uri.startswith("s3://"):
            import tempfile

            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.close()
            self.download_to_path(uri, Path(tmp.name))
            return open(tmp.name, "rb")
        return self.local_path(uri).open("rb")

    def exists(self, uri: str) -> bool:
        if uri.startswith("db://"):
            return self._blob_exists(self.key_from_uri(uri))
        if uri.startswith("s3://"):
            try:
                self.s3().head_object(Bucket=self.settings.s3_bucket, Key=self.key_from_uri(uri))
                return True
            except Exception:
                return False
        return self.local_path(uri).exists()

    def resolve_existing_uri(self, uri: str) -> str:
        if self.exists(uri):
            return uri
        key = self.key_from_uri(uri)
        if uri.startswith("local://"):
            db_uri = f"db://{key}"
            if self._blob_exists(key):
                return db_uri
        if uri.startswith("db://"):
            local_uri = f"local://{key}"
            if self.local_path(local_uri).exists():
                return local_uri
        return uri

    def size(self, uri: str) -> int:
        if uri.startswith("db://"):
            return self._blob_size(self.key_from_uri(uri))
        if uri.startswith("s3://"):
            return int(self.s3().head_object(Bucket=self.settings.s3_bucket, Key=self.key_from_uri(uri))["ContentLength"])
        return self.local_path(uri).stat().st_size

    def iter_bytes(self, uri: str) -> Iterator[bytes]:
        if uri.startswith("db://"):
            yield from self._iter_blob_chunks(self.key_from_uri(uri))
            return
        if uri.startswith("s3://"):
            body = self.s3().get_object(Bucket=self.settings.s3_bucket, Key=self.key_from_uri(uri))["Body"]
            try:
                for chunk in body.iter_chunks(chunk_size=READ_CHUNK_SIZE):
                    if chunk:
                        yield chunk
            finally:
                body.close()
            return
        with self.local_path(uri).open("rb") as handle:
            while chunk := handle.read(READ_CHUNK_SIZE):
                yield chunk

    def iter_range(self, uri: str, start: int, end: int) -> Iterator[bytes]:
        if end < start:
            return
        remaining = end - start + 1
        if uri.startswith("db://"):
            offset = start
            for chunk in self.iter_bytes(uri):
                if offset >= len(chunk):
                    offset -= len(chunk)
                    continue
                data = chunk[offset:]
                offset = 0
                if len(data) > remaining:
                    data = data[:remaining]
                remaining -= len(data)
                yield data
                if remaining <= 0:
                    return
            return
        if uri.startswith("s3://"):
            body = self.s3().get_object(
                Bucket=self.settings.s3_bucket,
                Key=self.key_from_uri(uri),
                Range=f"bytes={start}-{end}",
            )["Body"]
            try:
                for chunk in body.iter_chunks(chunk_size=READ_CHUNK_SIZE):
                    if not chunk:
                        continue
                    if len(chunk) > remaining:
                        chunk = chunk[:remaining]
                    remaining -= len(chunk)
                    yield chunk
                    if remaining <= 0:
                        return
            finally:
                body.close()
            return
        with self.local_path(uri).open("rb") as handle:
            handle.seek(start)
            while remaining > 0:
                chunk = handle.read(min(READ_CHUNK_SIZE, remaining))
                if not chunk:
                    return
                remaining -= len(chunk)
                yield chunk

    def delete(self, uri: str | None) -> None:
        if not uri:
            return
        if uri.startswith("db://"):
            self._delete_blob(self.key_from_uri(uri))
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

    @property
    def is_database_backend(self) -> bool:
        return self.backend in {"db", "database", "postgres", "postgresql"}

    def _put_blob(self, key: str, data: bytes) -> None:
        view = memoryview(data)
        self._replace_blob(key, (bytes(view[index : index + DB_CHUNK_SIZE]) for index in range(0, len(view), DB_CHUNK_SIZE)), len(data))

    async def _put_blob_from_upload(self, key: str, file: UploadFile) -> int:
        from app.database import SessionLocal
        from app.models import StoredObject, StoredObjectChunk

        with SessionLocal() as db:
            db.execute(delete(StoredObjectChunk).where(StoredObjectChunk.key == key))
            db.execute(delete(StoredObject).where(StoredObject.key == key))
            item = StoredObject(key=key, size=0)
            db.add(item)
            total = 0
            index = 0
            while chunk := await file.read(DB_CHUNK_SIZE):
                db.add(StoredObjectChunk(key=key, chunk_index=index, data=chunk, size=len(chunk)))
                total += len(chunk)
                index += 1
                if index % 16 == 0:
                    db.flush()
            item.size = total
            db.commit()
            return total

    def _put_blob_from_path(self, key: str, source: Path) -> None:
        total = source.stat().st_size

        def chunks() -> Iterator[bytes]:
            with source.open("rb") as handle:
                while chunk := handle.read(DB_CHUNK_SIZE):
                    yield chunk

        self._replace_blob(key, chunks(), total)

    def _replace_blob(self, key: str, chunks: Iterator[bytes], total: int) -> None:
        from app.database import SessionLocal
        from app.models import StoredObject, StoredObjectChunk

        with SessionLocal() as db:
            db.execute(delete(StoredObjectChunk).where(StoredObjectChunk.key == key))
            db.execute(delete(StoredObject).where(StoredObject.key == key))
            db.add(StoredObject(key=key, size=total))
            for index, chunk in enumerate(chunks):
                db.add(StoredObjectChunk(key=key, chunk_index=index, data=chunk, size=len(chunk)))
                if index % 16 == 15:
                    db.flush()
            db.commit()

    def _get_blob_data(self, key: str) -> bytes:
        from app.database import SessionLocal
        from app.models import StoredObject, StoredObjectChunk

        with SessionLocal() as db:
            item = db.get(StoredObject, key)
            if not item:
                raise FileNotFoundError(key)
            chunks = db.scalars(
                select(StoredObjectChunk.data)
                .where(StoredObjectChunk.key == key)
                .order_by(StoredObjectChunk.chunk_index)
            ).all()
            return b"".join(bytes(chunk) for chunk in chunks)

    def _iter_blob_chunks(self, key: str) -> Iterator[bytes]:
        from app.database import SessionLocal
        from app.models import StoredObject, StoredObjectChunk

        with SessionLocal() as db:
            if not db.get(StoredObject, key):
                raise FileNotFoundError(key)
            for chunk in db.scalars(
                select(StoredObjectChunk.data)
                .where(StoredObjectChunk.key == key)
                .order_by(StoredObjectChunk.chunk_index)
            ):
                yield bytes(chunk)

    def _blob_exists(self, key: str) -> bool:
        from app.database import SessionLocal
        from app.models import StoredObject

        with SessionLocal() as db:
            return db.get(StoredObject, key) is not None

    def _blob_size(self, key: str) -> int:
        from app.database import SessionLocal
        from app.models import StoredObject

        with SessionLocal() as db:
            item = db.get(StoredObject, key)
            if not item:
                raise FileNotFoundError(key)
            return item.size

    def _delete_blob(self, key: str) -> None:
        from app.database import SessionLocal
        from app.models import StoredObject, StoredObjectChunk

        with SessionLocal() as db:
            db.execute(delete(StoredObjectChunk).where(StoredObjectChunk.key == key))
            item = db.get(StoredObject, key)
            if item:
                db.delete(item)
            db.commit()


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
