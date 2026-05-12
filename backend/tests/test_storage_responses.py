from __future__ import annotations

import base64
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TMP = tempfile.TemporaryDirectory()
TMP_PATH = Path(TMP.name)
os.environ["DATABASE_URL"] = f"sqlite:///{(TMP_PATH / 'app.db').as_posix()}"
os.environ["STORAGE_BACKEND"] = "db"
os.environ["STORAGE_ROOT"] = str(TMP_PATH / "storage")
os.environ["WORK_ROOT"] = str(TMP_PATH / "work")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.main import app, storage  # noqa: E402
from app.models import Artifact, MediaAsset, Task  # noqa: E402
from app.task_control import task_cancel_key  # noqa: E402


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class FakeRedis:
    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.values: dict[str, tuple[str, int | None]] = {}

    def rpush(self, name: str, value: str) -> int:
        self.lists.setdefault(name, []).append(value)
        return len(self.lists[name])

    def lrem(self, name: str, count: int, value: str) -> int:
        items = self.lists.get(name, [])
        kept = [item for item in items if item != value]
        removed = len(items) - len(kept)
        self.lists[name] = kept
        return removed

    def set(self, name: str, value: str, ex: int | None = None) -> bool:
        self.values[name] = (value, ex)
        return True

    def exists(self, name: str) -> int:
        return int(name in self.values)


class StorageResponseTests(unittest.TestCase):
    def test_db_upload_can_be_previewed(self) -> None:
        with TestClient(app) as client:
            headers = auth_headers(client)
            asset = upload_image(client, headers)

            self.assertTrue(asset["object_uri"].startswith("db://"))
            response = client.get(f"/api/media/{asset['id']}/file", headers=headers)

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, PNG_BYTES)
            self.assertEqual(response.headers["content-length"], str(len(PNG_BYTES)))

    def test_legacy_local_uri_falls_back_to_database_blob(self) -> None:
        with TestClient(app) as client:
            headers = auth_headers(client)
            asset = upload_image(client, headers)
            key = storage.key_from_uri(asset["object_uri"])

            with SessionLocal() as db:
                media = db.get(MediaAsset, asset["id"])
                assert media is not None
                media.object_uri = f"local://{key}"
                db.commit()

            response = client.get(f"/api/media/{asset['id']}/file", headers=headers)

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, PNG_BYTES)
            with SessionLocal() as db:
                media = db.get(MediaAsset, asset["id"])
                assert media is not None
                self.assertEqual(media.object_uri, asset["object_uri"])

    def test_missing_local_uri_returns_404(self) -> None:
        with TestClient(app) as client:
            headers = auth_headers(client)
            asset = upload_image(client, headers)

            with SessionLocal() as db:
                media = db.get(MediaAsset, asset["id"])
                assert media is not None
                media.object_uri = "local://missing/image.png"
                db.commit()

            response = client.get(f"/api/media/{asset['id']}/file", headers=headers)

            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["detail"], "Stored file not found")

    def test_ply_artifact_is_exportable(self) -> None:
        with TestClient(app) as client:
            headers = auth_headers(client)
            project_response = client.post(
                "/api/projects",
                json={"name": "ply export test", "input_type": "images", "tags": []},
                headers=headers,
            )
            project_response.raise_for_status()
            project_id = project_response.json()["id"]
            object_uri = storage.write_bytes("tests/model.ply", b"ply\n")

            with SessionLocal() as db:
                task = Task(project_id=project_id, type="preview", status="succeeded", current_stage="done")
                db.add(task)
                db.flush()
                artifact = Artifact(
                    project_id=project_id,
                    task_id=task.id,
                    kind="mesh_ply",
                    object_uri=object_uri,
                    file_name="model.ply",
                    file_size=4,
                )
                db.add(artifact)
                db.commit()
                artifact_id = artifact.id

            list_response = client.get(f"/api/projects/{project_id}/artifacts", headers=headers)
            list_response.raise_for_status()
            artifacts = list_response.json()["artifacts"]
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0]["file_name"], "model.ply")

            download_response = client.get(f"/api/artifacts/{artifact_id}/download-url", headers=headers)
            download_response.raise_for_status()

            file_response = client.get(download_response.json()["url"])
            self.assertEqual(file_response.status_code, 200)
            self.assertEqual(file_response.content, b"ply\n")
            self.assertIn("attachment", file_response.headers["content-disposition"])

    def test_intermediate_ply_metadata_is_exportable(self) -> None:
        with TestClient(app) as client:
            headers = auth_headers(client)
            project_response = client.post(
                "/api/projects",
                json={"name": "intermediate ply export test", "input_type": "images", "tags": []},
                headers=headers,
            )
            project_response.raise_for_status()
            project_id = project_response.json()["id"]
            spz_uri = storage.write_bytes("tests/preview.spz", b"spz")
            ply_path = TMP_PATH / "work" / "original.ply"
            ply_path.parent.mkdir(parents=True, exist_ok=True)
            ply_path.write_bytes(b"legacy ply\n")

            with SessionLocal() as db:
                task = Task(project_id=project_id, type="preview", status="succeeded", current_stage="done")
                db.add(task)
                db.flush()
                artifact = Artifact(
                    project_id=project_id,
                    task_id=task.id,
                    kind="preview_spz",
                    object_uri=spz_uri,
                    file_name="preview.spz",
                    file_size=3,
                    metadata_json={"intermediate_ply": str(ply_path)},
                )
                db.add(artifact)
                db.commit()
                artifact_id = artifact.id

            download_response = client.get(f"/api/artifacts/{artifact_id}/original-ply/download-url", headers=headers)
            download_response.raise_for_status()

            file_response = client.get(download_response.json()["url"])
            self.assertEqual(file_response.status_code, 200)
            self.assertEqual(file_response.content, b"legacy ply\n")
            self.assertIn("attachment", file_response.headers["content-disposition"])

    def test_fine_task_can_start_without_preview_after_three_images(self) -> None:
        with TestClient(app) as client, patch("app.main.enqueue_fine_task", return_value=None):
            headers = auth_headers(client)
            project_id = create_image_project(client, headers, "direct fine start")
            for index in range(8):
                upload_response = client.post(
                    f"/api/projects/{project_id}/media",
                    files={"file": (f"{index}.png", PNG_BYTES, "image/png")},
                    headers=headers,
                )
                upload_response.raise_for_status()

            response = client.post(f"/api/projects/{project_id}/tasks/fine", json={"options": {}}, headers=headers)
            response.raise_for_status()
            payload = response.json()

            self.assertEqual(payload["type"], "fine")
            self.assertEqual(payload["status"], "queued")
            self.assertEqual(payload["options"]["fine_pipeline"], "litevggt_fastgs_deblur_gsplat")
            self.assertEqual(payload["options"]["source_version"], 3)
            self.assertEqual(payload["options"]["fine_iterations"], 1000)
            self.assertNotIn("fine_amb3r_memory_device", payload["options"])
            self.assertNotIn("fine_amb3r_init_candidates", payload["options"])

    def test_fine_task_preserves_explicit_pycolmap_options(self) -> None:
        with TestClient(app) as client, patch("app.main.enqueue_fine_task", return_value=None):
            headers = auth_headers(client)
            project_id = create_image_project(client, headers, "explicit pycolmap options")
            for index in range(8):
                upload_response = client.post(
                    f"/api/projects/{project_id}/media",
                    files={"file": (f"{index}.png", PNG_BYTES, "image/png")},
                    headers=headers,
                )
                upload_response.raise_for_status()

            response = client.post(
                f"/api/projects/{project_id}/tasks/fine",
                json={"options": {"fine_sfm_backend": "colmap", "fine_colmap_threads": 4}},
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()

            self.assertEqual(payload["options"]["fine_sfm_backend"], "colmap")
            self.assertEqual(payload["options"]["fine_colmap_threads"], 4)

    def test_preview_rejects_legacy_edgs_pipeline(self) -> None:
        with TestClient(app) as client:
            headers = auth_headers(client)
            project_id = create_image_project(client, headers, "legacy edgs preview")
            upload_response = client.post(
                f"/api/projects/{project_id}/media",
                files={"file": ("image.png", PNG_BYTES, "image/png")},
                headers=headers,
            )
            upload_response.raise_for_status()

            response = client.post(
                f"/api/projects/{project_id}/tasks/preview",
                json={"options": {"preview_pipeline": "litevggt_edgs"}},
                headers=headers,
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("Unsupported preview pipeline", response.text)

    def test_preview_defaults_to_litevggt_spz(self) -> None:
        fake_redis = FakeRedis()
        with TestClient(app) as client, patch("app.main.get_redis", return_value=fake_redis):
            headers = auth_headers(client)
            project_id = create_image_project(client, headers, "default preview")
            upload_response = client.post(
                f"/api/projects/{project_id}/media",
                files={"file": ("image.png", PNG_BYTES, "image/png")},
                headers=headers,
            )
            upload_response.raise_for_status()

            response = client.post(f"/api/projects/{project_id}/tasks/preview", json={"options": {}}, headers=headers)

            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["options"]["preview_pipeline"], "litevggt_spz")

    def test_video_fine_task_uses_artdeco_speed3r_pipeline(self) -> None:
        with TestClient(app) as client, patch("app.main.enqueue_fine_task", return_value=None):
            headers = auth_headers(client)
            project_id = create_video_project(client, headers, "direct video fine start")
            upload_video(client, headers, project_id, b"not-a-real-video")

            response = client.post(f"/api/projects/{project_id}/tasks/fine", json={"options": {}}, headers=headers)
            response.raise_for_status()
            payload = response.json()

            self.assertEqual(payload["type"], "fine")
            self.assertEqual(payload["status"], "queued")
            self.assertEqual(payload["options"]["fine_pipeline"], "video_artdeco_speed3r")
            self.assertEqual(payload["options"]["source_version"], 1)
            self.assertNotIn("fine_amb3r_memory_device", payload["options"])
            self.assertNotIn("fine_amb3r_init_candidates", payload["options"])

    def test_video_preview_task_uses_lingbot_pipeline(self) -> None:
        fake_redis = FakeRedis()
        with TestClient(app) as client, patch("app.main.get_redis", return_value=fake_redis):
            headers = auth_headers(client)
            project_id = create_video_project(client, headers, "video preview start")
            upload_video(client, headers, project_id, b"not-a-real-video")

            response = client.post(f"/api/projects/{project_id}/tasks/preview", json={"options": {}}, headers=headers)
            response.raise_for_status()
            payload = response.json()

            self.assertEqual(payload["type"], "preview")
            self.assertEqual(payload["status"], "queued")
            self.assertEqual(payload["options"]["preview_pipeline"], "lingbot_map_spz")
            self.assertEqual(payload["options"]["source_version"], 1)
            self.assertIn(payload["id"], fake_redis.lists["preview_tasks"])

    def test_video_preview_task_rejects_missing_or_multiple_videos(self) -> None:
        fake_redis = FakeRedis()
        with TestClient(app) as client, patch("app.main.get_redis", return_value=fake_redis):
            headers = auth_headers(client)
            empty_project = create_video_project(client, headers, "empty video preview")

            empty_response = client.post(f"/api/projects/{empty_project}/tasks/preview", json={"options": {}}, headers=headers)
            self.assertEqual(empty_response.status_code, 400)

            multi_project = create_video_project(client, headers, "multi video preview")
            upload_video(client, headers, multi_project, b"video-1", "one.mp4")
            upload_video(client, headers, multi_project, b"video-2", "two.mp4")

            multi_response = client.post(f"/api/projects/{multi_project}/tasks/preview", json={"options": {}}, headers=headers)
            self.assertEqual(multi_response.status_code, 400)

    def test_video_fine_task_rejects_missing_or_multiple_videos(self) -> None:
        with TestClient(app) as client, patch("app.main.enqueue_fine_task", return_value=None):
            headers = auth_headers(client)
            empty_project = create_video_project(client, headers, "empty video fine")

            empty_response = client.post(f"/api/projects/{empty_project}/tasks/fine", json={"options": {}}, headers=headers)
            self.assertEqual(empty_response.status_code, 400)

            multi_project = create_video_project(client, headers, "multi video fine")
            upload_video(client, headers, multi_project, b"video-1", "one.mp4")
            upload_video(client, headers, multi_project, b"video-2", "two.mp4")

            multi_response = client.post(f"/api/projects/{multi_project}/tasks/fine", json={"options": {}}, headers=headers)
            self.assertEqual(multi_response.status_code, 400)

    def test_cancel_queued_task_removes_it_from_redis_queue(self) -> None:
        fake_redis = FakeRedis()
        with TestClient(app) as client, patch("app.main.get_redis", return_value=fake_redis):
            headers = auth_headers(client)
            project_id = create_image_project(client, headers, "cancel queued preview")
            upload_response = client.post(
                f"/api/projects/{project_id}/media",
                files={"file": ("one.png", PNG_BYTES, "image/png")},
                headers=headers,
            )
            upload_response.raise_for_status()
            task_response = client.post(f"/api/projects/{project_id}/tasks/preview", json={"options": {}}, headers=headers)
            task_response.raise_for_status()
            task_id = task_response.json()["id"]

            self.assertIn(task_id, fake_redis.lists["preview_tasks"])

            cancel_response = client.post(f"/api/tasks/{task_id}/cancel", headers=headers)
            cancel_response.raise_for_status()

            self.assertEqual(cancel_response.json()["status"], "canceled")
            self.assertNotIn(task_id, fake_redis.lists["preview_tasks"])
            self.assertIn(task_cancel_key(task_id), fake_redis.values)
            with SessionLocal() as db:
                task = db.get(Task, task_id)
                assert task is not None
                self.assertEqual(task.status, "canceled")

    def test_delete_project_requests_active_task_cancel(self) -> None:
        fake_redis = FakeRedis()
        with TestClient(app) as client, patch("app.main.get_redis", return_value=fake_redis):
            headers = auth_headers(client)
            project_id = create_image_project(client, headers, "delete cancels active task")
            upload_response = client.post(
                f"/api/projects/{project_id}/media",
                files={"file": ("one.png", PNG_BYTES, "image/png")},
                headers=headers,
            )
            upload_response.raise_for_status()
            task_response = client.post(f"/api/projects/{project_id}/tasks/preview", json={"options": {}}, headers=headers)
            task_response.raise_for_status()
            task_id = task_response.json()["id"]

            delete_response = client.delete(f"/api/projects/{project_id}", headers=headers)
            delete_response.raise_for_status()

            self.assertTrue(delete_response.json()["deleted"])
            self.assertIn(task_cancel_key(task_id), fake_redis.values)

    def test_bulk_delete_projects_requests_active_task_cancel(self) -> None:
        fake_redis = FakeRedis()
        with TestClient(app) as client, patch("app.main.get_redis", return_value=fake_redis):
            headers = auth_headers(client)
            running_project_id = create_image_project(client, headers, "bulk delete cancels active task")
            idle_project_id = create_image_project(client, headers, "bulk delete idle")
            upload_response = client.post(
                f"/api/projects/{running_project_id}/media",
                files={"file": ("one.png", PNG_BYTES, "image/png")},
                headers=headers,
            )
            upload_response.raise_for_status()
            task_response = client.post(f"/api/projects/{running_project_id}/tasks/preview", json={"options": {}}, headers=headers)
            task_response.raise_for_status()
            task_id = task_response.json()["id"]

            delete_response = client.post(
                "/api/projects/bulk-delete",
                json={"project_ids": [running_project_id, idle_project_id]},
                headers=headers,
            )
            delete_response.raise_for_status()

            payload = delete_response.json()
            self.assertEqual(payload["deleted"], 2)
            self.assertCountEqual(payload["project_ids"], [running_project_id, idle_project_id])
            self.assertIn(task_cancel_key(task_id), fake_redis.values)
            list_response = client.get("/api/projects", headers=headers)
            list_response.raise_for_status()
            remaining_ids = {item["id"] for item in list_response.json()["projects"]}
            self.assertNotIn(running_project_id, remaining_ids)
            self.assertNotIn(idle_project_id, remaining_ids)

    def test_viewer_config_prefers_fresh_final_spz(self) -> None:
        with TestClient(app) as client:
            headers = auth_headers(client)
            project_id = create_image_project(client, headers, "final viewer priority")
            preview_uri = storage.write_bytes("tests/preview-viewer.spz", b"preview")
            final_uri = storage.write_bytes("tests/final-viewer.spz", b"final")
            with SessionLocal() as db:
                task = Task(project_id=project_id, type="fine", status="succeeded", current_stage="done")
                db.add(task)
                db.flush()
                db.add(
                    Artifact(
                        project_id=project_id,
                        task_id=task.id,
                        kind="preview_spz",
                        object_uri=preview_uri,
                        file_name="preview.spz",
                        file_size=7,
                        source_version=0,
                    )
                )
                db.add(
                    Artifact(
                        project_id=project_id,
                        task_id=task.id,
                        kind="final_spz",
                        object_uri=final_uri,
                        file_name="final_web.spz",
                        file_size=5,
                        source_version=0,
                    )
                )
                db.commit()

            response = client.get(f"/api/projects/{project_id}/viewer-config", headers=headers)
            response.raise_for_status()
            payload = response.json()

            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["source"], "final")
            self.assertEqual(payload["format"], "spz")

    def test_chunked_upload_resumes_and_completes_file(self) -> None:
        content = b"0123456789abcdefghi"
        with TestClient(app) as client:
            headers = auth_headers(client)
            project_id = create_video_project(client, headers, "chunked upload")
            payload = chunk_check_payload("clip.mp4", content, chunk_size=6)

            check_response = client.post(f"/api/projects/{project_id}/uploads/check", json=payload, headers=headers)
            check_response.raise_for_status()
            upload_id = check_response.json()["upload_id"]
            self.assertEqual(check_response.json()["uploaded_chunks"], [])

            first_chunk = client.put(
                f"/api/uploads/{upload_id}/chunks/0",
                files={"file": ("chunk-0", content[:6], "application/octet-stream")},
                headers=headers,
            )
            first_chunk.raise_for_status()

            resume_response = client.post(f"/api/projects/{project_id}/uploads/check", json=payload, headers=headers)
            resume_response.raise_for_status()
            self.assertEqual(resume_response.json()["uploaded_chunks"], [0])

            for index, start in enumerate(range(6, len(content), 6), start=1):
                chunk_response = client.put(
                    f"/api/uploads/{upload_id}/chunks/{index}",
                    files={"file": (f"chunk-{index}", content[start : start + 6], "application/octet-stream")},
                    headers=headers,
                )
                chunk_response.raise_for_status()

            complete_response = client.post(f"/api/uploads/{upload_id}/complete", headers=headers)
            complete_response.raise_for_status()
            asset = complete_response.json()["media"]
            self.assertEqual(asset["file_size"], len(content))

            file_response = client.get(f"/api/media/{asset['id']}/file", headers=headers)
            self.assertEqual(file_response.status_code, 200)
            self.assertEqual(file_response.content, content)

    def test_chunked_upload_rejects_incomplete_complete(self) -> None:
        content = b"0123456789"
        with TestClient(app) as client:
            headers = auth_headers(client)
            project_id = create_video_project(client, headers, "incomplete chunked upload")
            payload = chunk_check_payload("clip.mp4", content, chunk_size=5)

            check_response = client.post(f"/api/projects/{project_id}/uploads/check", json=payload, headers=headers)
            check_response.raise_for_status()
            upload_id = check_response.json()["upload_id"]
            chunk_response = client.put(
                f"/api/uploads/{upload_id}/chunks/0",
                files={"file": ("chunk-0", content[:5], "application/octet-stream")},
                headers=headers,
            )
            chunk_response.raise_for_status()

            complete_response = client.post(f"/api/uploads/{upload_id}/complete", headers=headers)

            self.assertEqual(complete_response.status_code, 400)

    def test_chunked_upload_fast_returns_existing_media(self) -> None:
        content = b"fast upload content"
        with TestClient(app) as client:
            headers = auth_headers(client)
            project_id = create_video_project(client, headers, "fast upload")
            payload = chunk_check_payload("clip.mp4", content, chunk_size=8)

            check_response = client.post(f"/api/projects/{project_id}/uploads/check", json=payload, headers=headers)
            check_response.raise_for_status()
            upload_id = check_response.json()["upload_id"]
            for index, start in enumerate(range(0, len(content), 8)):
                chunk_response = client.put(
                    f"/api/uploads/{upload_id}/chunks/{index}",
                    files={"file": (f"chunk-{index}", content[start : start + 8], "application/octet-stream")},
                    headers=headers,
                )
                chunk_response.raise_for_status()
            complete_response = client.post(f"/api/uploads/{upload_id}/complete", headers=headers)
            complete_response.raise_for_status()
            media_id = complete_response.json()["media"]["id"]

            second_check = client.post(f"/api/projects/{project_id}/uploads/check", json=payload, headers=headers)
            second_check.raise_for_status()

            self.assertTrue(second_check.json()["completed"])
            self.assertEqual(second_check.json()["media"]["id"], media_id)


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    response.raise_for_status()
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def create_image_project(client: TestClient, headers: dict[str, str], name: str) -> str:
    project_response = client.post(
        "/api/projects",
        json={"name": name, "input_type": "images", "tags": []},
        headers=headers,
    )
    project_response.raise_for_status()
    return str(project_response.json()["id"])


def create_video_project(client: TestClient, headers: dict[str, str], name: str) -> str:
    project_response = client.post(
        "/api/projects",
        json={"name": name, "input_type": "video", "tags": []},
        headers=headers,
    )
    project_response.raise_for_status()
    return str(project_response.json()["id"])


def chunk_check_payload(file_name: str, content: bytes, chunk_size: int) -> dict[str, object]:
    return {
        "file_name": file_name,
        "file_size": len(content),
        "chunk_size": chunk_size,
        "total_chunks": (len(content) + chunk_size - 1) // chunk_size,
        "file_hash": hashlib.sha256(content).hexdigest(),
        "content_type": "video/mp4",
    }


def upload_image(client: TestClient, headers: dict[str, str]) -> dict[str, object]:
    project_response = client.post(
        "/api/projects",
        json={"name": "storage response test", "input_type": "images", "tags": []},
        headers=headers,
    )
    project_response.raise_for_status()
    project_id = project_response.json()["id"]
    upload_response = client.post(
        f"/api/projects/{project_id}/media",
        files={"file": ("one.png", PNG_BYTES, "image/png")},
        headers=headers,
    )
    upload_response.raise_for_status()
    return upload_response.json()


def upload_video(client: TestClient, headers: dict[str, str], project_id: str, content: bytes, file_name: str = "clip.mp4") -> dict[str, object]:
    upload_response = client.post(
        f"/api/projects/{project_id}/media",
        files={"file": (file_name, content, "video/mp4")},
        headers=headers,
    )
    upload_response.raise_for_status()
    return upload_response.json()


def tearDownModule() -> None:
    from app.database import engine

    engine.dispose()
    TMP.cleanup()


if __name__ == "__main__":
    unittest.main()
