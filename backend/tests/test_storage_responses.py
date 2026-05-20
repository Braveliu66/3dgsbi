from __future__ import annotations

import base64
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import select
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


TMP = tempfile.TemporaryDirectory()
TMP_PATH = Path(TMP.name)
os.environ["DATABASE_URL"] = f"sqlite:///{(TMP_PATH / 'app.db').as_posix()}"
os.environ["STORAGE_BACKEND"] = "db"
os.environ["STORAGE_ROOT"] = str(TMP_PATH / "storage")
os.environ["WORK_ROOT"] = str(TMP_PATH / "work")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.database import SessionLocal, initialize_database_schema  # noqa: E402
from app.fine.colmap_defaults import FINE_ITERATIONS, FINE_PIPELINE_NAME  # noqa: E402
from app.litevggt_defaults import LITEVGGT_DEFAULTS_PRESET_KEY, LITEVGGT_OFFICIAL_PAD_PRESET  # noqa: E402
from app.main import app, storage  # noqa: E402
from app.models import Artifact, MediaAsset, PipelineParameterDefault, Task  # noqa: E402
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
    def test_task_work_dir_uses_user_mode_type_time_without_changing_project_name(self) -> None:
        fake_redis = FakeRedis()
        with TestClient(app) as client, patch("app.main.get_redis", return_value=fake_redis):
            from app.worker import task_work_dir

            headers = auth_headers(client)
            project_id = create_image_project(client, headers, "preview naming")
            upload_response = client.post(
                f"/api/projects/{project_id}/media",
                files={"file": ("one.png", PNG_BYTES, "image/png")},
                headers=headers,
            )
            upload_response.raise_for_status()

            task_response = client.post(f"/api/projects/{project_id}/tasks/preview", json={"options": {}}, headers=headers)
            task_response.raise_for_status()
            task_id = task_response.json()["id"]

            project_response = client.get(f"/api/projects/{project_id}", headers=headers)
            project_response.raise_for_status()
            self.assertEqual(project_response.json()["name"], "preview naming")
            self.assertRegex(task_work_dir(task_id).name, r"^admin-preview-images-\d{20}$")
            self.assertNotEqual(task_work_dir(task_id).name, task_id)

    def test_db_upload_can_be_previewed(self) -> None:
        with TestClient(app) as client:
            headers = auth_headers(client)
            asset = upload_image(client, headers)

            self.assertTrue(asset["object_uri"].startswith("db://"))
            response = client.get(f"/api/media/{asset['id']}/file", headers=headers)

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, PNG_BYTES)
            self.assertEqual(response.headers["content-length"], str(len(PNG_BYTES)))

    def test_db_media_file_supports_range_response(self) -> None:
        with TestClient(app) as client:
            headers = auth_headers(client)
            asset = upload_image(client, headers)
            response = client.get(f"/api/media/{asset['id']}/file", headers={**headers, "Range": "bytes=1-3"})

            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.content, PNG_BYTES[1:4])
            self.assertEqual(response.headers["content-range"], f"bytes 1-3/{len(PNG_BYTES)}")
            self.assertEqual(response.headers["accept-ranges"], "bytes")

    def test_project_events_does_not_hold_db_connections_open(self) -> None:
        import app.database as database
        import app.main as main

        engine = create_engine(
            f"sqlite:///{(TMP_PATH / 'events_pool.db').as_posix()}",
            connect_args={"check_same_thread": False},
            pool_size=1,
            max_overflow=0,
            pool_timeout=1,
        )
        session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

        try:
            database.Base.metadata.create_all(bind=engine)
            with patch.object(database, "engine", engine), patch.object(database, "SessionLocal", session_local), patch.object(main, "engine", engine), patch.object(main, "SessionLocal", session_local):
                with TestClient(app) as client:
                    headers = auth_headers(client)
                    project_id = create_image_project(client, headers, "event pool")

                    with client.stream("GET", f"/api/projects/{project_id}/events", headers=headers) as stream:
                        first_line = next(stream.iter_lines())
                        self.assertTrue(first_line)

                        login_response = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
                        self.assertEqual(login_response.status_code, 200)
        finally:
            engine.dispose()

    def test_pipeline_parameter_schema_lists_supported_pipelines_and_fields(self) -> None:
        with TestClient(app) as client:
            response = client.get("/api/pipeline-parameters/schema")
            response.raise_for_status()
            payload = response.json()

            pipelines = {item["pipeline"]: item for item in payload["pipelines"]}
            self.assertEqual({item["value"] for item in payload["scene_types"]}, {"indoor", "outdoor"})
            self.assertIn("litevggt_spz", pipelines)
            self.assertNotIn("lingbot_video_pointcloud_fast", pipelines)
            self.assertIn(FINE_PIPELINE_NAME, pipelines)
            litevggt_keys = {item["key"] for item in pipelines["litevggt_spz"]["fields"]}
            self.assertIn("preview_max_points", litevggt_keys)
            self.assertIn("preview_fixed_splat_radius_scale", litevggt_keys)
            self.assertNotIn("preview_video_fps", litevggt_keys)
            self.assertNotIn("preview_video_max_frames", litevggt_keys)
            colmap_keys = {item["key"] for item in pipelines[FINE_PIPELINE_NAME]["fields"]}
            self.assertIn("fine_sfm_backend", colmap_keys)
            self.assertIn("fine_colmap_matcher", colmap_keys)
            self.assertIn("fine_blur_reject_ratio", colmap_keys)
            self.assertIn("fine_eap_enabled", colmap_keys)
            self.assertIn("fine_eap_dbscan_eps", colmap_keys)
            self.assertIn("fine_eap_min_samples", colmap_keys)
            self.assertIn("fine_eap_mask_radius", colmap_keys)
            self.assertIn("fine_eap_max_point_multiplier", colmap_keys)
            self.assertIn("fine_gsplat_enabled", colmap_keys)
            self.assertIn("blur_code_dim", colmap_keys)
            colmap_fields = {item["key"]: item for item in pipelines[FINE_PIPELINE_NAME]["fields"]}
            self.assertEqual(colmap_fields["fine_sfm_backend"]["label"], "SfM 后端")
            self.assertEqual(colmap_fields["fine_sfm_backend"]["group"], "COLMAP")
            self.assertEqual(colmap_fields["fine_trainer_repo"]["type"], "text")
            self.assertEqual(colmap_fields["fine_eap_enabled"]["type"], "boolean")
            self.assertEqual(colmap_fields["fine_eap_dbscan_eps"]["type"], "number")
            self.assertEqual(colmap_fields["fine_gsplat_enabled"]["type"], "boolean")
            self.assertEqual(colmap_fields["blur_code_dim"]["type"], "select")
            self.assertEqual(colmap_fields["blur_code_dim"]["options"], ["4", "8", "16"])
            self.assertEqual(colmap_fields["fine_deblur_mode"]["option_labels"]["motion"], "运动模糊")
            self.assertEqual(colmap_fields["fine_deblur_mode"]["options"], ["motion", "defocus", "sharp"])
            self.assertEqual(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["pts_N_pts"], 0)
            self.assertEqual(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["pts_iter"], 999999)
            self.assertEqual(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["pts_rate"], 0.0)
            self.assertEqual(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["resolution"], -1)
            self.assertEqual(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["fine_deblur_mode"], "motion")
            self.assertEqual(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["blur_code_dim"], 4)
            self.assertEqual(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["fine_image_max_side"], 1500)
            self.assertEqual(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["fine_colmap_max_image_size"], 1080)
            self.assertEqual(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["fine_blur_reject_ratio"], 0.0)
            self.assertTrue(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["fine_eap_enabled"])
            self.assertEqual(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["fine_eap_dbscan_eps"], 30)
            self.assertEqual(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["fine_eap_min_samples"], 10)
            self.assertEqual(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["fine_eap_mask_radius"], 20)
            self.assertEqual(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["fine_eap_max_point_multiplier"], 10)
            self.assertTrue(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["fine_gsplat_enabled"])
            self.assertTrue(pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"]["densify_with_depth"])
            self.assertNotIn("dash_enable", pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"])
            self.assertNotIn("Grouping", pipelines[FINE_PIPELINE_NAME]["defaults"]["indoor"])

    def test_pipeline_parameter_defaults_are_admin_only_and_filter_unknown_keys(self) -> None:
        clear_pipeline_defaults()
        with TestClient(app) as client:
            admin_headers = auth_headers(client)
            user_headers = non_admin_headers(client, "pipeline-user")

            denied = client.put(
                "/api/admin/pipeline-parameter-defaults/litevggt_spz/outdoor",
                json={"options": {"preview_max_points": 1234}},
                headers=user_headers,
            )
            self.assertEqual(denied.status_code, 403)

            saved = client.put(
                "/api/admin/pipeline-parameter-defaults/litevggt_spz/outdoor",
                json={"options": {"preview_max_points": 1234, "unknown_option": True}},
                headers=admin_headers,
            )
            saved.raise_for_status()
            self.assertEqual(saved.json()["options"]["preview_max_points"], 1234)
            self.assertNotIn("unknown_option", saved.json()["options"])
            with SessionLocal() as db:
                row = db.scalar(
                    select(PipelineParameterDefault).where(
                        PipelineParameterDefault.pipeline == "litevggt_spz",
                        PipelineParameterDefault.scene_type == "outdoor",
                    )
                )
            self.assertIsNotNone(row)
            self.assertEqual(row.options[LITEVGGT_DEFAULTS_PRESET_KEY], LITEVGGT_OFFICIAL_PAD_PRESET)

            loaded = client.get("/api/admin/pipeline-parameter-defaults", headers=admin_headers)
            loaded.raise_for_status()
            self.assertEqual(loaded.json()["defaults"]["litevggt_spz"]["outdoor"]["preview_max_points"], 1234)
        clear_pipeline_defaults()

    def test_lingbot_pipeline_parameter_defaults_are_not_supported(self) -> None:
        clear_pipeline_defaults()
        with TestClient(app) as client:
            admin_headers = auth_headers(client)

            response = client.put(
                "/api/admin/pipeline-parameter-defaults/lingbot_video_pointcloud_fast/outdoor",
                json={"options": {"preview_lingbot_preprocess_mode": "pad", "unknown_option": True}},
                headers=admin_headers,
            )
            self.assertEqual(response.status_code, 404)

            loaded = client.get("/api/admin/pipeline-parameter-defaults", headers=admin_headers)
            loaded.raise_for_status()
            self.assertNotIn("lingbot_video_pointcloud_fast", loaded.json()["defaults"])
        clear_pipeline_defaults()

    def test_task_options_merge_scene_defaults_and_request_overrides(self) -> None:
        clear_pipeline_defaults()
        fake_redis = FakeRedis()
        with TestClient(app) as client, patch("app.main.get_redis", return_value=fake_redis):
            headers = auth_headers(client)
            saved = client.put(
                "/api/admin/pipeline-parameter-defaults/litevggt_spz/outdoor",
                json={"options": {"preview_max_points": 1234, "litevggt_target_size": 240}},
                headers=headers,
            )
            saved.raise_for_status()
            project_id = create_image_project(client, headers, "parameter defaults preview")
            upload_response = client.post(
                f"/api/projects/{project_id}/media",
                files={"file": ("image.png", PNG_BYTES, "image/png")},
                headers=headers,
            )
            upload_response.raise_for_status()

            response = client.post(
                f"/api/projects/{project_id}/tasks/preview",
                json={"options": {"scene_type": "outdoor", "litevggt_target_size": 336}},
                headers=headers,
            )
            response.raise_for_status()
            options = response.json()["options"]

            self.assertEqual(options["scene_type"], "outdoor")
            self.assertEqual(options["preview_scene_profile"], "outdoor_fast_clean")
            self.assertEqual(options["preview_max_points"], 1234)
            self.assertEqual(options["litevggt_target_size"], 336)
        clear_pipeline_defaults()

    def test_litevggt_stale_saved_defaults_without_preset_marker_are_ignored(self) -> None:
        clear_pipeline_defaults()
        with SessionLocal() as db:
            db.add(
                PipelineParameterDefault(
                    pipeline="litevggt_spz",
                    scene_type="indoor",
                    options={
                        "litevggt_keep_ratio": 0.55,
                        "preview_max_points": 5_000_000,
                        "litevggt_target_size": 476,
                        "litevggt_point_selection_strategy": "scene_coverage",
                    },
                )
            )
            db.commit()

        fake_redis = FakeRedis()
        with TestClient(app) as client, patch("app.main.get_redis", return_value=fake_redis):
            headers = auth_headers(client)
            loaded = client.get("/api/admin/pipeline-parameter-defaults", headers=headers)
            loaded.raise_for_status()
            defaults = loaded.json()["defaults"]["litevggt_spz"]["indoor"]
            self.assertEqual(defaults["litevggt_keep_ratio"], 0.46)
            self.assertEqual(defaults["preview_max_points"], 3_200_000)
            self.assertEqual(defaults["litevggt_target_size"], 420)
            self.assertEqual(defaults["litevggt_point_selection_strategy"], "scene_coverage")

            project_id = create_image_project(client, headers, "stale litevggt defaults preview")
            upload_response = client.post(
                f"/api/projects/{project_id}/media",
                files={"file": ("image.png", PNG_BYTES, "image/png")},
                headers=headers,
            )
            upload_response.raise_for_status()

            response = client.post(
                f"/api/projects/{project_id}/tasks/preview",
                json={"options": {"scene_type": "indoor"}},
                headers=headers,
            )
            response.raise_for_status()
            options = response.json()["options"]
            self.assertEqual(options["litevggt_keep_ratio"], 0.46)
            self.assertEqual(options["preview_max_points"], 3_200_000)
            self.assertEqual(options["litevggt_target_size"], 420)
            self.assertEqual(options["litevggt_point_selection_strategy"], "scene_coverage")
            self.assertEqual(options["litevggt_parameter_sources"]["litevggt_target_size"], "system_default")
        clear_pipeline_defaults()

    def test_colmap_defaults_save_and_fine_task_uses_them(self) -> None:
        clear_pipeline_defaults()
        with TestClient(app) as client, patch("app.main.enqueue_fine_task", return_value=None):
            headers = auth_headers(client)
            saved = client.put(
                f"/api/admin/pipeline-parameter-defaults/{FINE_PIPELINE_NAME}/indoor",
                json={"options": {"fine_colmap_threads": 4}},
                headers=headers,
            )
            saved.raise_for_status()
            self.assertEqual(saved.json()["options"]["fine_colmap_threads"], 4)

            project_id = create_image_project(client, headers, "colmap fine")
            for index in range(3):
                upload_response = client.post(
                    f"/api/projects/{project_id}/media",
                    files={"file": (f"{index}.png", PNG_BYTES, "image/png")},
                    headers=headers,
                )
                upload_response.raise_for_status()

            response = client.post(f"/api/projects/{project_id}/tasks/fine", json={"options": {"scene_type": "indoor"}}, headers=headers)
            response.raise_for_status()

            self.assertEqual(response.json()["options"]["fine_colmap_threads"], 4)
        clear_pipeline_defaults()

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

    def test_fine_task_can_start_without_preview_after_eight_images(self) -> None:
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
            self.assertEqual(payload["options"]["fine_pipeline"], FINE_PIPELINE_NAME)
            self.assertEqual(payload["options"]["fine_sfm_backend"], "colmap_global")
            self.assertEqual(payload["options"]["quality_mode"], "auto")
            self.assertEqual(payload["options"]["camera_distortion"], "undistorted")
            self.assertTrue(payload["options"]["prefer_gpu"])
            self.assertNotIn("fastgs_target", payload["options"])
            self.assertEqual(payload["options"]["fine_scene_profile"], "indoor_full")
            self.assertEqual(payload["options"]["scene_type"], "indoor")
            self.assertEqual(payload["options"]["fine_scene_type"], "indoor")
            self.assertEqual(payload["options"]["input_type"], "images")
            self.assertEqual(payload["options"]["source_version"], 8)
            self.assertEqual(payload["options"]["fine_iterations"], FINE_ITERATIONS)
            self.assertTrue(payload["options"]["fine_eap_enabled"])
            self.assertFalse(payload["options"]["fine_gsplat_enabled"])
            self.assertNotIn("fine_amb3r_memory_device", payload["options"])
            self.assertNotIn("fine_amb3r_init_candidates", payload["options"])

    def test_fine_task_accepts_scene_profile(self) -> None:
        with TestClient(app) as client, patch("app.main.enqueue_fine_task", return_value=None):
            headers = auth_headers(client)
            project_id = create_image_project(client, headers, "indoor fine start")
            for index in range(8):
                upload_response = client.post(
                    f"/api/projects/{project_id}/media",
                    files={"file": (f"{index}.png", PNG_BYTES, "image/png")},
                    headers=headers,
                )
                upload_response.raise_for_status()

            response = client.post(
                f"/api/projects/{project_id}/tasks/fine",
                json={"options": {"fine_scene_profile": "indoor_full"}},
                headers=headers,
            )
            response.raise_for_status()

            self.assertEqual(response.json()["options"]["fine_scene_profile"], "indoor_full")
            self.assertEqual(response.json()["options"]["scene_type"], "indoor")
            self.assertEqual(response.json()["options"]["fine_scene_type"], "indoor")

    def test_fine_task_accepts_scene_type(self) -> None:
        with TestClient(app) as client, patch("app.main.enqueue_fine_task", return_value=None):
            headers = auth_headers(client)
            project_id = create_image_project(client, headers, "outdoor fine start")
            for index in range(8):
                upload_response = client.post(
                    f"/api/projects/{project_id}/media",
                    files={"file": (f"{index}.png", PNG_BYTES, "image/png")},
                    headers=headers,
                )
                upload_response.raise_for_status()

            response = client.post(
                f"/api/projects/{project_id}/tasks/fine",
                json={"options": {"scene_type": "outdoor"}},
                headers=headers,
            )
            response.raise_for_status()

            options = response.json()["options"]
            self.assertEqual(options["scene_type"], "outdoor")
            self.assertEqual(options["fine_scene_type"], "outdoor")
            self.assertEqual(options["fine_scene_profile"], "outdoor_fast_clean")

    def test_fine_task_preserves_explicit_colmap_cli_options(self) -> None:
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

    def test_fine_task_rejects_invalid_scene_type_auto(self) -> None:
        with TestClient(app) as client, patch("app.main.enqueue_fine_task", return_value=None):
            headers = auth_headers(client)
            project_id = create_image_project(client, headers, "invalid fine scene")
            for index in range(8):
                upload_response = client.post(
                    f"/api/projects/{project_id}/media",
                    files={"file": (f"{index}.png", PNG_BYTES, "image/png")},
                    headers=headers,
                )
                upload_response.raise_for_status()

            response = client.post(
                f"/api/projects/{project_id}/tasks/fine",
                json={"options": {"scene_type": "auto"}},
                headers=headers,
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("Unsupported scene_type", response.text)

    def test_fine_task_rejects_input_type_mismatch(self) -> None:
        with TestClient(app) as client, patch("app.main.enqueue_fine_task", return_value=None):
            headers = auth_headers(client)
            project_id = create_image_project(client, headers, "input type mismatch")
            for index in range(8):
                upload_response = client.post(
                    f"/api/projects/{project_id}/media",
                    files={"file": (f"{index}.png", PNG_BYTES, "image/png")},
                    headers=headers,
                )
                upload_response.raise_for_status()

            response = client.post(
                f"/api/projects/{project_id}/tasks/fine",
                json={"options": {"input_type": "video", "scene_type": "indoor"}},
                headers=headers,
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("input_type must match", response.text)

    def test_fine_task_rejects_edgs_option(self) -> None:
        with TestClient(app) as client, patch("app.main.enqueue_fine_task", return_value=None):
            headers = auth_headers(client)
            project_id = create_image_project(client, headers, "removed edgs option")
            for index in range(8):
                upload_response = client.post(
                    f"/api/projects/{project_id}/media",
                    files={"file": (f"{index}.png", PNG_BYTES, "image/png")},
                    headers=headers,
                )
                upload_response.raise_for_status()

            response = client.post(
                f"/api/projects/{project_id}/tasks/fine",
                json={"options": {"fine_edgs_enabled": True}},
                headers=headers,
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("EDGS/RoMA dense initialization has been removed", response.text)

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
            self.assertEqual(response.json()["options"]["preview_scene_profile"], "indoor_full")
            self.assertEqual(response.json()["options"]["scene_type"], "indoor")

    def test_image_preview_accepts_scene_profiles(self) -> None:
        for profile in ("indoor_full", "outdoor_fast_clean"):
            with self.subTest(profile=profile):
                fake_redis = FakeRedis()
                with TestClient(app) as client, patch("app.main.get_redis", return_value=fake_redis):
                    headers = auth_headers(client)
                    project_id = create_image_project(client, headers, f"{profile} preview")
                    upload_response = client.post(
                        f"/api/projects/{project_id}/media",
                        files={"file": ("image.png", PNG_BYTES, "image/png")},
                        headers=headers,
                    )
                    upload_response.raise_for_status()

                    response = client.post(
                        f"/api/projects/{project_id}/tasks/preview",
                        json={"options": {"preview_scene_profile": profile}},
                        headers=headers,
                    )

                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["options"]["preview_scene_profile"], profile)

    def test_image_preview_rejects_invalid_scene_profile(self) -> None:
        with TestClient(app) as client:
            headers = auth_headers(client)
            project_id = create_image_project(client, headers, "invalid profile preview")
            upload_response = client.post(
                f"/api/projects/{project_id}/media",
                files={"file": ("image.png", PNG_BYTES, "image/png")},
                headers=headers,
            )
            upload_response.raise_for_status()

            response = client.post(
                f"/api/projects/{project_id}/tasks/preview",
                json={"options": {"preview_scene_profile": "auto"}},
                headers=headers,
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("Unsupported preview scene profile", response.text)

    def test_video_fine_task_is_enabled(self) -> None:
        with TestClient(app) as client, patch("app.main.enqueue_fine_task", return_value=None):
            headers = auth_headers(client)
            project_id = create_video_project(client, headers, "direct video fine start")
            upload_video(client, headers, project_id, b"not-a-real-video")

            response = client.post(f"/api/projects/{project_id}/tasks/fine", json={"options": {"scene_type": "indoor"}}, headers=headers)

            response.raise_for_status()
            payload = response.json()
            self.assertEqual(payload["type"], "fine")
            self.assertEqual(payload["status"], "queued")
            self.assertEqual(payload["options"]["input_type"], "video")
            self.assertEqual(payload["options"]["scene_type"], "indoor")
            self.assertEqual(payload["options"]["fine_sfm_backend"], "colmap_global")

    def test_video_preview_task_uses_litevggt_pipeline(self) -> None:
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
            self.assertEqual(payload["options"]["pipeline"], "litevggt_spz")
            self.assertEqual(payload["options"]["preview_pipeline"], "litevggt_spz")
            self.assertEqual(payload["options"]["scene_type"], "indoor")
            self.assertEqual(payload["options"]["preview_video_max_frames"], 48)
            self.assertEqual(payload["options"]["litevggt_max_input_frames"], 48)
            self.assertEqual(payload["options"]["litevggt_inference_mode"], "single")
            self.assertEqual(payload["options"]["litevggt_point_selection_strategy"], "scene_coverage")
            self.assertEqual(payload["options"]["source_version"], 1)
            self.assertIn(payload["id"], fake_redis.lists["preview_tasks"])

    def test_video_preview_task_rejects_legacy_lingbot_spz_pipeline(self) -> None:
        fake_redis = FakeRedis()
        with TestClient(app) as client, patch("app.main.get_redis", return_value=fake_redis):
            headers = auth_headers(client)
            project_id = create_video_project(client, headers, "legacy video preview start")
            upload_video(client, headers, project_id, b"not-a-real-video")

            response = client.post(
                f"/api/projects/{project_id}/tasks/preview",
                json={"options": {"preview_pipeline": "lingbot_map_spz", "scene_type": "outdoor"}},
                headers=headers,
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("Unsupported preview pipeline for video input: lingbot_map_spz", response.text)

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
            with SessionLocal() as db:
                task = db.get(Task, task_id)
                assert task is not None
                task.status = "running"
                task.worker_id = "preview-worker-test"
                db.commit()

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
                self.assertIsNone(task.worker_id)

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
            final_ply_uri = storage.write_bytes("tests/final-viewer.ply", b"ply")
            viewer_meta_uri = storage.write_bytes("tests/final-viewer-meta.json", b"{}")
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
                db.add(
                    Artifact(
                        project_id=project_id,
                        task_id=task.id,
                        kind="final_ply",
                        object_uri=final_ply_uri,
                        file_name="final.ply",
                        file_size=3,
                        source_version=0,
                    )
                )
                db.add(
                    Artifact(
                        project_id=project_id,
                        task_id=task.id,
                        kind="viewer_meta_json",
                        object_uri=viewer_meta_uri,
                        file_name="final_viewer_meta.json",
                        file_size=2,
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
            self.assertEqual(payload["file_size"], 5)
            self.assertIn("/api/artifacts/", payload["model_url"])
            self.assertIn("/api/artifacts/", payload["gaussian_ply_url"])
            self.assertIn("/api/artifacts/", payload["viewer_meta_url"])
            self.assertIn("/api/artifacts/", payload["download_spz_url"])
            self.assertIn("/api/artifacts/", payload["download_ply_url"])

    def test_viewer_config_supports_ply_only_preview(self) -> None:
        with TestClient(app) as client:
            headers = auth_headers(client)
            project_id = create_video_project(client, headers, "ply preview viewer")
            fast_uri = storage.write_bytes("tests/preview-fast.ply", b"fast")
            full_uri = storage.write_bytes("tests/preview-full.ply", b"full")
            meta_uri = storage.write_bytes("tests/preview-meta.json", b"{}")
            camera_path_uri = storage.write_bytes("tests/camera-path.json", b'{"poses":[],"frames":[]}')
            with SessionLocal() as db:
                task = Task(project_id=project_id, type="preview", status="succeeded", current_stage="done")
                db.add(task)
                db.flush()
                db.add(
                    Artifact(
                        project_id=project_id,
                        task_id=task.id,
                        kind="preview_pointcloud_ply",
                        object_uri=fast_uri,
                        file_name="preview_fast.ply",
                        file_size=4,
                        source_version=0,
                        metadata_json={
                            "point_source": "world_points_from_depth",
                            "format": "ply",
                            "scene_type": "outdoor",
                            "artifact_display": "pointcloud",
                            "viewer_default_point_size": 0.00001,
                            "viewer_default_downsample_factor": 10,
                            "viewer_default_conf_threshold": 1.5,
                        },
                    )
                )
                db.add(
                    Artifact(
                        project_id=project_id,
                        task_id=task.id,
                        kind="preview_full_ply",
                        object_uri=full_uri,
                        file_name="preview_full.ply",
                        file_size=4,
                        source_version=0,
                    )
                )
                db.add(
                    Artifact(
                        project_id=project_id,
                        task_id=task.id,
                        kind="preview_meta_json",
                        object_uri=meta_uri,
                        file_name="preview_meta.json",
                        file_size=2,
                        source_version=0,
                    )
                )
                db.add(
                    Artifact(
                        project_id=project_id,
                        task_id=task.id,
                        kind="camera_path_json",
                        object_uri=camera_path_uri,
                        file_name="camera_path.json",
                        file_size=23,
                        source_version=0,
                    )
                )
                db.commit()

            response = client.get(f"/api/projects/{project_id}/viewer-config", headers=headers)
            response.raise_for_status()
            payload = response.json()

            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["source"], "preview")
            self.assertEqual(payload["format"], "ply")
            self.assertEqual(payload["artifact_kind"], "preview_pointcloud_ply")
            self.assertIsNone(payload["model_url"])
            self.assertIsNone(payload["download_spz_url"])
            self.assertEqual(payload["point_source"], "world_points_from_depth")
            self.assertEqual(payload["scene_type"], "outdoor")
            self.assertEqual(payload["artifact_display"], "pointcloud")
            self.assertEqual(payload["viewer_default_point_size"], 0.00001)
            self.assertEqual(payload["viewer_default_downsample_factor"], 10)
            self.assertEqual(payload["viewer_default_conf_threshold"], 1.5)
            self.assertIn("/api/artifacts/", payload["debug_points_ply_url"])
            self.assertIn("/api/artifacts/", payload["download_ply_url"])
            self.assertIn("/api/artifacts/", payload["preview_meta_url"])
            self.assertIn("/api/artifacts/", payload["camera_path_url"])

    def test_project_share_returns_public_viewer_downloads_and_revokes(self) -> None:
        with TestClient(app) as client:
            headers = auth_headers(client)
            project_id = create_image_project(client, headers, "shared viewer")
            spz_uri = storage.write_bytes("tests/shared-final.spz", b"spz")
            ply_uri = storage.write_bytes("tests/shared-final.ply", b"ply")
            with SessionLocal() as db:
                task = Task(project_id=project_id, type="fine", status="succeeded", current_stage="done")
                db.add(task)
                db.flush()
                db.add(
                    Artifact(
                        project_id=project_id,
                        task_id=task.id,
                        kind="final_spz",
                        object_uri=spz_uri,
                        file_name="final_web.spz",
                        file_size=3,
                        source_version=0,
                    )
                )
                db.add(
                    Artifact(
                        project_id=project_id,
                        task_id=task.id,
                        kind="final_ply",
                        object_uri=ply_uri,
                        file_name="final.ply",
                        file_size=3,
                        source_version=0,
                    )
                )
                db.commit()

            share_response = client.post(f"/api/projects/{project_id}/share", headers=headers)
            share_response.raise_for_status()
            share_payload = share_response.json()
            token = share_payload["share_token"]
            self.assertEqual(share_payload["share_url"], f"/share/{token}")

            public_response = client.get(f"/api/shared-projects/{token}")
            public_response.raise_for_status()
            public_payload = public_response.json()
            self.assertEqual(public_payload["id"], project_id)
            self.assertEqual(public_payload["total_size_bytes"], 0)
            self.assertEqual(public_payload["viewer"]["source"], "final")
            self.assertEqual(public_payload["viewer"]["file_size"], 3)
            self.assertIn("/api/artifacts/", public_payload["viewer"]["download_spz_url"])
            self.assertIn("/api/artifacts/", public_payload["viewer"]["download_ply_url"])

            delete_response = client.delete(f"/api/projects/{project_id}/share", headers=headers)
            delete_response.raise_for_status()
            self.assertEqual(client.get(f"/api/shared-projects/{token}").status_code, 404)

    def test_chunked_upload_accepts_lightweight_signature_without_pre_hash(self) -> None:
        content = b"quick first upload content"
        with TestClient(app) as client:
            headers = auth_headers(client)
            project_id = create_video_project(client, headers, "lightweight upload")
            payload = {
                "file_name": "clip.mp4",
                "file_size": len(content),
                "chunk_size": 8,
                "total_chunks": (len(content) + 7) // 8,
                "file_signature": "clip.mp4|24|123|video/mp4",
                "content_type": "video/mp4",
            }

            check_response = client.post(f"/api/projects/{project_id}/uploads/check", json=payload, headers=headers)
            check_response.raise_for_status()
            upload_id = check_response.json()["upload_id"]

            for index, start in enumerate(range(0, len(content), 8)):
                chunk_response = client.put(
                    f"/api/uploads/{upload_id}/chunks/{index}/raw",
                    content=content[start : start + 8],
                    headers={**headers, "Content-Type": "application/octet-stream"},
                )
                chunk_response.raise_for_status()

            complete_response = client.post(f"/api/uploads/{upload_id}/complete", headers=headers)
            complete_response.raise_for_status()
            asset = complete_response.json()["media"]

            self.assertEqual(asset["file_size"], len(content))
            self.assertEqual(client.get(f"/api/media/{asset['id']}/file", headers=headers).content, content)

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

    def test_chunked_upload_lists_media_by_client_order_not_completion_order(self) -> None:
        first_content = b"first upload content"
        second_content = b"second upload content"
        with TestClient(app) as client:
            headers = auth_headers(client)
            project_id = create_video_project(client, headers, "ordered chunked upload")
            first_payload = {**chunk_check_payload("first.mp4", first_content, chunk_size=8), "client_order": 0}
            second_payload = {**chunk_check_payload("second.mp4", second_content, chunk_size=8), "client_order": 1}

            first_check = client.post(f"/api/projects/{project_id}/uploads/check", json=first_payload, headers=headers)
            second_check = client.post(f"/api/projects/{project_id}/uploads/check", json=second_payload, headers=headers)
            first_check.raise_for_status()
            second_check.raise_for_status()
            first_upload_id = first_check.json()["upload_id"]
            second_upload_id = second_check.json()["upload_id"]

            for upload_id, content in ((second_upload_id, second_content), (first_upload_id, first_content)):
                for index, start in enumerate(range(0, len(content), 8)):
                    chunk_response = client.put(
                        f"/api/uploads/{upload_id}/chunks/{index}/raw",
                        content=content[start : start + 8],
                        headers={**headers, "Content-Type": "application/octet-stream"},
                    )
                    chunk_response.raise_for_status()
                complete_response = client.post(f"/api/uploads/{upload_id}/complete", headers=headers)
                complete_response.raise_for_status()

            media_response = client.get(f"/api/projects/{project_id}/media", headers=headers)
            media_response.raise_for_status()
            media_names = [item["file_name"] for item in media_response.json()["media"]]
            self.assertEqual(media_names, ["first.mp4", "second.mp4"])

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


def non_admin_headers(client: TestClient, username: str) -> dict[str, str]:
    client.post("/api/auth/register", json={"username": username, "password": "pw123456"})
    response = client.post("/api/auth/login", json={"username": username, "password": "pw123456"})
    response.raise_for_status()
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def clear_pipeline_defaults() -> None:
    initialize_database_schema()
    with SessionLocal() as db:
        for row in db.query(PipelineParameterDefault).all():
            db.delete(row)
        db.commit()


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
