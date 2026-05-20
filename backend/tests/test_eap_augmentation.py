from __future__ import annotations

import sys
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fine.eap import (  # noqa: E402
    _build_eap_feature_command,
    _camera_params_text,
    _check_point_multiplier,
    _install_eap_points,
    _single_undistorted_camera,
    _sync_eap_database_single_camera,
    _write_seed_text_model,
)
from app.fine.types import FineFailure  # noqa: E402


class EapAugmentationTests(unittest.TestCase):
    def test_seed_model_reuses_original_camera_pose_for_aug_images(self) -> None:
        reconstruction = SimpleNamespace(
            cameras={
                7: SimpleNamespace(
                    camera_id=7,
                    model="PINHOLE",
                    width=640,
                    height=480,
                    params=[500.0, 510.0, 320.0, 240.0],
                )
            },
            images={
                3: SimpleNamespace(
                    name="frame.jpg",
                    camera_id=7,
                    qvec=[1.0, 0.0, 0.0, 0.0],
                    tvec=[1.0, 2.0, 3.0],
                )
            },
        )

        with tempfile.TemporaryDirectory() as tmp:
            seed_dir = Path(tmp) / "seed"
            _write_seed_text_model(seed_dir, reconstruction, {"frame.jpg": 2, "frame_aug.jpg": 9})
            images_text = (seed_dir / "images.txt").read_text(encoding="utf-8")
            cameras_text = (seed_dir / "cameras.txt").read_text(encoding="utf-8")

        self.assertIn("2 1 0 0 0 1 2 3 7 frame.jpg", images_text)
        self.assertIn("9 1 0 0 0 1 2 3 7 frame_aug.jpg", images_text)
        self.assertIn("7 PINHOLE 640 480 500 510 320 240", cameras_text)

    def test_eap_single_camera_uses_undistorted_pinhole_params(self) -> None:
        reconstruction = SimpleNamespace(
            cameras={
                1: SimpleNamespace(
                    camera_id=1,
                    model="PINHOLE",
                    width=1503,
                    height=1000,
                    params=[1046.24075173, 1046.24075173, 751.5, 500.0],
                )
            }
        )

        camera = _single_undistorted_camera(reconstruction)

        self.assertEqual(camera.model, "PINHOLE")
        self.assertEqual(_camera_params_text(camera.params), "1046.24075173,1046.24075173,751.5,500")

        command = _build_eap_feature_command(
            "colmap",
            Path("database.db"),
            Path("images"),
            camera,
            use_gpu="1",
            gpu_index="0",
        )
        self.assertIn("--ImageReader.single_camera", command)
        self.assertEqual(command[command.index("--ImageReader.single_camera") + 1], "1")
        self.assertEqual(command[command.index("--ImageReader.camera_model") + 1], "PINHOLE")
        self.assertEqual(command[command.index("--ImageReader.camera_params") + 1], "1046.24075173,1046.24075173,751.5,500")

    def test_eap_rejects_multi_camera_sparse_model(self) -> None:
        reconstruction = SimpleNamespace(
            cameras={
                1: SimpleNamespace(camera_id=1, model="PINHOLE", width=640, height=480, params=[500.0, 500.0, 320.0, 240.0]),
                2: SimpleNamespace(camera_id=2, model="PINHOLE", width=640, height=480, params=[500.0, 500.0, 320.0, 240.0]),
            }
        )

        with self.assertRaises(FineFailure) as raised:
            _single_undistorted_camera(reconstruction)

        self.assertEqual(raised.exception.code, "EAP_REQUIRES_SINGLE_CAMERA")

    def test_eap_database_camera_is_synced_to_seed_pinhole_camera(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "database.db"
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "create table cameras(camera_id integer primary key, model integer, width integer, height integer, params blob, prior_focal_length integer)"
                )
                connection.execute("create table images(image_id integer primary key, name text, camera_id integer)")
                simple_radial = struct.pack("<dddd", 900.0, 320.0, 240.0, 0.0)
                connection.execute("insert into cameras values (?, ?, ?, ?, ?, ?)", (1, 2, 640, 480, simple_radial, 0))
                connection.execute("insert into images values (?, ?, ?)", (1, "frame.jpg", 1))
                connection.execute("insert into images values (?, ?, ?)", (2, "frame_aug.jpg", 1))
                connection.commit()
            finally:
                connection.close()

            camera = SimpleNamespace(camera_id=7, model="PINHOLE", width=640, height=480, params=[500.0, 510.0, 320.0, 240.0])
            _sync_eap_database_single_camera(database, _single_undistorted_camera(SimpleNamespace(cameras={7: camera})))
            connection = sqlite3.connect(database)
            try:
                cameras = connection.execute("select camera_id, model, width, height, params, prior_focal_length from cameras").fetchall()
                image_camera_ids = connection.execute("select camera_id from images order by image_id").fetchall()
            finally:
                connection.close()

        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0][:4], (7, 1, 640, 480))
        self.assertEqual(struct.unpack("<dddd", cameras[0][4]), (500.0, 510.0, 320.0, 240.0))
        self.assertEqual(cameras[0][5], 1)
        self.assertEqual(image_camera_ids, [(7,), (7,)])

    def test_point_multiplier_guard_fails_before_installing_large_eap_output(self) -> None:
        self.assertEqual(_check_point_multiplier(10, 100, 10), 10.0)

        with self.assertRaises(FineFailure) as raised:
            _check_point_multiplier(10, 101, 10)

        self.assertEqual(raised.exception.code, "EAP_POINT_MULTIPLIER_EXCEEDED")

    def test_install_eap_points_does_not_overwrite_original_sparse_points(self) -> None:
        reconstruction = SimpleNamespace(
            points3D={
                1: SimpleNamespace(xyz=[1.0, 2.0, 3.0], color=[10, 20, 30]),
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_sparse = root / "output"
            sparse = root / "sparse" / "0"
            output_sparse.mkdir()
            sparse.mkdir(parents=True)
            (output_sparse / "points3D.bin").write_bytes(b"enhanced")
            (sparse / "points3D.bin").write_bytes(b"original")

            points_path, ply_path = _install_eap_points(output_sparse, sparse, reconstruction)
            original_bytes = (sparse / "points3D.bin").read_bytes()
            eap_bytes = (sparse / "points3D_eap.bin").read_bytes()
            ply_text = ply_path.read_text(encoding="ascii")

        self.assertEqual(points_path.name, "points3D_eap.bin")
        self.assertEqual(original_bytes, b"original")
        self.assertEqual(eap_bytes, b"enhanced")
        self.assertIn("element vertex 1", ply_text)
        self.assertIn("1 2 3 10 20 30", ply_text)


if __name__ == "__main__":
    unittest.main()
