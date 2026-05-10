from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np

    from app.preview.types import PreviewContext
    from app.preview.utils import VENDOR_ROOT, prepend_sys_path
    from app.preview.vendor.litevggt_runtime import write_litevggt_colmap_scene
except Exception as exc:  # pragma: no cover
    np = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class PreviewLiteVGGTEdgsTests(unittest.TestCase):
    @unittest.skipIf(np is None, f"preview dependencies unavailable: {IMPORT_ERROR}")
    def test_litevggt_colmap_writer_creates_sparse_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "scene"
            images = np.zeros((3, 16, 20, 3), dtype=np.float32)
            images[:, :, :, 1] = 0.5
            w2c = np.tile(np.eye(4, dtype=np.float32)[:3, :], (3, 1, 1))
            w2c[:, 0, 3] = np.arange(3, dtype=np.float32)
            intrinsics = np.tile(np.eye(3, dtype=np.float32), (3, 1, 1))
            intrinsics[:, 0, 0] = 12.0
            intrinsics[:, 1, 1] = 13.0
            intrinsics[:, 0, 2] = 10.0
            intrinsics[:, 1, 2] = 8.0
            points = np.array([[0.0, 0.0, 1.0], [0.2, 0.1, 1.3]], dtype=np.float32)
            colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

            count = write_litevggt_colmap_scene(scene_dir, images, w2c, intrinsics, points, colors)

            sparse_dir = scene_dir / "sparse" / "0"
            self.assertEqual(count, 2)
            self.assertTrue((scene_dir / "images" / "00000000.png").exists())
            self.assertTrue((sparse_dir / "cameras.bin").exists())
            self.assertTrue((sparse_dir / "images.bin").exists())
            self.assertTrue((sparse_dir / "points3D.bin").exists())
            self.assertTrue((sparse_dir / "points3D.ply").stat().st_size > 0)

            with prepend_sys_path(VENDOR_ROOT / "edgs" / "gaussian_splatting" / "utils"):
                from read_write_model import read_model

            cameras, colmap_images, points3d = read_model(str(sparse_dir), ext=".bin")
            self.assertEqual(len(cameras), 3)
            self.assertEqual(len(colmap_images), 3)
            self.assertEqual(len(points3d), 2)

    @unittest.skipIf(np is None, f"preview dependencies unavailable: {IMPORT_ERROR}")
    def test_litevggt_edgs_adapter_uses_litevggt_scene_without_colmap_args(self) -> None:
        from app.preview.adapters import edgs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            for index in range(8):
                (input_dir / f"{index:02d}.jpg").write_bytes(b"placeholder")

            model_cache = root / "models"
            for relative in ("litevggt/te_dict.pt", "roma/roma_indoor.pth", "roma/dinov2_vitl14_pretrain.pth"):
                path = model_cache / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"weight")

            work_dir = root / "work"
            work_dir.mkdir()
            ply_path = work_dir / "edgs.ply"
            ply_path.write_bytes(b"ply\n")
            ctx = PreviewContext(
                task_id="task",
                project_id="project",
                pipeline="litevggt_edgs",
                input_dir=input_dir,
                work_dir=work_dir,
                output_spz=work_dir / "preview.spz",
                model_cache_dir=model_cache,
                source_version=1,
                options={},
                progress=lambda *_: None,
            )

            with patch(
                "app.preview.adapters.edgs.build_litevggt_colmap_scene",
                return_value={"sfm_backend": "litevggt_colmap_no_pycolmap", "pycolmap_used": False, "litevggt_pad_mode": True},
            ) as scene_builder, patch(
                "app.preview.adapters.edgs.run_edgs_preview",
                return_value={"ply_path": ply_path, "pycolmap_used": False},
            ) as edgs_runtime, patch("app.preview.adapters.edgs.convert_ply_to_spz", return_value=9):
                result = edgs.run(ctx)

            scene_builder.assert_called_once()
            edgs_runtime.assert_called_once()
            runtime_kwargs = edgs_runtime.call_args.kwargs
            self.assertNotIn("input_dir", runtime_kwargs)
            self.assertFalse(any(key.startswith("colmap_") for key in runtime_kwargs))
            self.assertFalse(result.metrics["pycolmap_used"])
            self.assertTrue(result.metrics["litevggt_pad_mode"])
            self.assertEqual(result.metrics["sfm_backend"], "litevggt_colmap_no_pycolmap")


if __name__ == "__main__":
    unittest.main()
