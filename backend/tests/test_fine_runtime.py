from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fine.colmap_defaults import FINE_PIPELINE_NAME  # noqa: E402
from app.fine.types import FineContext, FineFailure  # noqa: E402


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class FineRuntimeTests(unittest.TestCase):
    def test_fine_runtime_registers_colmap_only_runtime(self) -> None:
        algorithms_source = (BACKEND_ROOT / "app" / "algorithms.py").read_text(encoding="utf-8")
        fine_status_block = algorithms_source.split("def fine_runtime_status", 1)[1].split("def ", 1)[0]

        self.assertIn("pycolmap", fine_status_block)
        self.assertIn("colmap_cli", fine_status_block)
        self.assertIn("ffmpeg", fine_status_block)
        self.assertNotIn("diff_gaussian", fine_status_block)
        self.assertNotIn("simple_knn", fine_status_block)
        self.assertNotIn("fused_ssim", fine_status_block)

    def test_fine_code_keeps_colmap_boundary(self) -> None:
        fine_root = BACKEND_ROOT / "app" / "fine"

        self.assertTrue((fine_root / "runner.py").exists())
        self.assertTrue((fine_root / "colmap_cli.py").exists())
        self.assertTrue((fine_root / "colmap_defaults.py").exists())
        self.assertFalse((fine_root / "official_fastgs_big_trainer.py").exists())
        self.assertFalse((fine_root / "deblur_schedule.py").exists())
        self.assertFalse((fine_root / "vendor" / "fastgs").exists())

    def test_normalize_fine_pipeline_uses_colmap_name(self) -> None:
        from app.fine.runner import normalize_fine_pipeline

        self.assertEqual(normalize_fine_pipeline(None), FINE_PIPELINE_NAME)
        self.assertEqual(normalize_fine_pipeline(FINE_PIPELINE_NAME), FINE_PIPELINE_NAME)
        self.assertEqual(normalize_fine_pipeline("mobilegs_lmrs"), "mobilegs_lmrs")

    def test_video_fine_pipeline_raises_unsupported(self) -> None:
        from app.fine.runner import run_fine_pipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = FineContext(
                task_id="task",
                project_id="project",
                pipeline="video_artdeco_speed3r",
                input_dir=root,
                input_video=root / "clip.mp4",
                work_dir=root / "work",
                model_cache_dir=root / "model-cache",
                final_ply=root / "work" / "final.ply",
                final_spz=root / "work" / "final_web.spz",
                viewer_meta_json=root / "work" / "final_viewer_meta.json",
                metrics_json=root / "work" / "metrics.json",
                lod_rad=None,
                source_version=7,
                options={},
            )

            with self.assertRaises(FineFailure) as raised:
                run_fine_pipeline(ctx)

        self.assertEqual(raised.exception.code, "UNSUPPORTED_FINE_PIPELINE")

    def test_sfm_defaults_to_colmap_cli(self) -> None:
        from app.fine.preprocess import SceneBuildResult
        from app.fine.runner import build_scene

        expected = SceneBuildResult(Path("scene"), "colmap_cli", 8, 8, 100, {"sfm_backend": "colmap_cli"})
        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={}, progress=None)
        with tempfile.TemporaryDirectory() as tmp, patch("app.fine.runner.build_colmap_cli_scene", return_value=expected) as colmap_cli:
            result = build_scene(
                ctx,
                Path(tmp),
                Path(tmp) / "scene",
                8192,
                1600,
                8,
                min_sparse_points=0,
            )

        self.assertEqual(result.backend, "colmap_cli")
        colmap_cli.assert_called_once()

    def test_build_scene_rejects_removed_sfm_backend(self) -> None:
        from app.fine.runner import build_scene

        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={"fine_sfm_backend": "amb3r"}, progress=None)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FineFailure) as raised:
                build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8)

        self.assertEqual(raised.exception.code, "UNSUPPORTED_FINE_SFM_BACKEND")

    def test_viewer_meta_accepts_sparse_point_ply(self) -> None:
        try:
            from app.fine.viewer_meta import read_ply_xyz_bounds, write_final_viewer_meta_json
        except Exception as exc:
            raise unittest.SkipTest(f"viewer meta dependencies unavailable: {exc}") from exc

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 2\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ply = root / "points.ply"
            meta = root / "meta.json"
            ply.write_bytes(header + struct.pack("<fffBBB", 0.0, 0.0, 0.0, 255, 0, 0) + struct.pack("<fffBBB", 1.0, 1.0, 1.0, 0, 255, 0))

            bounds = read_ply_xyz_bounds(ply)
            payload = write_final_viewer_meta_json(meta, final_ply=ply, scene_dir=root / "scene")

        self.assertEqual(bounds["vertex_count"], 2)
        self.assertEqual(payload["asset_type"], "fine_colmap_sparse_pointcloud")


if __name__ == "__main__":
    unittest.main()
