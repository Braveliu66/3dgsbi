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
WORKSPACE_ROOT = BACKEND_ROOT.parent


class FineRuntimeTests(unittest.TestCase):
    def test_fine_runtime_registers_dash_deblur_group_runtime(self) -> None:
        algorithms_source = (BACKEND_ROOT / "app" / "algorithms.py").read_text(encoding="utf-8")
        fine_status_block = algorithms_source.split("def fine_runtime_status", 1)[1].split("def ", 1)[0]

        self.assertIn("pycolmap", fine_status_block)
        self.assertIn("colmap_cli", fine_status_block)
        self.assertIn("dash_deblur_group", fine_status_block)
        self.assertIn("dependencies", fine_status_block)
        self.assertIn("ffmpeg", fine_status_block)

    def test_algorithm_dependency_matrix_covers_active_algorithms(self) -> None:
        algorithms_source = (BACKEND_ROOT / "app" / "algorithms.py").read_text(encoding="utf-8")

        self.assertIn('"LiteVGGT"', algorithms_source)
        self.assertIn('"transformer_engine.pytorch"', algorithms_source)
        self.assertIn('"Spark SPZ"', algorithms_source)
        self.assertIn('"node"', algorithms_source)
        self.assertIn('"DashDeblurGroupGS Fine"', algorithms_source)
        self.assertIn('"diff_gaussian_rasterization"', algorithms_source)
        self.assertIn('"simple_knn._C"', algorithms_source)

    def test_worker_requirements_include_embedded_trainer_dependencies(self) -> None:
        requirements_source = (WORKSPACE_ROOT / "worker" / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("configargparse==1.7.1", requirements_source)

    def test_worker_dockerfile_bakes_dash_deblur_group_runtime(self) -> None:
        dockerfile_source = (WORKSPACE_ROOT / "worker" / "Dockerfile").read_text(encoding="utf-8")
        compose_source = (WORKSPACE_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("ARG COLMAP_REPO_URL", dockerfile_source)
        self.assertIn("ARG COLMAP_REPO_COMMIT", dockerfile_source)
        self.assertIn("libopenexr-dev", dockerfile_source)
        self.assertIn("libopencv-dev", dockerfile_source)
        self.assertIn("libglm-dev", dockerfile_source)
        self.assertIn("ccache", dockerfile_source)
        self.assertNotIn("libimath-dev", dockerfile_source)
        self.assertIn("cmake --build", dockerfile_source)
        self.assertIn("global_mapper", dockerfile_source)
        self.assertIn("hierarchical_mapper", dockerfile_source)
        self.assertIn("model_clusterer", dockerfile_source)
        self.assertIn("model_splitter", dockerfile_source)
        self.assertNotIn("ARG DASH_DEBLUR_GROUP_REPO_URL", dockerfile_source)
        self.assertNotIn("YouyuChen0207/DashGaussian", dockerfile_source)
        self.assertIn("COPY worker/trainer/dash_deblur_group_gs /opt/dash_deblur_group_gs", dockerfile_source)
        self.assertIn("ENV DASH_DEBLUR_GROUP_REPO=/opt/dash_deblur_group_gs", dockerfile_source)
        self.assertIn("submodules/diff-gaussian-rasterization", dockerfile_source)
        self.assertIn("submodules/simple-knn", dockerfile_source)
        self.assertIn("submodules/fused-ssim", dockerfile_source)
        self.assertIn("third_party/glm/glm/glm.hpp", dockerfile_source)
        self.assertIn("extension wheel cache hit", dockerfile_source)
        self.assertIn("extension wheel cache miss", dockerfile_source)
        self.assertNotIn("rm -rf \"$source_dir/build\"", dockerfile_source)
        self.assertIn("diff_gaussian_rasterization", dockerfile_source)
        self.assertIn("simple_knn._C", dockerfile_source)
        self.assertIn("COLMAP_REPO_URL: ${COLMAP_REPO_URL:-https://github.com/colmap/colmap.git}", compose_source)
        self.assertNotIn("DASH_DEBLUR_GROUP_REPO_URL", compose_source)
        self.assertIn("3dgsbi-worker:local", compose_source)
        self.assertIn("BUILDKIT_INLINE_CACHE", compose_source)
        self.assertIn("DASH_DEBLUR_GROUP_REPO: /opt/dash_deblur_group_gs", compose_source)
        self.assertGreaterEqual(compose_source.count("./backend/app:/app/app"), 3)
        self.assertIn('"--reload"', compose_source)
        self.assertIn("./frontend:/app", compose_source)
        self.assertIn('"next", "dev"', compose_source)
        self.assertEqual(compose_source.count("./worker/trainer/dash_deblur_group_gs:/opt/dash_deblur_group_gs"), 3)

    def test_fine_code_keeps_colmap_boundary_and_training_wrapper(self) -> None:
        fine_root = BACKEND_ROOT / "app" / "fine"

        self.assertTrue((fine_root / "runner.py").exists())
        self.assertTrue((fine_root / "colmap_cli.py").exists())
        self.assertTrue((fine_root / "colmap_defaults.py").exists())
        self.assertTrue((fine_root / "dash_deblur_group.py").exists())
        self.assertFalse((fine_root / "official_fastgs_big_trainer.py").exists())
        self.assertFalse((fine_root / "deblur_schedule.py").exists())
        self.assertFalse((fine_root / "vendor" / "fastgs").exists())
        trainer_root = WORKSPACE_ROOT / "worker" / "trainer" / "dash_deblur_group_gs"
        self.assertTrue((trainer_root / "train.py").exists())
        self.assertTrue((trainer_root / "utils" / "schedule_utils.py").exists())
        self.assertTrue((trainer_root / "gaussians_grouping" / "__init__.py").exists())
        self.assertTrue((trainer_root / "gaussians_grouping" / "grouping_method.py").exists())
        self.assertTrue((trainer_root / "submodules" / "diff-gaussian-rasterization").exists())
        self.assertTrue((trainer_root / "submodules" / "simple-knn").exists())

        schedule_source = (trainer_root / "utils" / "schedule_utils.py").read_text(encoding="utf-8")
        gaussian_model_source = (trainer_root / "scene" / "gaussian_model.py").read_text(encoding="utf-8")
        grouping_source = (trainer_root / "gaussians_grouping" / "__init__.py").read_text(encoding="utf-8")
        grouping_method_source = (trainer_root / "gaussians_grouping" / "grouping_method.py").read_text(encoding="utf-8")
        self.assertIn("torch.fft.fft2", schedule_source)
        self.assertIn("get_densify_rate", schedule_source)
        self.assertIn("get_delayed_expon_lr_func", schedule_source)
        self.assertIn("return decay(step - decay_from_iter)", schedule_source)
        self.assertNotIn("birth_iter", gaussian_model_source)
        self.assertNotIn("protect_new_points_iters", gaussian_model_source)
        self.assertIn("prune mask has", gaussian_model_source)
        self.assertIn("gaussian_model.prune_points(mask_cache)", grouping_source)
        self.assertIn('"Opacity-weighted"', grouping_method_source)
        self.assertIn("torch.multinomial", grouping_method_source)

    def test_normalize_fine_pipeline_uses_dash_name_and_colmap_alias(self) -> None:
        from app.fine.runner import normalize_fine_pipeline

        self.assertEqual(normalize_fine_pipeline(None), FINE_PIPELINE_NAME)
        self.assertEqual(normalize_fine_pipeline(FINE_PIPELINE_NAME), FINE_PIPELINE_NAME)
        self.assertEqual(normalize_fine_pipeline("colmap_sparse"), FINE_PIPELINE_NAME)
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

        expected = SceneBuildResult(Path("scene"), "colmap_cli", 8, 8, 2816, {"sfm_backend": "colmap_cli"})
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
        self.assertEqual(result.point_count, 2816)
        self.assertTrue(result.metrics["sfm_sparse_points_below_target"])
        colmap_cli.assert_called_once()

    def test_sparse_point_gate_only_fails_when_explicitly_requested(self) -> None:
        from app.fine.preprocess import SceneBuildResult
        from app.fine.runner import build_scene

        expected = SceneBuildResult(Path("scene"), "colmap_cli", 8, 8, 2816, {"sfm_backend": "colmap_cli"})
        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={}, progress=None)
        with tempfile.TemporaryDirectory() as tmp, patch("app.fine.runner.build_colmap_cli_scene", return_value=expected):
            result = build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8, min_sparse_points=0)
        self.assertEqual(result.point_count, 2816)

        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={}, progress=None)
        with tempfile.TemporaryDirectory() as tmp, patch("app.fine.runner.build_colmap_cli_scene", return_value=expected):
            with self.assertRaises(FineFailure) as raised:
                build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8, min_sparse_points=15000)
        self.assertEqual(raised.exception.code, "SFM_SPARSE_POINTS_TOO_LOW")

    def test_build_scene_rejects_removed_sfm_backend(self) -> None:
        from app.fine.runner import build_scene

        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={"fine_sfm_backend": "amb3r"}, progress=None)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FineFailure) as raised:
                build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8)

        self.assertEqual(raised.exception.code, "UNSUPPORTED_FINE_SFM_BACKEND")

    def test_training_runtime_preflight_uses_embedded_trainer(self) -> None:
        from app.fine.runner import assert_training_runtime_ready

        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {}, clear=True):
            assert_training_runtime_ready({}, Path(tmp) / "repo-cache")

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
