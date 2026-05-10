from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fine.types import FineFailure  # noqa: E402


def import_fine_runtime():
    try:
        from app.fine.amb3r_sfm import amb3r_weight_path
        from app.fine.preprocess import BlurScore, SceneBuildResult, summarize_blur_scores
        from app.fine.runner import build_scene, deblur_mlp_enabled_by_default, normalize_fine_pipeline
    except Exception as exc:
        raise unittest.SkipTest(f"fine runtime dependencies unavailable: {exc}") from exc
    return (
        amb3r_weight_path,
        BlurScore,
        SceneBuildResult,
        summarize_blur_scores,
        build_scene,
        deblur_mlp_enabled_by_default,
        normalize_fine_pipeline,
    )


class FineRuntimeTests(unittest.TestCase):
    def test_fine_runtime_does_not_require_romatch(self) -> None:
        algorithms_source = (BACKEND_ROOT / "app" / "algorithms.py").read_text(encoding="utf-8")
        fine_status_block = algorithms_source.split("def fine_runtime_status", 1)[1].split("def ", 1)[0]

        self.assertNotIn("romatch", fine_status_block)
        self.assertIn("amb3r", fine_status_block)
        self.assertNotIn("transformer_engine", fine_status_block)

    def test_trainer_uses_local_runtime_not_lmrs_repo_training(self) -> None:
        trainer_source = (BACKEND_ROOT / "app" / "fine" / "mobilegs_trainer.py").read_text(encoding="utf-8")

        self.assertIn("local_3dgs_runtime", trainer_source)
        self.assertIn("FastGS_local_multiview_score", trainer_source)
        self.assertIn("LM-RS_local_matrix_free", trainer_source)
        self.assertIn("lmrs_phase_iterations", trainer_source)
        self.assertIn("policy.cuda_metric_calls > 0", trainer_source)
        self.assertIn("fine_lmrs_enabled", trainer_source)
        self.assertIn("LM-RS temporarily isolated due to unstable local backend", trainer_source)
        self.assertIn("fine_lmrs_lambda_dssim", trainer_source)
        self.assertIn("Scene(dataset, gaussians, shuffle=True)", trainer_source)
        self.assertNotIn("Scene(dataset, gaussians, opt", trainer_source)
        self.assertNotIn("resolve_lmrs_root", trainer_source)
        self.assertNotIn("prepend_sys_path", trainer_source)
        self.assertNotIn("/opt/lm-rs", trainer_source)
        self.assertNotIn("gauss_newton_step", trainer_source)

    def test_lmrs_is_isolated_by_default(self) -> None:
        runner_source = (BACKEND_ROOT / "app" / "fine" / "runner.py").read_text(encoding="utf-8")
        trainer_source = (BACKEND_ROOT / "app" / "fine" / "mobilegs_trainer.py").read_text(encoding="utf-8")

        self.assertIn("lm_default = iterations", runner_source)
        self.assertNotIn("min(15_000, iterations)", runner_source)
        self.assertIn('read_bool((options or {}).get("fine_lmrs_enabled"), False)', trainer_source)
        self.assertIn('"active": False', trainer_source)
        self.assertIn("LM-RS temporarily isolated due to unstable local backend", trainer_source)

    def test_compact_box_patch_is_build_input(self) -> None:
        patch_source = (BACKEND_ROOT.parent / "worker" / "patches" / "lmrs-fastgs-compact-box.patch").read_text(encoding="utf-8")
        metric_patch = (BACKEND_ROOT.parent / "worker" / "patches" / "fastgs-cuda-metric-accumulation.patch").read_text(encoding="utf-8")

        self.assertIn("duplicateToTilesTouched", patch_source)
        self.assertIn("MOBILEGS_COMPACT_BOX", patch_source)
        self.assertIn("fastgs_accumulate_metrics", metric_patch)
        self.assertIn("MOBILEGS_FASTGS_METRIC", metric_patch)
        dockerfile = (BACKEND_ROOT.parent / "worker" / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("get_JTv", dockerfile)
        self.assertIn("fastgs_accumulate_metrics", dockerfile)
        self.assertIn("fastgs-cuda-metric-accumulation.patch", dockerfile)
        self.assertIn("retry_git", dockerfile)
        self.assertIn("submodule update --init --recursive", dockerfile)
        self.assertIn("LMRS_GLM", dockerfile)
        self.assertIn("glm/glm.hpp", dockerfile)
        self.assertIn("VENDORED_GLM", dockerfile)
        self.assertNotIn("source.trainer", dockerfile)
        self.assertIn("fastgs-cuda-metric-accumulation.patch", (BACKEND_ROOT.parent / "scripts" / "bootstrap-repos.sh").read_text(encoding="utf-8"))
        self.assertIn("fastgs-cuda-metric-accumulation.patch", (BACKEND_ROOT.parent / "scripts" / "bootstrap-repos.ps1").read_text(encoding="utf-8"))

    def test_fine_code_is_split_by_integration_boundary(self) -> None:
        fine_root = BACKEND_ROOT / "app" / "fine"

        self.assertTrue((fine_root / "fastgs_policy.py").exists())
        self.assertTrue((fine_root / "local_3dgs" / "runtime.py").exists())
        self.assertTrue((fine_root / "local_3dgs" / "scene_quality.py").exists())
        self.assertTrue((fine_root / "local_3dgs" / "sparse_compensation.py").exists())
        self.assertTrue((fine_root / "local_3dgs" / "cg_state.py").exists())
        self.assertTrue((fine_root / "local_3dgs" / "cg_solver.py").exists())
        self.assertTrue((fine_root / "local_3dgs" / "cg_optimizer.py").exists())
        self.assertTrue((fine_root / "local_3dgs" / "lmrs_step.py").exists())
        self.assertTrue((fine_root / "local_3dgs" / "render.py").exists())
        self.assertTrue((fine_root / "lmrs_runtime.py").exists())
        self.assertTrue((fine_root / "option_utils.py").exists())
        self.assertTrue((fine_root / "amb3r_sfm.py").exists())
        self.assertTrue((fine_root / "amb3r_runtime").exists())

    def test_compose_uses_one_worker_image(self) -> None:
        compose_source = (BACKEND_ROOT.parent / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("image: 3dgsbi-worker:local", compose_source)
        self.assertIn("target: worker", compose_source)
        self.assertNotIn("target: worker-preview", compose_source)
        self.assertNotIn("target: worker-fine", compose_source)

    def test_legacy_fine_pipeline_aliases_to_mobilegs_lmrs(self) -> None:
        *_, normalize_fine_pipeline = import_fine_runtime()

        self.assertEqual(normalize_fine_pipeline("fused_quality_3dgs"), "mobilegs_lmrs")
        self.assertEqual(normalize_fine_pipeline(None), "mobilegs_lmrs")

    def test_blur_summary_reports_kept_images(self) -> None:
        _, BlurScore, _, summarize_blur_scores, *_ = import_fine_runtime()
        scores = [
            BlurScore(path=Path(f"{index}.jpg"), laplacian=140.0, gradient=50.0, fft_high_ratio=0.1)
            for index in range(10)
        ]

        summary = summarize_blur_scores(scores, reject_ratio=0.2)

        self.assertEqual(summary.rejected_images, 2)
        self.assertEqual(summary.kept_images, 8)

    def test_any_blurry_training_image_triggers_deblur(self) -> None:
        _, BlurScore, _, summarize_blur_scores, _, deblur_mlp_enabled_by_default, _ = import_fine_runtime()
        scores = [
            BlurScore(path=Path(f"sharp_{index}.jpg"), laplacian=180.0, gradient=55.0, fft_high_ratio=0.12)
            for index in range(9)
        ]
        scores.append(BlurScore(path=Path("blur.jpg"), laplacian=50.0, gradient=45.0, fft_high_ratio=0.04))

        summary = summarize_blur_scores(scores, reject_ratio=0.0)

        self.assertEqual(summary.training_blur_frames, 1)
        self.assertEqual(summary.mode, "mixed")
        self.assertTrue(deblur_mlp_enabled_by_default(summary.mode, {}))
        self.assertEqual(summary.deblur_trigger_reason, "training_blur:mixed")

    def test_rejected_blurry_image_does_not_trigger_deblur(self) -> None:
        _, BlurScore, _, summarize_blur_scores, _, deblur_mlp_enabled_by_default, _ = import_fine_runtime()
        scores = [
            BlurScore(path=Path(f"sharp_{index}.jpg"), laplacian=180.0, gradient=55.0, fft_high_ratio=0.12)
            for index in range(9)
        ]
        scores.append(BlurScore(path=Path("blur.jpg"), laplacian=10.0, gradient=10.0, fft_high_ratio=0.01))

        summary = summarize_blur_scores(scores, reject_ratio=0.1)

        self.assertEqual(summary.training_blur_frames, 0)
        self.assertEqual(summary.rejected_blur_frames, 1)
        self.assertEqual(summary.mode, "sharp")
        self.assertFalse(deblur_mlp_enabled_by_default(summary.mode, {}))

    def test_sfm_defaults_to_amb3r(self) -> None:
        _, _, SceneBuildResult, _, build_scene, *_ = import_fine_runtime()
        expected = SceneBuildResult(Path("scene"), "amb3r_sfm_colmap_no_exif", 8, 8, 100, {"sfm_backend": "amb3r_sfm_colmap_no_exif"})
        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={}, progress=None)
        with tempfile.TemporaryDirectory() as tmp, patch("app.fine.runner.build_amb3r_colmap_scene", return_value=expected) as amb3r, patch(
            "app.fine.runner.build_pycolmap_scene"
        ) as pycolmap:
            result = build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8)

        self.assertEqual(result.backend, "amb3r_sfm_colmap_no_exif")
        amb3r.assert_called_once()
        pycolmap.assert_not_called()

    def test_sfm_does_not_auto_fallback_to_pycolmap(self) -> None:
        _, _, _, _, build_scene, *_ = import_fine_runtime()
        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={}, progress=None)
        with tempfile.TemporaryDirectory() as tmp, patch(
            "app.fine.runner.build_amb3r_colmap_scene",
            side_effect=FineFailure("AMB3R_WEIGHT_MISSING", "missing"),
        ), patch("app.fine.runner.build_pycolmap_scene") as pycolmap:
            with self.assertRaises(FineFailure) as raised:
                build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8)

        self.assertEqual(raised.exception.code, "AMB3R_WEIGHT_MISSING")
        pycolmap.assert_not_called()

    def test_explicit_pycolmap_backend_is_diagnostic_only(self) -> None:
        _, _, SceneBuildResult, _, build_scene, *_ = import_fine_runtime()
        expected = SceneBuildResult(Path("scene"), "pycolmap", 3, 3, 42, {"sfm_backend": "pycolmap"})
        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={"fine_sfm_backend": "pycolmap"}, progress=None)
        with tempfile.TemporaryDirectory() as tmp, patch("app.fine.runner.build_amb3r_colmap_scene") as amb3r, patch(
            "app.fine.runner.build_pycolmap_scene", return_value=expected
        ) as pycolmap:
            result = build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8)

        self.assertEqual(result.backend, "pycolmap")
        self.assertEqual(result.metrics["sfm_backend_requested"], "pycolmap_diagnostic")
        amb3r.assert_not_called()
        pycolmap.assert_called_once()

    def test_litevggt_fine_backend_alias_maps_to_amb3r(self) -> None:
        _, _, SceneBuildResult, _, build_scene, *_ = import_fine_runtime()
        expected = SceneBuildResult(Path("scene"), "amb3r_sfm_colmap_no_exif", 3, 3, 42, {"sfm_backend": "amb3r_sfm_colmap_no_exif"})
        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={"fine_sfm_backend": "litevggt"}, progress=None)
        with tempfile.TemporaryDirectory() as tmp, patch("app.fine.runner.build_amb3r_colmap_scene", return_value=expected) as amb3r:
            result = build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8)

        self.assertEqual(result.backend, "amb3r_sfm_colmap_no_exif")
        self.assertEqual(result.metrics["sfm_backend_requested_alias"], "litevggt_deprecated_maps_to_amb3r")
        amb3r.assert_called_once()
        runner_source = (BACKEND_ROOT / "app" / "fine" / "runner.py").read_text(encoding="utf-8")
        self.assertIn("fine_amb3r_keep_ratio", runner_source)
        self.assertIn("0.12", runner_source)

    def test_amb3r_weight_directory_and_auto_download_registration(self) -> None:
        self.assertTrue((BACKEND_ROOT.parent / "model-cache" / "amb3r").exists())
        fine_worker_source = (BACKEND_ROOT / "app" / "fine_worker.py").read_text(encoding="utf-8")
        self.assertIn("ensure_amb3r_weight", fine_worker_source)
        self.assertIn("download_model_weights", fine_worker_source)
        self.assertIn("weights_for_pipeline", fine_worker_source)
        try:
            from app.fine.amb3r_sfm import amb3r_weight_path, build_amb3r_colmap_scene
        except Exception as exc:
            raise unittest.SkipTest(f"AMB3R lightweight import unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as tmp:
            weight = amb3r_weight_path(Path(tmp))
            self.assertTrue(weight.parent.exists())
            self.assertEqual(weight.name, "amb3r.pt")
            with self.assertRaises(FineFailure) as raised:
                build_amb3r_colmap_scene(
                    Path(tmp),
                    Path(tmp) / "scene",
                    checkpoint_path=weight,
                    keep_ratio=0.12,
                    max_points=10000,
                    progress=lambda *_: None,
                )

        self.assertEqual(raised.exception.code, "AMB3R_WEIGHT_MISSING")

    def test_amb3r_colmap_writer_creates_sparse_outputs(self) -> None:
        try:
            import numpy as np
        except Exception as exc:
            raise unittest.SkipTest(f"numpy unavailable: {exc}") from exc
        from app.fine.amb3r_sfm import ProcessedAmb3rImage, write_colmap_model, write_gaussian_splatting_ply

        with tempfile.TemporaryDirectory() as tmp:
            sparse_dir = Path(tmp) / "sparse" / "0"
            sparse_dir.mkdir(parents=True)
            images = [
                ProcessedAmb3rImage(Path(f"{index}.jpg"), 640, 480, 518, 392, 0, 0, 640, 480, np.zeros((392, 518, 3), dtype=np.float32))
                for index in range(3)
            ]
            poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
            poses[:, 0, 3] = np.arange(3, dtype=np.float32)
            intrinsics = np.tile(np.eye(3, dtype=np.float32), (3, 1, 1))
            intrinsics[:, 0, 0] = 500.0
            intrinsics[:, 1, 1] = 510.0
            intrinsics[:, 0, 2] = 259.0
            intrinsics[:, 1, 2] = 196.0
            points = np.array([[0.0, 0.0, 1.0], [0.1, 0.2, 1.2]], dtype=np.float32)
            colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

            write_gaussian_splatting_ply(sparse_dir / "points3D.ply", points, colors)
            write_colmap_model(sparse_dir, images, [0, 1, 2], poses, intrinsics, points, colors)

            self.assertTrue((sparse_dir / "cameras.bin").exists())
            self.assertTrue((sparse_dir / "images.bin").exists())
            self.assertTrue((sparse_dir / "points3D.bin").exists())
            self.assertTrue((sparse_dir / "points3D.ply").stat().st_size > 0)

    def test_amb3r_auto_resolution_preserves_aspect_and_patch_multiple(self) -> None:
        try:
            from PIL import Image
            from app.fine.amb3r_sfm import AMB3R_PATCH_SIZE, prepare_amb3r_images, resolve_amb3r_resolution
        except Exception as exc:
            raise unittest.SkipTest(f"AMB3R helpers unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = []
            for index in range(4):
                path = root / f"{index}.jpg"
                Image.new("RGB", (1600, 1066), color=(index, index, index)).save(path)
                files.append(path)

            plan = resolve_amb3r_resolution(files, {"fine_amb3r_token_budget": 4000})
            images_dir = root / "images"
            images_dir.mkdir()
            processed = prepare_amb3r_images(files, images_dir, width=plan.selected_width, height=plan.selected_height, target_aspect=plan.target_aspect)

        self.assertEqual(plan.selected_width % AMB3R_PATCH_SIZE, 0)
        self.assertEqual(plan.selected_height % AMB3R_PATCH_SIZE, 0)
        self.assertLess(abs((plan.selected_width / plan.selected_height) - (1600 / 1066)), 0.05)
        self.assertEqual(processed[0].processed_width, plan.selected_width)
        self.assertEqual(processed[0].processed_height, plan.selected_height)

    def test_amb3r_pose_alignment_accepts_bfloat16_numpy_boundary(self) -> None:
        try:
            import torch
            from app.fine.amb3r_runtime.amb3r.tools.pose_align import average_transforms_with_weights
        except Exception as exc:
            raise unittest.SkipTest(f"AMB3R pose alignment dependencies unavailable: {exc}") from exc

        transforms = torch.eye(4, dtype=torch.bfloat16).repeat(3, 1, 1)
        transforms[:, 0, 3] = torch.tensor([0.0, 1.0, 2.0], dtype=torch.bfloat16)
        weights = torch.ones(3, dtype=torch.bfloat16)

        averaged = average_transforms_with_weights(transforms, weights)

        self.assertEqual(averaged.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(averaged.float()).all())

    def test_amb3r_to_numpy_casts_bfloat16_to_float32(self) -> None:
        try:
            import numpy as np
            import torch
            from app.fine.amb3r_sfm import to_numpy
        except Exception as exc:
            raise unittest.SkipTest(f"AMB3R numpy conversion dependencies unavailable: {exc}") from exc

        value = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)

        converted = to_numpy(value)

        self.assertEqual(converted.dtype, np.float32)
        self.assertEqual(converted.shape, (1, 2))

    def test_amb3r_window_planning_uses_overlap_and_tail_window(self) -> None:
        try:
            from app.fine.amb3r_sfm import plan_amb3r_windows
        except Exception as exc:
            raise unittest.SkipTest(f"AMB3R window helpers unavailable: {exc}") from exc

        windows = plan_amb3r_windows(
            150,
            {
                "fine_amb3r_windowed": "true",
                "fine_amb3r_window_size": 64,
                "fine_amb3r_window_overlap": 12,
            },
        )

        self.assertEqual([(window.start, window.end) for window in windows], [(0, 64), (52, 116), (104, 150)])

    def test_amb3r_window_planning_keeps_small_sets_full_scene(self) -> None:
        try:
            from app.fine.amb3r_sfm import plan_amb3r_windows
        except Exception as exc:
            raise unittest.SkipTest(f"AMB3R window helpers unavailable: {exc}") from exc

        windows = plan_amb3r_windows(94, {})

        self.assertEqual([(window.start, window.end) for window in windows], [(0, 94)])

    def test_amb3r_similarity_transform_aligns_window_centers(self) -> None:
        try:
            import numpy as np
            from app.fine.amb3r_sfm import apply_similarity_to_points, estimate_similarity_transform
        except Exception as exc:
            raise unittest.SkipTest(f"AMB3R similarity helpers unavailable: {exc}") from exc

        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
        target = source * 2.0 + np.array([3.0, -1.0, 0.5], dtype=np.float32)

        transform = estimate_similarity_transform(source, target)
        aligned = apply_similarity_to_points(source, transform)

        np.testing.assert_allclose(aligned, target, rtol=1e-5, atol=1e-5)

    def test_amb3r_materialize_result_clones_inference_tensors(self) -> None:
        try:
            import torch
            from app.fine.amb3r_runtime.sfm.pipeline import materialize_result
        except Exception as exc:
            raise unittest.SkipTest(f"AMB3R materialize helper unavailable: {exc}") from exc

        with torch.inference_mode():
            result = {"pose": torch.eye(4)}

        materialized = materialize_result(result)
        materialized["pose"][0, 3] = 1.0

        self.assertEqual(float(materialized["pose"][0, 3]), 1.0)

    def test_preview_litevggt_and_fine_amb3r_imports_are_isolated(self) -> None:
        sys.modules.pop("vggt", None)

        try:
            import app.preview.vendor.litevggt_runtime  # noqa: F401
            import app.fine.amb3r_sfm  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest(f"preview/fine lightweight imports unavailable: {exc}") from exc

        self.assertNotIn("vggt", sys.modules)

    def test_deblur_mlp_replaces_scaling_heuristic(self) -> None:
        trainer_source = (BACKEND_ROOT / "app" / "fine" / "mobilegs_trainer.py").read_text(encoding="utf-8")
        deblur_source = (BACKEND_ROOT / "app" / "fine" / "deblur_mlp.py").read_text(encoding="utf-8")

        self.assertNotIn("deblur_scaling_modifier", trainer_source)
        self.assertNotIn("fine_deblur_scaling_modifier", trainer_source)
        self.assertIn("GTnet", deblur_source)
        self.assertIn("FourierEmbedding", deblur_source)
        self.assertIn("position_delta", deblur_source)
        self.assertIn("render_with_deblur_mlp", trainer_source)
        self.assertIn("fine_deblur_min_clamp", deblur_source)

    def test_deblur_auto_controls_lmrs_default(self) -> None:
        *_, deblur_mlp_enabled_by_default, _ = import_fine_runtime()

        self.assertTrue(deblur_mlp_enabled_by_default("motion", {}))
        self.assertTrue(deblur_mlp_enabled_by_default("defocus", {}))
        self.assertTrue(deblur_mlp_enabled_by_default("mixed", {}))
        self.assertFalse(deblur_mlp_enabled_by_default("sharp", {}))
        self.assertFalse(deblur_mlp_enabled_by_default("motion", {"fine_deblur_enabled": "false"}))

    def test_worker_runtime_avoids_duplicate_cuda_stacks(self) -> None:
        worker_root = BACKEND_ROOT.parent / "worker"
        requirements = (worker_root / "requirements.txt").read_text(encoding="utf-8").lower()
        dockerfile = (worker_root / "Dockerfile").read_text(encoding="utf-8").lower()
        combined = requirements + "\n" + dockerfile

        self.assertNotIn("transformer-engine", requirements)
        self.assertNotIn("pycolmap", requirements)
        self.assertIn("transformer-engine[pytorch]==2.4.0", dockerfile)
        self.assertIn("pycolmap==3.12.6", dockerfile)
        self.assertIn("spconv-cu118==2.3.8", dockerfile)
        self.assertIn("torch-scatter==2.1.2", dockerfile)
        constraints = (worker_root / "constraints.txt").read_text(encoding="utf-8").lower()
        self.assertIn("torch==2.8.0", constraints)
        self.assertIn("torchvision==0.23.0", constraints)
        self.assertNotIn("flashinfer-python==", requirements)
        self.assertNotIn("flashinfer-cubin", requirements)
        self.assertIn("--index-url https://pypi.org/simple flashinfer-python", dockerfile)
        self.assertNotIn('version("flashinfer-python") ==', dockerfile)
        self.assertIn("timm==0.6.7", requirements)
        self.assertIn("addict==2.4.0", requirements)
        self.assertIn("import app.fine.amb3r_sfm", dockerfile)
        self.assertIn("mkdir -p /model-cache/amb3r", dockerfile)
        self.assertIn("import pycolmap", dockerfile)
        self.assertIn("'einops==0.8.0' 'transformer-engine[pytorch]==2.4.0'", dockerfile)
        self.assertLess(
            dockerfile.index("'einops==0.8.0' 'transformer-engine[pytorch]==2.4.0'"),
            dockerfile.index("import transformer_engine.pytorch as te"),
        )
        self.assertIn("import einops", dockerfile)
        self.assertIn("libcudnn9-dev-cuda-12", dockerfile)
        self.assertIn("cudnn.h", dockerfile)
        self.assertNotIn("nvidia-cudnn-cu12", combined)
        self.assertNotIn("import flashinfer", dockerfile)
        self.assertIn('importlib_metadata.version("flashinfer-python")', dockerfile)
        self.assertNotIn("flashinfer-cubin", dockerfile)
        self.assertNotIn("kaolin", combined)
        self.assertNotIn("open3d", combined)
        self.assertNotIn("xformers", combined)
        self.assertNotIn("pytorch3d", combined)
        self.assertNotIn("gdown", combined)
        self.assertNotIn("pip install torch", dockerfile)
        self.assertNotIn("pip install torchvision", dockerfile)


if __name__ == "__main__":
    unittest.main()
