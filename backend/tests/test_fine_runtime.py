from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fine.types import FineContext, FineFailure  # noqa: E402


def import_fine_runtime():
    try:
        from app.fine.preprocess import BlurScore, SceneBuildResult, summarize_blur_scores
        from app.fine.runner import build_scene, deblur_mlp_enabled_by_default, normalize_fine_pipeline
    except Exception as exc:
        raise unittest.SkipTest(f"fine runtime dependencies unavailable: {exc}") from exc
    return (
        BlurScore,
        SceneBuildResult,
        summarize_blur_scores,
        build_scene,
        deblur_mlp_enabled_by_default,
        normalize_fine_pipeline,
    )


class FineRuntimeTests(unittest.TestCase):
    def test_fine_runtime_registers_official_fastgs_runtime(self) -> None:
        algorithms_source = (BACKEND_ROOT / "app" / "algorithms.py").read_text(encoding="utf-8")
        fine_status_block = algorithms_source.split("def fine_runtime_status", 1)[1].split("def ", 1)[0]

        self.assertIn("pycolmap", fine_status_block)
        self.assertIn("diff_gaussian_rasterization_fastgs", fine_status_block)
        self.assertIn("simple_knn", fine_status_block)
        self.assertIn("fused_ssim", fine_status_block)
        self.assertNotIn("litevggt_runtime", fine_status_block)
        self.assertNotIn("deblur_mlp", fine_status_block)
        self.assertNotIn('"diff_gaussian_rasterization"', fine_status_block)
        self.assertNotIn('"gsplat"', fine_status_block)
        self.assertNotIn("amb3r", fine_status_block)

    def test_trainer_uses_local_runtime_not_lmrs_repo_training(self) -> None:
        trainer_source = (BACKEND_ROOT / "app" / "fine" / "mobilegs_trainer.py").read_text(encoding="utf-8")
        render_source = (BACKEND_ROOT / "app" / "fine" / "local_3dgs" / "render.py").read_text(encoding="utf-8")

        self.assertIn("local_3dgs_runtime", trainer_source)
        self.assertIn("FastGS_local_multiview_score", trainer_source)
        self.assertIn("diff_gaussian_rasterization_fastgs", render_source)
        self.assertIn("pc.get_features_dc", render_source)
        self.assertIn("pc.get_features_rest", render_source)
        self.assertIn("accum_metric_counts", render_source)
        self.assertIn("metric_render_fn=topology_render", trainer_source)
        self.assertIn("def photometric_render", trainer_source)
        self.assertIn("def topology_render", trainer_source)
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

    def test_image_fine_runner_uses_official_fastgs_big_not_lmrs(self) -> None:
        runner_source = (BACKEND_ROOT / "app" / "fine" / "runner.py").read_text(encoding="utf-8")
        trainer_source = (BACKEND_ROOT / "app" / "fine" / "mobilegs_trainer.py").read_text(encoding="utf-8")

        self.assertIn("train_official_fastgs_big", runner_source)
        self.assertNotIn("train_mobile_3dgs", runner_source)
        self.assertNotIn("fine_lm_start_iter", runner_source)
        self.assertNotIn("lm_default = iterations", runner_source)
        self.assertNotIn("min(15_000, iterations)", runner_source)
        self.assertIn('read_bool((options or {}).get("fine_lmrs_enabled"), False)', trainer_source)
        self.assertIn('"active": False', trainer_source)
        self.assertIn("LM-RS temporarily isolated due to unstable local backend", trainer_source)

    def test_worker_builds_vendored_3dgs_extensions_without_runtime_fastgs_clone(self) -> None:
        dockerfile = (BACKEND_ROOT.parent / "worker" / "Dockerfile").read_text(encoding="utf-8")

        self.assertNotIn("lm-rs", dockerfile.lower())
        self.assertNotIn("LMRS_ROOT", dockerfile)
        self.assertNotIn("lmrs-fastgs-compact-box.patch", dockerfile)
        self.assertNotIn("fastgs-cuda-metric-accumulation.patch", dockerfile)
        self.assertNotIn("FASTGS_REPO_URL", dockerfile)
        self.assertNotIn("github.com/fastgs/FastGS.git", dockerfile)
        self.assertNotIn('ensure_git_checkout "$FASTGS_REPO_URL"', dockerfile)
        self.assertIn("COPY backend/app/fine/vendor/fastgs ./app/fine/vendor/fastgs", dockerfile)
        self.assertIn("ENV FASTGS_VENDOR_ROOT=/app/app/fine/vendor/fastgs", dockerfile)
        self.assertIn("cached_wheel_install diff-gaussian-rasterization-fastgs /app/app/fine/vendor/fastgs/submodules/diff-gaussian-rasterization_fastgs", dockerfile)
        self.assertIn("cached_wheel_install simple-knn /app/app/fine/vendor/fastgs/submodules/simple-knn", dockerfile)
        self.assertIn("cached_wheel_install fused-ssim /app/app/fine/vendor/fastgs/submodules/fused-ssim", dockerfile)
        self.assertNotIn("cached_wheel_install simple-knn-artdeco", dockerfile)
        self.assertIn("retry_pip --force-reinstall --no-deps \"$wheel\"", dockerfile)
        self.assertIn("libc10.so", dockerfile)
        self.assertIn("/etc/ld.so.conf.d/pytorch.conf", dockerfile)
        self.assertIn("retry_git", dockerfile)
        self.assertIn("libeigen3-dev", dockerfile)
        self.assertNotIn("test -f /usr/include/eigen3/Eigen/Sparse", dockerfile)
        self.assertNotIn('"/usr/include/eigen3"', dockerfile)
        self.assertNotIn("submodule update --init --recursive VSLAM/thirdparty/eigen", dockerfile)
        self.assertNotIn("D11.scalar_type()", dockerfile)
        self.assertNotIn("dx.pow(2).sum().sqrt()", dockerfile)
        self.assertIn("cat-file -e", dockerfile)
        self.assertIn("three-dgs-worker-lingbot-map-git-cache", dockerfile)
        self.assertIn("CACHED_LINGBOT_MAP", dockerfile)
        self.assertNotIn("ARTDECO_REPO_URL", dockerfile)
        self.assertNotIn("SPEED3R_REPO_URL", dockerfile)
        self.assertNotIn('retry_pip --no-deps "git+$LINGBOT_MAP_REPO_URL@$LINGBOT_MAP_REPO_COMMIT"', dockerfile)
        self.assertNotIn("source.trainer", dockerfile)
        self.assertNotIn("lm-rs", (BACKEND_ROOT.parent / "scripts" / "bootstrap-repos.sh").read_text(encoding="utf-8").lower())
        self.assertNotIn("lm-rs", (BACKEND_ROOT.parent / "scripts" / "bootstrap-repos.ps1").read_text(encoding="utf-8").lower())

    def test_video_fine_runtime_is_not_registered(self) -> None:
        algorithms_source = (BACKEND_ROOT / "app" / "algorithms.py").read_text(encoding="utf-8")
        runner_source = (BACKEND_ROOT / "app" / "fine" / "runner.py").read_text(encoding="utf-8")

        self.assertNotIn("Video Fine ARTDECO + Speed3R-Pi3", algorithms_source)
        self.assertNotIn("video_fine_runtime", algorithms_source)
        self.assertNotIn("artdeco_video_runtime_status", algorithms_source)
        self.assertNotIn("run_video_artdeco_speed3r_pipeline", runner_source)

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
        self.assertFalse((fine_root / "amb3r_sfm.py").exists())
        self.assertFalse((fine_root / "amb3r_runtime").exists())
        self.assertFalse((fine_root / "edgs_init.py").exists())
        self.assertFalse((fine_root / "edgs_runtime" / "corr_init.py").exists())

    def test_compose_uses_one_worker_image(self) -> None:
        compose_source = (BACKEND_ROOT.parent / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("image: 3dgsbi-worker:local", compose_source)
        self.assertIn("target: worker", compose_source)
        self.assertNotIn("target: worker-preview", compose_source)
        self.assertNotIn("target: worker-fine", compose_source)

    def test_normalize_fine_pipeline_maps_legacy_to_official_fastgs_big(self) -> None:
        *_, normalize_fine_pipeline = import_fine_runtime()

        aliases = [
            "fused_quality_3dgs",
            "fine_fused_quality",
            "fused_quality",
            "mobilegs_lmrs",
            "litevggt_fastgs",
            "litevggt_fastgs_deblur",
            "litevggt_fastgs_deblur_gsplat",
            None,
        ]
        for alias in aliases:
            self.assertEqual(normalize_fine_pipeline(alias), "official_fastgs_big")

    def test_video_fine_pipeline_aliases_to_artdeco_speed3r(self) -> None:
        *_, normalize_fine_pipeline = import_fine_runtime()

        self.assertEqual(normalize_fine_pipeline("video_artdeco_speed3r"), "video_artdeco_speed3r")
        self.assertEqual(normalize_fine_pipeline("video_artdeco_litevggt"), "video_artdeco_speed3r")
        self.assertEqual(normalize_fine_pipeline("video_litevggt"), "video_artdeco_speed3r")
        self.assertEqual(normalize_fine_pipeline("artdeco_litevggt"), "video_artdeco_speed3r")

    def test_video_fine_does_not_route_to_runtime(self) -> None:
        runner_source = (BACKEND_ROOT / "app" / "fine" / "runner.py").read_text(encoding="utf-8")

        self.assertNotIn("if pipeline == VIDEO_PIPELINE_NAME:", runner_source)
        self.assertNotIn("run_video_artdeco_speed3r_pipeline", runner_source)
        self.assertIn("Unsupported fine pipeline", runner_source)

    def test_fine_worker_rejects_video_inputs(self) -> None:
        source = (BACKEND_ROOT / "app" / "fine_worker.py").read_text(encoding="utf-8")

        self.assertIn('project.input_type == "images"', source)
        self.assertIn('project.input_type == "video"', source)
        self.assertIn("Video fine reconstruction is disabled", source)
        self.assertNotIn("ensure_video_artdeco_weights", source)

    def test_video_fine_pipeline_raises_unsupported(self) -> None:
        try:
            from app.fine.runner import run_fine_pipeline
        except Exception as exc:
            raise unittest.SkipTest(f"fine runner dependencies unavailable: {exc}") from exc

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
                metrics_json=root / "work" / "metrics.json",
                lod_rad=None,
                source_version=7,
                options={},
            )

            with self.assertRaises(FineFailure) as raised:
                run_fine_pipeline(ctx)

        self.assertEqual(raised.exception.code, "UNSUPPORTED_FINE_PIPELINE")

    def test_blur_summary_reports_kept_images(self) -> None:
        BlurScore, _, summarize_blur_scores, *_ = import_fine_runtime()
        scores = [
            BlurScore(path=Path(f"{index}.jpg"), laplacian=140.0, gradient=50.0, fft_high_ratio=0.1)
            for index in range(10)
        ]

        summary = summarize_blur_scores(scores, reject_ratio=0.2)

        self.assertEqual(summary.rejected_images, 2)
        self.assertEqual(summary.kept_images, 8)
        self.assertIn("0.jpg", summary.per_frame_blur)

    def test_prepare_mobile_images_writes_normalized_blur_registry(self) -> None:
        try:
            from PIL import Image
            from app.fine.preprocess import BlurScore, prepare_mobile_images
        except Exception as exc:
            raise unittest.SkipTest(f"fine preprocess dependencies unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            for name in ("0.jpg", "1.jpg", "2.jpg"):
                Image.new("RGB", (16, 16), color=(128, 128, 128)).save(input_dir / name)

            with patch(
                "app.fine.preprocess.score_blur_images",
                return_value=[
                    BlurScore(input_dir / "0.jpg", 20.0, 45.0, 0.04),
                    BlurScore(input_dir / "1.jpg", 180.0, 55.0, 0.12),
                    BlurScore(input_dir / "2.jpg", 190.0, 55.0, 0.12),
                ],
            ):
                _, analysis = prepare_mobile_images(input_dir, output_dir, reject_ratio=0.0, min_images=3)

            self.assertTrue((output_dir / "000000.jpg").exists())
            self.assertEqual(analysis.per_frame_blur["000000.jpg"]["source_image"], "0.jpg")
            self.assertEqual(analysis.per_frame_blur["000000.jpg"]["training_image"], "000000.jpg")
            self.assertFalse(analysis.per_frame_blur["000000.jpg"]["rejected"])
            self.assertTrue(analysis.per_frame_blur["000000.jpg"]["blurred"])

    def test_prepare_mobile_images_records_rejected_blur_frames(self) -> None:
        try:
            from PIL import Image
            from app.fine.preprocess import BlurScore, prepare_mobile_images
        except Exception as exc:
            raise unittest.SkipTest(f"fine preprocess dependencies unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            for name in ("0.jpg", "1.jpg", "2.jpg", "3.jpg"):
                Image.new("RGB", (16, 16), color=(128, 128, 128)).save(input_dir / name)

            with patch(
                "app.fine.preprocess.score_blur_images",
                return_value=[
                    BlurScore(input_dir / "0.jpg", 5.0, 10.0, 0.01),
                    BlurScore(input_dir / "1.jpg", 180.0, 55.0, 0.12),
                    BlurScore(input_dir / "2.jpg", 190.0, 55.0, 0.12),
                    BlurScore(input_dir / "3.jpg", 200.0, 55.0, 0.12),
                ],
            ):
                _, analysis = prepare_mobile_images(input_dir, output_dir, reject_ratio=0.25, min_images=3)

            rejected = [entry for entry in analysis.per_frame_blur.values() if entry["rejected"]]
            kept = [entry for entry in analysis.per_frame_blur.values() if not entry["rejected"]]
            self.assertEqual(len(rejected), 1)
            self.assertEqual(len(kept), 3)
            self.assertTrue(rejected[0]["blurred"])
            self.assertIsNone(rejected[0]["training_image"])

    def test_any_blurry_training_image_triggers_deblur(self) -> None:
        BlurScore, _, summarize_blur_scores, _, deblur_mlp_enabled_by_default, _ = import_fine_runtime()
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
        BlurScore, _, summarize_blur_scores, _, deblur_mlp_enabled_by_default, _ = import_fine_runtime()
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

    def test_sfm_defaults_to_pycolmap(self) -> None:
        _, SceneBuildResult, _, build_scene, *_ = import_fine_runtime()
        expected = SceneBuildResult(Path("scene"), "pycolmap", 8, 8, 100, {"sfm_backend": "pycolmap"})
        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={}, progress=None)
        with tempfile.TemporaryDirectory() as tmp, patch("app.fine.runner.build_pycolmap_scene", return_value=expected) as pycolmap:
            result = build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8)

        self.assertEqual(result.backend, "pycolmap")
        pycolmap.assert_called_once()

    def test_build_scene_rejects_litevggt_for_image_fine(self) -> None:
        *_, build_scene, _, _ = import_fine_runtime()
        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={"fine_sfm_backend": "litevggt"}, progress=None)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FineFailure) as raised:
                build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8)

        self.assertEqual(raised.exception.code, "UNSUPPORTED_FINE_SFM_BACKEND")

    def test_explicit_pycolmap_backend_uses_same_default_path(self) -> None:
        _, SceneBuildResult, _, build_scene, *_ = import_fine_runtime()
        expected = SceneBuildResult(Path("scene"), "pycolmap", 3, 3, 42, {"sfm_backend": "pycolmap"})
        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={"fine_sfm_backend": "pycolmap"}, progress=None)
        with tempfile.TemporaryDirectory() as tmp, patch("app.fine.runner.build_pycolmap_scene", return_value=expected) as pycolmap:
            result = build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8)

        self.assertEqual(result.backend, "pycolmap")
        self.assertNotIn("sfm_backend_requested", result.metrics)
        pycolmap.assert_called_once()

    def test_colmap_backend_alias_maps_to_pycolmap(self) -> None:
        _, SceneBuildResult, _, build_scene, *_ = import_fine_runtime()
        expected = SceneBuildResult(Path("scene"), "pycolmap", 3, 3, 42, {"sfm_backend": "pycolmap"})
        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={"fine_sfm_backend": "colmap"}, progress=None)
        with tempfile.TemporaryDirectory() as tmp, patch("app.fine.runner.build_pycolmap_scene", return_value=expected) as pycolmap:
            result = build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8)

        self.assertEqual(result.backend, "pycolmap")
        self.assertEqual(result.metrics["sfm_backend_requested_alias"], "colmap_maps_to_pycolmap")
        pycolmap.assert_called_once()

    def test_removed_fine_sfm_backends_are_unsupported(self) -> None:
        *_, build_scene, _, _ = import_fine_runtime()
        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={"fine_sfm_backend": "amb3r"}, progress=None)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FineFailure) as raised:
                build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8)

        self.assertEqual(raised.exception.code, "UNSUPPORTED_FINE_SFM_BACKEND")

    def test_fine_weight_registration_uses_official_fastgs_without_litevggt_weights(self) -> None:
        fine_worker_source = (BACKEND_ROOT / "app" / "fine_worker.py").read_text(encoding="utf-8")
        weights_source = (BACKEND_ROOT / "app" / "preview" / "weights.py").read_text(encoding="utf-8")
        self.assertIn("download_model_weights", fine_worker_source)
        self.assertIn("weights_for_pipeline", fine_worker_source)
        self.assertNotIn("ensure_roma_weights", fine_worker_source)
        self.assertIn('"official_fastgs_big": ()', weights_source)
        self.assertNotIn("roma_outdoor.pth", weights_source)
        self.assertNotIn("speed3r_pi3/model.safetensors", weights_source)
        self.assertNotIn("ensure_amb3r_weight", fine_worker_source)

    def test_preview_litevggt_and_fine_runner_imports_are_isolated(self) -> None:
        sys.modules.pop("vggt", None)

        try:
            import app.preview.vendor.litevggt_runtime  # noqa: F401
            import app.fine.runner  # noqa: F401
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
        self.assertIn("fine_deblur_warmup_iters", trainer_source)
        self.assertIn("set_xyz_learning_rate", trainer_source)
        self.assertIn("deblur_regularization", deblur_source)

    def test_deblur_registry_controls_view_activation(self) -> None:
        try:
            from app.fine.mobilegs_trainer import is_blurred_view
        except Exception as exc:
            raise unittest.SkipTest(f"mobile trainer dependencies unavailable: {exc}") from exc

        registry = {
            "000000.jpg": {"blurred": True, "kind": "motion"},
            "000001.jpg": {"blurred": False, "kind": "sharp"},
        }

        self.assertTrue(is_blurred_view(SimpleNamespace(image_name="000000.jpg"), registry))
        self.assertFalse(is_blurred_view(SimpleNamespace(image_name="000001.jpg"), registry))
        self.assertFalse(is_blurred_view(SimpleNamespace(image_name="000002.jpg"), registry))

    def test_deblur_warmup_is_adaptive_and_less_than_iterations(self) -> None:
        try:
            from app.fine.mobilegs_trainer import resolve_deblur_warmup
        except Exception as exc:
            raise unittest.SkipTest(f"mobile trainer dependencies unavailable: {exc}") from exc

        self.assertEqual(resolve_deblur_warmup(500, {}, True), 166)
        self.assertEqual(resolve_deblur_warmup(500, {"fine_deblur_warmup_iters": 500}, True), 499)
        self.assertEqual(resolve_deblur_warmup(500, {}, False), 500)

    def test_deblur_densify_is_disabled_after_activation_but_prune_remains(self) -> None:
        trainer_source = (BACKEND_ROOT / "app" / "fine" / "mobilegs_trainer.py").read_text(encoding="utf-8")

        self.assertIn("can_densify = bool(iteration < opt.densify_until_iter and not deblur_active)", trainer_source)
        self.assertIn("deblur_densify_disabled_after_activation", trainer_source)
        self.assertIn("policy.apply_final_prune(gaussians, min_opacity=policy.final_prune_min_opacity)", trainer_source)

    def test_xyz_lr_setter_is_not_cumulative(self) -> None:
        try:
            from app.fine.mobilegs_trainer import optimizer_lr_value, set_xyz_learning_rate
        except Exception as exc:
            raise unittest.SkipTest(f"mobile trainer dependencies unavailable: {exc}") from exc

        dummy = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"name": "xyz", "lr": 0.01}]))

        set_xyz_learning_rate(dummy, optimizer_lr_value(dummy, "xyz") * 0.1)
        set_xyz_learning_rate(dummy, 0.01 * 0.1)

        self.assertAlmostEqual(dummy.optimizer.param_groups[0]["lr"], 0.001)

    def test_edgs_option_is_rejected_before_training(self) -> None:
        trainer_source = (BACKEND_ROOT / "app" / "fine" / "mobilegs_trainer.py").read_text(encoding="utf-8")
        runner_source = (BACKEND_ROOT / "app" / "fine" / "runner.py").read_text(encoding="utf-8")

        self.assertIn("fine_edgs_enabled", runner_source)
        self.assertIn("EDGS/RoMA dense initialization has been removed", runner_source)
        self.assertNotIn("initialize_edgs_if_enabled", trainer_source)
        self.assertNotIn("matches_per_ref=read_int", trainer_source)
        self.assertNotIn("opt.densify_until_iter = 0", trainer_source)

    def test_deblur_auto_controls_lmrs_default(self) -> None:
        *_, deblur_mlp_enabled_by_default, _ = import_fine_runtime()

        self.assertTrue(deblur_mlp_enabled_by_default("motion", {}))
        self.assertTrue(deblur_mlp_enabled_by_default("defocus", {}))
        self.assertTrue(deblur_mlp_enabled_by_default("mixed", {}))
        self.assertFalse(deblur_mlp_enabled_by_default("sharp", {}))
        self.assertFalse(deblur_mlp_enabled_by_default("motion", {"fine_deblur_enabled": "false"}))
        self.assertTrue(deblur_mlp_enabled_by_default("sharp", {"fine_deblur_enabled": "true"}))

    def test_worker_runtime_avoids_duplicate_cuda_stacks(self) -> None:
        worker_root = BACKEND_ROOT.parent / "worker"
        requirements = (worker_root / "requirements.txt").read_text(encoding="utf-8").lower()
        dockerfile = (worker_root / "Dockerfile").read_text(encoding="utf-8").lower()
        combined = requirements + "\n" + dockerfile

        self.assertNotIn("transformer-engine", requirements)
        self.assertNotIn("pycolmap", requirements)
        self.assertIn("transformer-engine[pytorch]==2.4.0", dockerfile)
        self.assertIn("pycolmap==3.12.6", dockerfile)
        self.assertNotIn("spconv-cu118==2.3.8", dockerfile)
        self.assertNotIn("torch-scatter==2.1.2", dockerfile)
        self.assertIn("cython==0.29.37", requirements)
        self.assertNotIn("romatch==0.1.2", requirements)
        self.assertIn("scikit-learn==1.6.1", requirements)
        self.assertIn("seaborn==0.13.2", requirements)
        self.assertIn("evo==1.36.4", requirements)
        self.assertIn("cupy-cuda12x==13.6.0", requirements)
        self.assertIn("moderngl==5.12.0", requirements)
        self.assertIn("moderngl-window==2.4.6", requirements)
        self.assertIn("glfw", requirements)
        self.assertIn("pyglm", requirements)
        self.assertIn("msgpack", requirements)
        self.assertIn("trimesh[easy]", requirements)
        self.assertIn("hf_endpoint", dockerfile)
        self.assertIn("https://hf-mirror.com", dockerfile)
        self.assertNotIn("/model-cache/roma", dockerfile)
        self.assertNotIn("compvis/edgs", dockerfile)
        constraints = (worker_root / "constraints.txt").read_text(encoding="utf-8").lower()
        self.assertIn("torch==2.8.0", constraints)
        self.assertIn("torchvision==0.23.0", constraints)
        self.assertNotIn("flashinfer-python==", requirements)
        self.assertNotIn("flashinfer-cubin", requirements)
        self.assertIn("--index-url https://pypi.org/simple flashinfer-python", dockerfile)
        self.assertNotIn('version("flashinfer-python") ==', dockerfile)
        self.assertIn("timm==0.6.7", requirements)
        self.assertIn("addict==2.4.0", requirements)
        self.assertIn("gsplat==1.5.3", requirements)
        self.assertIn("safetensors==0.7.0", requirements)
        self.assertNotIn("pypose==0.7.3", requirements)
        self.assertNotIn("natsort==8.4.0", requirements)
        self.assertNotIn("/model-cache/amb3r", dockerfile)
        self.assertNotIn("/model-cache/speed3r_pi3", dockerfile)
        self.assertNotIn("/model-cache/mast3r", dockerfile)
        self.assertNotIn("artdeco_repo_commit", dockerfile)
        self.assertNotIn("speed3r_repo_commit", dockerfile)
        self.assertIn("fastgs_vendor_root=/app/app/fine/vendor/fastgs", dockerfile)
        self.assertNotIn("fastgs_repo_commit", dockerfile)
        self.assertNotIn("fastgs_repo_url", dockerfile)
        self.assertIn("cached_wheel_install diff-gaussian-rasterization-fastgs", dockerfile)
        self.assertNotIn("env pythonpath=${artdeco_root}/vslam/thirdparty/mast3r/dust3r/croco", dockerfile)
        self.assertIn("three-dgs-worker-extension-wheel-cache", dockerfile)
        self.assertNotIn("three-dgs-worker-lmrs-git-cache", dockerfile)
        self.assertNotIn("lmrs_repo_url", dockerfile)
        self.assertNotIn("lmrs_root", dockerfile)
        self.assertIn("cached_wheel_install simple-knn /app/app/fine/vendor/fastgs/submodules/simple-knn", dockerfile)
        self.assertIn("retry_pip --force-reinstall --no-deps \"$wheel\"", dockerfile)
        self.assertIn("libc10.so", dockerfile)
        self.assertIn("/etc/ld.so.conf.d/pytorch.conf", dockerfile)
        self.assertNotIn("copy backend/app/fine/video/artdeco_optimizer_compat.py /tmp", dockerfile)
        self.assertNotIn("cached_wheel_install simple-knn-artdeco", dockerfile)
        self.assertNotIn("cached_wheel_install pyimgui", dockerfile)
        self.assertNotIn("cached_wheel_install curope", dockerfile)
        self.assertNotIn("artdeco pi3 rope2d patch did not apply", dockerfile)
        self.assertNotIn("artdeco simple-knn distindex2 import ok", dockerfile)
        self.assertNotIn("artdeco vslam visualization dependencies import ok", dockerfile)
        self.assertNotIn("artdeco mapping entrypoint imports ok", dockerfile)
        self.assertNotIn("artdeco rope2d cuda extension import ok", dockerfile)
        self.assertNotIn("speed3r rope2d cuda extension import ok", dockerfile)
        self.assertNotIn("cached_wheel_install artdeco-vslam", dockerfile)
        self.assertLess(
            dockerfile.index("copy backend/app/fine/vendor/fastgs ./app/fine/vendor/fastgs"),
            dockerfile.index("cached_wheel_install fused-ssim"),
        )
        self.assertGreater(
            dockerfile.index("copy backend/app ./app"),
            dockerfile.index("cached_wheel_install lingbot-map"),
        )
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
        self.assertNotIn("flashinfer-cubin", dockerfile)
        self.assertNotIn("unified worker runtime import ok", dockerfile)
        self.assertNotIn("kaolin", combined)
        self.assertNotIn("open3d", combined)
        self.assertNotIn("xformers", combined)
        self.assertNotIn("gradio", combined)
        self.assertNotIn("pyrealsense2", combined)
        self.assertNotIn("geocalib", combined)
        self.assertNotIn("pytorch3d", combined)
        self.assertNotIn("gdown", combined)
        self.assertNotIn("pip install torch", dockerfile)
        self.assertNotIn("pip install torchvision", dockerfile)


if __name__ == "__main__":
    unittest.main()
