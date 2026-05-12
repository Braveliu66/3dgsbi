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
    def test_fine_runtime_registers_litevggt_runtime(self) -> None:
        algorithms_source = (BACKEND_ROOT / "app" / "algorithms.py").read_text(encoding="utf-8")
        fine_status_block = algorithms_source.split("def fine_runtime_status", 1)[1].split("def ", 1)[0]

        self.assertIn("litevggt_runtime", fine_status_block)
        self.assertIn("transformer_engine", fine_status_block)
        self.assertIn("diff_gaussian_rasterization_fastgs", fine_status_block)
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

    def test_lmrs_is_isolated_by_default(self) -> None:
        runner_source = (BACKEND_ROOT / "app" / "fine" / "runner.py").read_text(encoding="utf-8")
        trainer_source = (BACKEND_ROOT / "app" / "fine" / "mobilegs_trainer.py").read_text(encoding="utf-8")

        self.assertIn("lm_default = iterations", runner_source)
        self.assertNotIn("min(15_000, iterations)", runner_source)
        self.assertIn('read_bool((options or {}).get("fine_lmrs_enabled"), False)', trainer_source)
        self.assertIn('"active": False', trainer_source)
        self.assertIn("LM-RS temporarily isolated due to unstable local backend", trainer_source)

    def test_worker_builds_vendored_3dgs_extensions_without_lmrs(self) -> None:
        dockerfile = (BACKEND_ROOT.parent / "worker" / "Dockerfile").read_text(encoding="utf-8")

        self.assertNotIn("lm-rs", dockerfile.lower())
        self.assertNotIn("LMRS_ROOT", dockerfile)
        self.assertNotIn("lmrs-fastgs-compact-box.patch", dockerfile)
        self.assertNotIn("fastgs-cuda-metric-accumulation.patch", dockerfile)
        self.assertIn("backend/app/fine/local_3dgs/vendor/gaussian_splatting/submodules/diff-gaussian-rasterization", dockerfile)
        self.assertIn("FASTGS_REPO_URL=https://github.com/fastgs/FastGS.git", dockerfile)
        self.assertIn("FASTGS_REPO_COMMIT=44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f", dockerfile)
        self.assertIn("cached_wheel_install diff-gaussian-rasterization-fastgs", dockerfile)
        self.assertIn("backend/app/fine/local_3dgs/vendor/gaussian_splatting/submodules/simple-knn", dockerfile)
        self.assertIn("cached_wheel_install diff-gaussian-rasterization /app/app/fine/local_3dgs/vendor/gaussian_splatting/submodules/diff-gaussian-rasterization", dockerfile)
        self.assertNotIn("cached_wheel_install simple-knn-local", dockerfile)
        self.assertIn("cached_wheel_install simple-knn-artdeco", dockerfile)
        self.assertIn("retry_pip --force-reinstall --no-deps \"$wheel\"", dockerfile)
        self.assertIn("ARTDECO simple-knn distIndex2 import ok", dockerfile)
        self.assertIn("libc10.so", dockerfile)
        self.assertIn("/etc/ld.so.conf.d/pytorch.conf", dockerfile)
        self.assertIn("retry_git", dockerfile)
        self.assertIn("libeigen3-dev", dockerfile)
        self.assertIn("test -f /usr/include/eigen3/Eigen/Sparse", dockerfile)
        self.assertIn('"/usr/include/eigen3"', dockerfile)
        self.assertNotIn("submodule update --init --recursive VSLAM/thirdparty/eigen", dockerfile)
        self.assertIn("D11.scalar_type()", dockerfile)
        self.assertIn("dx.pow(2).sum().sqrt()", dockerfile)
        self.assertIn("ensure_git_checkout", dockerfile)
        self.assertIn("cat-file -e", dockerfile)
        self.assertIn("three-dgs-worker-lingbot-map-git-cache", dockerfile)
        self.assertIn("CACHED_LINGBOT_MAP", dockerfile)
        self.assertNotIn('retry_pip --no-deps "git+$LINGBOT_MAP_REPO_URL@$LINGBOT_MAP_REPO_COMMIT"', dockerfile)
        self.assertNotIn("source.trainer", dockerfile)
        self.assertNotIn("lm-rs", (BACKEND_ROOT.parent / "scripts" / "bootstrap-repos.sh").read_text(encoding="utf-8").lower())
        self.assertNotIn("lm-rs", (BACKEND_ROOT.parent / "scripts" / "bootstrap-repos.ps1").read_text(encoding="utf-8").lower())

    def test_artdeco_command_uses_official_quality_defaults_without_gaussian_cap(self) -> None:
        trainer_source = (BACKEND_ROOT / "app" / "fine" / "video" / "artdeco_trainer.py").read_text(encoding="utf-8")
        entrypoint_source = (BACKEND_ROOT / "app" / "fine" / "video" / "artdeco_entrypoint.py").read_text(encoding="utf-8")

        self.assertNotIn("ARTDECO_MAX_GAUSSIANS", trainer_source)
        self.assertNotIn("gaussian cap pruned", entrypoint_source)
        self.assertIn("gaussian_total_cap=disabled", entrypoint_source)
        self.assertIn('options.get("fine_artdeco_gs_add_ratio"), 1.0', trainer_source)
        self.assertIn('options.get("fine_artdeco_visible_threshold"), 0.0', trainer_source)
        self.assertIn('options.get("fine_artdeco_sh_degree"), 3', trainer_source)
        self.assertIn('options.get("fine_artdeco_max_active_keyframes"), 400', trainer_source)

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
        self.assertTrue((fine_root / "edgs_init.py").exists())
        self.assertTrue((fine_root / "edgs_runtime" / "corr_init.py").exists())

    def test_compose_uses_one_worker_image(self) -> None:
        compose_source = (BACKEND_ROOT.parent / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("image: 3dgsbi-worker:local", compose_source)
        self.assertIn("target: worker", compose_source)
        self.assertNotIn("target: worker-preview", compose_source)
        self.assertNotIn("target: worker-fine", compose_source)

    def test_legacy_fine_pipeline_aliases_to_litevggt_fastgs(self) -> None:
        *_, normalize_fine_pipeline = import_fine_runtime()

        self.assertEqual(normalize_fine_pipeline("fused_quality_3dgs"), "litevggt_fastgs_deblur_gsplat")
        self.assertEqual(normalize_fine_pipeline("mobilegs_lmrs"), "litevggt_fastgs_deblur_gsplat")
        self.assertEqual(normalize_fine_pipeline(None), "litevggt_fastgs_deblur_gsplat")

    def test_video_fine_pipeline_aliases_to_artdeco_speed3r(self) -> None:
        *_, normalize_fine_pipeline = import_fine_runtime()

        self.assertEqual(normalize_fine_pipeline("video_artdeco_speed3r"), "video_artdeco_speed3r")
        self.assertEqual(normalize_fine_pipeline("video_artdeco_litevggt"), "video_artdeco_speed3r")
        self.assertEqual(normalize_fine_pipeline("video_litevggt"), "video_artdeco_speed3r")
        self.assertEqual(normalize_fine_pipeline("artdeco_litevggt"), "video_artdeco_speed3r")

    def test_video_fine_does_not_route_through_image_training(self) -> None:
        runner_source = (BACKEND_ROOT / "app" / "fine" / "runner.py").read_text(encoding="utf-8")
        video_block = runner_source.split("if pipeline == VIDEO_PIPELINE_NAME:", 1)[1].split("if pipeline != PIPELINE_NAME:", 1)[0]

        self.assertIn("run_video_artdeco_speed3r_pipeline", video_block)
        self.assertNotIn("train_mobile_3dgs", video_block)
        self.assertNotIn("build_scene", video_block)

        video_source = "\n".join(path.read_text(encoding="utf-8") for path in (BACKEND_ROOT / "app" / "fine" / "video").glob("*.py"))
        for forbidden in ("train_mobile_3dgs", "build_scene(", "AMB3R", "MobileGS", "LM-RS", "DeblurMLP"):
            self.assertNotIn(forbidden, video_source)

    def test_fine_worker_splits_image_and_video_inputs(self) -> None:
        source = (BACKEND_ROOT / "app" / "fine_worker.py").read_text(encoding="utf-8")

        self.assertIn('project.input_type == "images"', source)
        self.assertIn('project.input_type == "video"', source)
        self.assertIn("download_single_video", source)
        self.assertIn("len(video_items) != 1 or len(project.media) != 1", source)
        self.assertIn("ensure_video_artdeco_weights", source)

    def test_video_artdeco_mock_output_becomes_final_artifacts(self) -> None:
        from app.fine.video.types import ArtdecoTrainingResult, ExtractedVideoFrames
        import app.fine.video.pipeline as video_pipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_video = root / "clip.mp4"
            input_video.write_bytes(b"video")
            dataset_root = root / "dataset"
            (dataset_root / "images").mkdir(parents=True)
            gs_ply = root / "artdeco_output" / "point_clouds" / "gs.ply"
            gs_ply.parent.mkdir(parents=True)
            gs_ply.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
            frames = ExtractedVideoFrames(
                frames_dir=dataset_root / "images",
                dataset_root=dataset_root,
                count=3,
                width=640,
                height=480,
                fps=30.0,
                source_video=input_video,
            )
            training = ArtdecoTrainingResult(output_dir=root / "artdeco_output", gs_ply=gs_ply, metrics={"artdeco_metric": 1})
            ctx = FineContext(
                task_id="task",
                project_id="project",
                pipeline="video_artdeco_speed3r",
                input_dir=input_video.parent,
                input_video=input_video,
                work_dir=root / "work",
                model_cache_dir=root / "model-cache",
                final_ply=root / "work" / "final.ply",
                final_spz=root / "work" / "final_web.spz",
                metrics_json=root / "work" / "metrics.json",
                lod_rad=None,
                source_version=7,
                options={},
            )

            with patch.object(video_pipeline, "extract_video_frames", return_value=frames), patch.object(
                video_pipeline, "run_artdeco_speed3r_training", return_value=training
            ), patch.object(video_pipeline, "convert_ply_to_spz", side_effect=lambda _ply, spz: (spz.write_bytes(b"spz"), 11)[1]):
                result = video_pipeline.run_video_artdeco_speed3r_pipeline(
                    ctx,
                    settings=SimpleNamespace(),
                    lod_builder=lambda _ctx: None,
                )

            self.assertEqual(ctx.final_ply.read_bytes(), gs_ply.read_bytes())
            self.assertEqual(ctx.final_spz.read_bytes(), b"spz")
            self.assertEqual(result.metrics["pipeline"], "video_artdeco_speed3r")
            self.assertEqual(result.metrics["splat_count"], 11)
            self.assertEqual(result.metrics["artdeco_metric"], 1)

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
        from PIL import Image
        try:
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
            self.assertTrue(analysis.per_frame_blur["000000.jpg"]["blurred"])

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

    def test_sfm_defaults_to_litevggt(self) -> None:
        _, SceneBuildResult, _, build_scene, *_ = import_fine_runtime()
        expected = SceneBuildResult(Path("scene"), "litevggt", 8, 8, 100, {"sfm_backend": "litevggt"})
        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={}, progress=None)
        with tempfile.TemporaryDirectory() as tmp, patch("app.fine.runner.build_litevggt_scene", return_value=expected) as litevggt:
            result = build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8)

        self.assertEqual(result.backend, "litevggt")
        litevggt.assert_called_once()

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

    def test_fine_weight_registration_uses_litevggt_by_default(self) -> None:
        fine_worker_source = (BACKEND_ROOT / "app" / "fine_worker.py").read_text(encoding="utf-8")
        self.assertIn("download_model_weights", fine_worker_source)
        self.assertIn("weights_for_pipeline", fine_worker_source)
        self.assertIn("ensure_roma_weights", fine_worker_source)
        self.assertIn('"litevggt/te_dict.pt"', (BACKEND_ROOT / "app" / "preview" / "weights.py").read_text(encoding="utf-8"))
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
        self.assertIn("policy.apply_final_prune(gaussians)", trainer_source)

    def test_xyz_lr_setter_is_not_cumulative(self) -> None:
        try:
            from app.fine.mobilegs_trainer import optimizer_lr_value, set_xyz_learning_rate
        except Exception as exc:
            raise unittest.SkipTest(f"mobile trainer dependencies unavailable: {exc}") from exc

        dummy = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"name": "xyz", "lr": 0.01}]))

        set_xyz_learning_rate(dummy, optimizer_lr_value(dummy, "xyz") * 0.1)
        set_xyz_learning_rate(dummy, 0.01 * 0.1)

        self.assertAlmostEqual(dummy.optimizer.param_groups[0]["lr"], 0.001)

    def test_edgs_initializes_before_training_setup_and_disables_densification(self) -> None:
        trainer_source = (BACKEND_ROOT / "app" / "fine" / "mobilegs_trainer.py").read_text(encoding="utf-8")
        runner_source = (BACKEND_ROOT / "app" / "fine" / "runner.py").read_text(encoding="utf-8")
        edgs_source = (BACKEND_ROOT / "app" / "fine" / "edgs_init.py").read_text(encoding="utf-8")
        corr_source = (BACKEND_ROOT / "app" / "fine" / "edgs_runtime" / "corr_init.py").read_text(encoding="utf-8")

        self.assertLess(trainer_source.index("initialize_edgs_if_enabled"), trainer_source.index("gaussians.training_setup(opt)"))
        self.assertIn("fine_edgs_enabled", trainer_source)
        self.assertIn("matches_per_ref=read_int(options.get(\"fine_edgs_matches_per_ref\")", trainer_source)
        self.assertIn("opt.densify_until_iter = 0", trainer_source)
        self.assertIn("densification_disabled_by_edgs", trainer_source)
        self.assertIn("disabled_by_edgs", runner_source)
        self.assertIn("EDGS_RUNTIME_UNAVAILABLE", edgs_source)
        self.assertIn("init_gaussians_with_corr", corr_source)
        self.assertIn("roma.match", corr_source)
        self.assertNotIn("gradio", corr_source.lower())

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
        self.assertNotIn("spconv-cu118==2.3.8", dockerfile)
        self.assertNotIn("torch-scatter==2.1.2", dockerfile)
        self.assertIn("cython==0.29.37", requirements)
        self.assertIn("romatch==0.1.2", requirements)
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
        self.assertIn("/model-cache/roma", dockerfile)
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
        self.assertIn("pypose==0.7.3", requirements)
        self.assertIn("natsort==8.4.0", requirements)
        self.assertNotIn("/model-cache/amb3r", dockerfile)
        self.assertIn("/model-cache/speed3r_pi3", dockerfile)
        self.assertIn("/model-cache/mast3r", dockerfile)
        self.assertIn("artdeco_repo_commit", dockerfile)
        self.assertIn("speed3r_repo_commit", dockerfile)
        self.assertIn("fastgs_repo_commit", dockerfile)
        self.assertIn("cached_wheel_install diff-gaussian-rasterization-fastgs", dockerfile)
        self.assertIn("env pythonpath=${artdeco_root}/vslam/thirdparty/mast3r/dust3r/croco", dockerfile)
        self.assertIn("three-dgs-worker-extension-wheel-cache", dockerfile)
        self.assertNotIn("three-dgs-worker-lmrs-git-cache", dockerfile)
        self.assertNotIn("lmrs_repo_url", dockerfile)
        self.assertNotIn("lmrs_root", dockerfile)
        self.assertNotIn("cached_wheel_install simple-knn-local", dockerfile)
        self.assertIn("retry_pip --force-reinstall --no-deps \"$wheel\"", dockerfile)
        self.assertIn("libc10.so", dockerfile)
        self.assertIn("/etc/ld.so.conf.d/pytorch.conf", dockerfile)
        self.assertNotIn("copy backend/app/fine/video/artdeco_optimizer_compat.py /tmp", dockerfile)
        self.assertIn("cached_wheel_install simple-knn-artdeco", dockerfile)
        self.assertIn("cached_wheel_install pyimgui", dockerfile)
        self.assertIn("cached_wheel_install curope", dockerfile)
        self.assertIn("artdeco pi3 rope2d patch did not apply", dockerfile)
        self.assertIn("artdeco simple-knn distindex2 import ok", dockerfile)
        self.assertNotIn("artdeco vslam visualization dependencies import ok", dockerfile)
        self.assertNotIn("artdeco mapping entrypoint imports ok", dockerfile)
        self.assertNotIn("artdeco rope2d cuda extension import ok", dockerfile)
        self.assertNotIn("speed3r rope2d cuda extension import ok", dockerfile)
        self.assertIn("cached_wheel_install artdeco-vslam \"$artdeco_root/vslam\"", dockerfile)
        self.assertLess(
            dockerfile.index("copy backend/app/fine/local_3dgs/vendor/gaussian_splatting/submodules/fused-ssim"),
            dockerfile.index("cached_wheel_install fused-ssim"),
        )
        self.assertGreater(
            dockerfile.index("copy backend/app ./app"),
            dockerfile.index("cached_wheel_install artdeco-vslam"),
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
