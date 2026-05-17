from __future__ import annotations

import sys
import struct
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

    def test_removed_mobilegs_lmrs_code_is_not_present(self) -> None:
        fine_root = BACKEND_ROOT / "app" / "fine"

        self.assertFalse((fine_root / "mobilegs_trainer.py").exists())
        self.assertFalse((fine_root / "local_3dgs").exists())
        self.assertFalse((fine_root / "lmrs_runtime.py").exists())
        self.assertFalse((fine_root / "fastgs_policy.py").exists())
        self.assertFalse((fine_root / "litevggt_scene.py").exists())

    def test_image_fine_runner_uses_official_fastgs_big_not_lmrs(self) -> None:
        runner_source = (BACKEND_ROOT / "app" / "fine" / "runner.py").read_text(encoding="utf-8")

        self.assertIn("train_official_fastgs_big", runner_source)
        self.assertNotIn("train_mobile_3dgs", runner_source)
        self.assertNotIn("fine_lm_start_iter", runner_source)
        self.assertNotIn("lm_default = iterations", runner_source)
        self.assertNotIn("min(15_000, iterations)", runner_source)

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

    def test_fine_code_keeps_only_current_integration_boundary(self) -> None:
        fine_root = BACKEND_ROOT / "app" / "fine"

        self.assertTrue((fine_root / "official_fastgs_big_trainer.py").exists())
        self.assertTrue((fine_root / "vendor" / "fastgs" / "train.py").exists())
        self.assertTrue((fine_root / "vendor" / "fastgs" / "scene" / "blur_kernel.py").exists())
        self.assertTrue((fine_root / "option_utils.py").exists())
        self.assertFalse((fine_root / "mobilegs_trainer.py").exists())
        self.assertFalse((fine_root / "local_3dgs").exists())
        self.assertFalse((fine_root / "lmrs_runtime.py").exists())
        self.assertFalse((fine_root / "video").exists())
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

    def test_normalize_fine_pipeline_only_accepts_current_name(self) -> None:
        *_, normalize_fine_pipeline = import_fine_runtime()

        self.assertEqual(normalize_fine_pipeline(None), "official_fastgs_big")
        self.assertEqual(normalize_fine_pipeline("official_fastgs_big"), "official_fastgs_big")
        self.assertEqual(normalize_fine_pipeline("mobilegs_lmrs"), "mobilegs_lmrs")
        self.assertEqual(normalize_fine_pipeline("video_artdeco_speed3r"), "video_artdeco_speed3r")

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
                viewer_meta_json=root / "work" / "final_viewer_meta.json",
                metrics_json=root / "work" / "metrics.json",
                lod_rad=None,
                source_version=7,
                options={},
            )

            with self.assertRaises(FineFailure) as raised:
                run_fine_pipeline(ctx)

        self.assertEqual(raised.exception.code, "UNSUPPORTED_FINE_PIPELINE")

    def test_viewer_ply_scale_multiplier_and_clamp_binary_ply(self) -> None:
        from app.fine.viewer_meta import write_scaled_viewer_ply

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 2\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float scale_0\n"
            "property float scale_1\n"
            "property float scale_2\n"
            "end_header\n"
        ).encode("ascii")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "final.ply"
            target = root / "final_viewer.ply"
            source.write_bytes(
                header
                + struct.pack("<ffffff", 0.0, 0.0, 0.0, 0.0, -2.0, -4.0)
                + struct.pack("<ffffff", 1.0, 1.0, 1.0, 2.0, -2.0, -4.0)
            )

            metrics = write_scaled_viewer_ply(source, target, scale_multiplier=0.5, max_scale=1.0)

            body = target.read_bytes().split(b"end_header\n", 1)[1]
            first = struct.unpack("<ffffff", body[:24])
            second = struct.unpack("<ffffff", body[24:48])
            self.assertAlmostEqual(first[3], -0.69314718, places=6)
            self.assertAlmostEqual(first[4], -2.69314718, places=6)
            self.assertAlmostEqual(second[3], 0.0, places=6)
            self.assertEqual(metrics["viewer_scale_clamped"], 1)
            self.assertEqual(metrics["viewer_scale_fields"], ["scale_0", "scale_1", "scale_2"])

    def test_blur_summary_reports_kept_images(self) -> None:
        BlurScore, _, summarize_blur_scores, *_ = import_fine_runtime()
        scores = [
            BlurScore(path=Path(f"{index}.jpg"), laplacian=140.0, gradient=50.0, fft_high_ratio=0.1)
            for index in range(9)
        ]
        scores.append(BlurScore(path=Path("extreme.jpg"), laplacian=5.0, gradient=10.0, fft_high_ratio=0.01))

        summary = summarize_blur_scores(scores, reject_ratio=0.2)

        self.assertEqual(summary.rejected_images, 1)
        self.assertEqual(summary.kept_images, 9)
        self.assertIn("0.jpg", summary.per_frame_blur)

    def test_medium_blurry_images_are_kept_under_reject_ratio(self) -> None:
        BlurScore, _, summarize_blur_scores, _, deblur_mlp_enabled_by_default, _ = import_fine_runtime()
        scores = [
            BlurScore(path=Path(f"sharp_{index}.jpg"), laplacian=180.0, gradient=55.0, fft_high_ratio=0.12)
            for index in range(9)
        ]
        scores.append(BlurScore(path=Path("medium_blur.jpg"), laplacian=50.0, gradient=45.0, fft_high_ratio=0.04))

        summary = summarize_blur_scores(scores, reject_ratio=0.1)

        self.assertEqual(summary.rejected_images, 0)
        self.assertEqual(summary.training_blur_frames, 1)
        self.assertTrue(deblur_mlp_enabled_by_default(summary.mode, {}))

    def test_relative_moderate_defocus_triggers_deblur(self) -> None:
        BlurScore, _, summarize_blur_scores, _, deblur_mlp_enabled_by_default, _ = import_fine_runtime()
        scores = [
            BlurScore(path=Path("000000.jpg"), laplacian=224.025827, gradient=851.799255, fft_high_ratio=0.36855204),
            BlurScore(path=Path("000001.jpg"), laplacian=263.694757, gradient=766.077026, fft_high_ratio=0.39951025),
            BlurScore(path=Path("000002.jpg"), laplacian=1179.796933, gradient=926.120972, fft_high_ratio=0.4989555),
            BlurScore(path=Path("000003.jpg"), laplacian=312.785666, gradient=836.301392, fft_high_ratio=0.42024099),
            BlurScore(path=Path("000004.jpg"), laplacian=339.566689, gradient=891.687195, fft_high_ratio=0.42951662),
            BlurScore(path=Path("000005.jpg"), laplacian=367.996115, gradient=807.933167, fft_high_ratio=0.41953409),
            BlurScore(path=Path("000006.jpg"), laplacian=269.82217, gradient=831.107666, fft_high_ratio=0.39806543),
            BlurScore(path=Path("000007.jpg"), laplacian=265.300783, gradient=838.973206, fft_high_ratio=0.38816451),
            BlurScore(path=Path("000008.jpg"), laplacian=1072.115027, gradient=903.514221, fft_high_ratio=0.49394342),
            BlurScore(path=Path("000009.jpg"), laplacian=130.345808, gradient=776.81012, fft_high_ratio=0.34764597),
            BlurScore(path=Path("000010.jpg"), laplacian=269.512496, gradient=732.538025, fft_high_ratio=0.391968),
            BlurScore(path=Path("000011.jpg"), laplacian=174.387577, gradient=789.878479, fft_high_ratio=0.36938038),
            BlurScore(path=Path("000012.jpg"), laplacian=457.84512, gradient=900.005554, fft_high_ratio=0.44153096),
            BlurScore(path=Path("000013.jpg"), laplacian=1200.670854, gradient=896.39386, fft_high_ratio=0.50059039),
            BlurScore(path=Path("000014.jpg"), laplacian=640.144877, gradient=898.354065, fft_high_ratio=0.47315539),
            BlurScore(path=Path("000015.jpg"), laplacian=144.078116, gradient=833.57666, fft_high_ratio=0.32637557),
            BlurScore(path=Path("000016.jpg"), laplacian=475.532711, gradient=869.833313, fft_high_ratio=0.4458538),
            BlurScore(path=Path("000017.jpg"), laplacian=222.696251, gradient=790.070862, fft_high_ratio=0.38397671),
        ]

        summary = summarize_blur_scores(scores, reject_ratio=0.0)

        self.assertGreater(summary.training_blur_frames, 0)
        self.assertEqual(summary.mode, "defocus")
        self.assertEqual(summary.deblur_trigger_reason, "default_mixed")
        self.assertTrue(deblur_mlp_enabled_by_default(summary.mode, {}))

    def test_low_texture_frame_is_not_marked_blurred(self) -> None:
        BlurScore, _, summarize_blur_scores, *_ = import_fine_runtime()
        scores = [
            BlurScore(path=Path(f"sharp_{index}.jpg"), laplacian=180.0, gradient=55.0, fft_high_ratio=0.12, texture_density=0.10)
            for index in range(9)
        ]
        scores.append(
            BlurScore(
                path=Path("white_wall.jpg"),
                laplacian=10.0,
                gradient=5.0,
                fft_high_ratio=0.01,
                texture_density=0.0,
                exposure_bad_ratio=0.0,
            )
        )

        summary = summarize_blur_scores(scores, reject_ratio=0.0)

        self.assertEqual(summary.training_blur_frames, 0)
        self.assertEqual(summary.mode, "sharp")
        self.assertFalse(summary.per_frame_blur["white_wall.jpg"]["blurred"])
        self.assertEqual(summary.per_frame_blur["white_wall.jpg"]["quality_label"], "sharp_low_texture")

    def test_exposure_bad_frame_can_be_rejected_without_deblur(self) -> None:
        BlurScore, _, summarize_blur_scores, *_ = import_fine_runtime()
        scores = [
            BlurScore(path=Path("overexposed.jpg"), laplacian=100.0, gradient=40.0, fft_high_ratio=0.08, exposure_bad_ratio=0.60),
            BlurScore(path=Path("sharp_0.jpg"), laplacian=180.0, gradient=55.0, fft_high_ratio=0.12),
            BlurScore(path=Path("sharp_1.jpg"), laplacian=190.0, gradient=55.0, fft_high_ratio=0.12),
            BlurScore(path=Path("sharp_2.jpg"), laplacian=200.0, gradient=55.0, fft_high_ratio=0.12),
        ]

        summary = summarize_blur_scores(scores, reject_ratio=0.25, min_images=3)

        self.assertEqual(summary.rejected_images, 1)
        self.assertEqual(summary.training_blur_frames, 0)
        self.assertEqual(summary.mode, "sharp")
        self.assertEqual(summary.per_frame_blur["overexposed.jpg"]["quality_label"], "low_quality")

    def test_prepare_fine_images_writes_normalized_blur_registry(self) -> None:
        try:
            from PIL import Image
            from app.fine.preprocess import BlurScore, prepare_fine_images
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
                _, analysis = prepare_fine_images(input_dir, output_dir, reject_ratio=0.0, min_images=3)

            self.assertTrue((output_dir / "000000.jpg").exists())
            self.assertEqual(analysis.per_frame_blur["000000.jpg"]["source_image"], "0.jpg")
            self.assertEqual(analysis.per_frame_blur["000000.jpg"]["training_image"], "000000.jpg")
            self.assertFalse(analysis.per_frame_blur["000000.jpg"]["rejected"])
            self.assertTrue(analysis.per_frame_blur["000000.jpg"]["blurred"])

    def test_prepare_fine_images_records_rejected_blur_frames(self) -> None:
        try:
            from PIL import Image
            from app.fine.preprocess import BlurScore, prepare_fine_images
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
                _, analysis = prepare_fine_images(input_dir, output_dir, reject_ratio=0.25, min_images=3)

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
        self.assertEqual(summary.deblur_trigger_reason, "default_mixed")

    def test_rejected_blurry_image_still_triggers_deblur(self) -> None:
        BlurScore, _, summarize_blur_scores, _, deblur_mlp_enabled_by_default, _ = import_fine_runtime()
        scores = [
            BlurScore(path=Path(f"sharp_{index}.jpg"), laplacian=180.0, gradient=55.0, fft_high_ratio=0.12)
            for index in range(9)
        ]
        scores.append(BlurScore(path=Path("blur.jpg"), laplacian=10.0, gradient=10.0, fft_high_ratio=0.01))

        summary = summarize_blur_scores(scores, reject_ratio=0.1)

        self.assertEqual(summary.training_blur_frames, 0)
        self.assertEqual(summary.rejected_blur_frames, 1)
        self.assertEqual(summary.mode, "defocus")
        self.assertEqual(summary.deblur_trigger_reason, "default_mixed")
        self.assertTrue(deblur_mlp_enabled_by_default(summary.mode, {}))

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

    def test_deblur_gtnet_lives_in_current_fastgs_vendor_path(self) -> None:
        deblur_source = (BACKEND_ROOT / "app" / "fine" / "vendor" / "fastgs" / "scene" / "blur_kernel.py").read_text(encoding="utf-8")
        trainer_source = (BACKEND_ROOT / "app" / "fine" / "vendor" / "fastgs" / "train.py").read_text(encoding="utf-8")

        self.assertFalse((BACKEND_ROOT / "app" / "fine" / "deblur_mlp.py").exists())
        self.assertIn("GTnet", deblur_source)
        self.assertIn("FourierEmbedding", deblur_source)
        self.assertIn("position_delta", deblur_source)
        self.assertIn("deblur_warmup_iters", trainer_source)
        self.assertIn("deblur_transform_regularization", deblur_source)

    def test_edgs_option_is_rejected_before_training(self) -> None:
        runner_source = (BACKEND_ROOT / "app" / "fine" / "runner.py").read_text(encoding="utf-8")

        self.assertIn("fine_edgs_enabled", runner_source)
        self.assertIn("EDGS/RoMA dense initialization has been removed", runner_source)

    def test_deblur_defaults_to_enabled_mixed_path(self) -> None:
        *_, deblur_mlp_enabled_by_default, _ = import_fine_runtime()

        self.assertTrue(deblur_mlp_enabled_by_default("motion", {}))
        self.assertTrue(deblur_mlp_enabled_by_default("defocus", {}))
        self.assertTrue(deblur_mlp_enabled_by_default("mixed", {}))
        self.assertTrue(deblur_mlp_enabled_by_default("mixed", {"fine_deblur_enabled": "auto"}))
        self.assertFalse(deblur_mlp_enabled_by_default("motion", {"fine_deblur_enabled": "false"}))
        self.assertFalse(deblur_mlp_enabled_by_default("sharp", {"fine_deblur_enabled": "true"}))

    def test_resolve_fine_deblur_mode_defaults_to_mixed(self) -> None:
        try:
            from app.fine.runner import resolve_fine_deblur_mode
        except Exception as exc:
            raise unittest.SkipTest(f"fine runner import unavailable: {exc}") from exc

        registry = {"000000.jpg": {"blurred": True, "rejected": False}}

        self.assertEqual(resolve_fine_deblur_mode({}, "defocus", registry), ("mixed", "default_mixed"))
        self.assertEqual(resolve_fine_deblur_mode({}, "motion", registry), ("mixed", "default_mixed"))
        self.assertEqual(resolve_fine_deblur_mode({}, "sharp", {}), ("mixed", "default_mixed"))
        self.assertEqual(resolve_fine_deblur_mode({"fine_deblur_mode": "mixed"}, "defocus", registry), ("mixed", "override"))

    def test_far_noise_filter_removes_indoor_outlier_more_than_outdoor(self) -> None:
        try:
            import numpy as np
            from app.fine.viewer_meta import write_far_noise_filtered_ply, read_binary_little_endian_ply_layout
        except Exception as exc:
            raise unittest.SkipTest(f"PLY filtering dependencies unavailable: {exc}") from exc

        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("opacity", "<f4"),
                ("scale_0", "<f4"),
                ("scale_1", "<f4"),
                ("scale_2", "<f4"),
            ]
        )
        points = np.zeros(101, dtype=dtype)
        points["x"][:100] = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        points["x"][100] = 1.35
        points["opacity"] = 10.0
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 101\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float opacity\n"
            "property float scale_0\n"
            "property float scale_1\n"
            "property float scale_2\n"
            "end_header\n"
        ).encode("ascii")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "final.ply"
            indoor = root / "indoor.ply"
            outdoor = root / "outdoor.ply"
            source.write_bytes(header + points.tobytes())

            indoor_metrics = write_far_noise_filtered_ply(source, indoor, profile="indoor_full")
            outdoor_metrics = write_far_noise_filtered_ply(source, outdoor, profile="outdoor_fast_clean")

            indoor_count, _, _ = read_binary_little_endian_ply_layout(indoor)
            outdoor_count, _, _ = read_binary_little_endian_ply_layout(outdoor)
            self.assertLess(indoor_count, outdoor_count)
            self.assertEqual(indoor_metrics["far_noise_removed_points"], 1)
            self.assertEqual(outdoor_metrics["far_noise_removed_points"], 0)

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
