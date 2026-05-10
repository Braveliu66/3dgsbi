from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.preview.adapters import lingbot  # noqa: E402
from app.preview.types import PreviewContext  # noqa: E402


class FakeCuda:
    @staticmethod
    def mem_get_info():
        return 20 * 1024 * 1024 * 1024, 24 * 1024 * 1024 * 1024

    @staticmethod
    def get_device_capability():
        return (8, 0)


class FakeTorch:
    cuda = FakeCuda()
    compile = staticmethod(lambda value, mode=None: value)


class LingBotAdapterTests(unittest.TestCase):
    def test_creates_output_parent_before_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weight = root / "models" / "lingbot-map" / "lingbot-map-long.pt"
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"weight")
            output_spz = root / "work" / "preview.spz"

            def fake_runtime(**kwargs):
                output_ply = kwargs["output_ply"]
                self.assertTrue(output_ply.parent.is_dir())
                self.assertEqual(kwargs["fps"], 5)
                self.assertEqual(kwargs["max_frames"], 0)
                self.assertEqual(kwargs["confidence_quantile"], 0.8)
                self.assertEqual(kwargs["max_points"], 0)
                self.assertEqual(kwargs["runtime_options"]["lingbot_input_mode"], "offline_video")
                self.assertEqual(kwargs["runtime_options"]["lingbot_compile_cache_dir"], str(root / "models" / "torchinductor"))
                output_ply.write_bytes(b"ply")
                return {"input_frame_count": 2, "point_count": 1}

            def fake_convert(ply_path: Path, spz_path: Path) -> int:
                self.assertEqual(ply_path.name, "recon.ply")
                spz_path.parent.mkdir(parents=True, exist_ok=True)
                spz_path.write_bytes(b"spz")
                return 1

            ctx = PreviewContext(
                task_id="task",
                project_id="project",
                pipeline="lingbot_spz",
                input_dir=root / "input",
                input_video=root / "input.mp4",
                work_dir=root / "work",
                output_spz=output_spz,
                model_cache_dir=root / "models",
                source_version=1,
                options={},
                progress=lambda *_args, **_kwargs: None,
            )

            with (
                patch.object(lingbot, "load_lingbot_runtime", return_value=fake_runtime),
                patch.object(lingbot, "convert_ply_to_spz", side_effect=fake_convert),
            ):
                result = lingbot.run(ctx)

            self.assertEqual(result.output_spz, output_spz)
            self.assertEqual(result.intermediate_ply.name, "recon.ply")
            self.assertEqual(result.splat_count, 1)
            self.assertTrue(output_spz.exists())
            self.assertEqual(result.metrics["output_spz_size"], 3)

    def test_video_plan_defaults_for_fast_preview(self) -> None:
        from app.preview.vendor import lingbot_runtime

        with patch.object(lingbot_runtime, "flashinfer_available", return_value=True):
            plan = lingbot_runtime.resolve_lingbot_plan(
                torch=FakeTorch(),
                frame_count=64,
                original_frame_count=64,
                frame_budget=0,
                input_mode="offline_video",
                runtime_options={},
            )

        self.assertEqual(plan.backend, "flashinfer")
        self.assertEqual(plan.mode, "streaming")
        self.assertTrue(plan.compile)
        self.assertEqual(plan.camera_num_iterations, 1)
        self.assertEqual(plan.keyframe_interval, 8)
        self.assertEqual(plan.num_scale_frames, 2)
        self.assertIsNone(plan.overlap_keyframes)
        self.assertTrue(plan.keyframes_only_points)

    def test_long_video_plan_uses_windowed_profile(self) -> None:
        from app.preview.vendor import lingbot_runtime

        with patch.object(lingbot_runtime, "flashinfer_available", return_value=True):
            plan = lingbot_runtime.resolve_lingbot_plan(
                torch=FakeTorch(),
                frame_count=501,
                original_frame_count=501,
                frame_budget=0,
                input_mode="offline_video",
                runtime_options={},
            )

        self.assertEqual(plan.mode, "windowed")
        self.assertEqual(plan.overlap_keyframes, 8)
        self.assertTrue(plan.keyframes_only_points)

    def test_short_video_compile_is_requested_but_not_effective(self) -> None:
        from app.preview.vendor import lingbot_runtime

        with patch.object(lingbot_runtime, "flashinfer_available", return_value=True):
            plan = lingbot_runtime.resolve_lingbot_plan(
                torch=FakeTorch(),
                frame_count=64,
                original_frame_count=64,
                frame_budget=0,
                input_mode="offline_video",
                runtime_options={},
            )

        self.assertTrue(plan.compile)
        self.assertFalse(lingbot_runtime.should_compile_lingbot_plan(plan))

    def test_long_video_compile_warmup_is_effective(self) -> None:
        from app.preview.vendor import lingbot_runtime

        with patch.object(lingbot_runtime, "flashinfer_available", return_value=True):
            plan = lingbot_runtime.resolve_lingbot_plan(
                torch=FakeTorch(),
                frame_count=501,
                original_frame_count=501,
                frame_budget=0,
                input_mode="offline_video",
                runtime_options={},
            )

        self.assertTrue(plan.compile)
        self.assertTrue(lingbot_runtime.should_compile_lingbot_plan(plan))

    def test_camera_plan_defaults_for_low_latency(self) -> None:
        from app.preview.vendor import lingbot_runtime

        with patch.object(lingbot_runtime, "flashinfer_available", return_value=True):
            plan = lingbot_runtime.resolve_lingbot_plan(
                torch=FakeTorch(),
                frame_count=24,
                original_frame_count=24,
                frame_budget=96,
                input_mode="realtime_camera",
                runtime_options={},
            )

        self.assertEqual(plan.mode, "streaming")
        self.assertFalse(plan.compile)
        self.assertEqual(plan.camera_num_iterations, 1)
        self.assertEqual(plan.keyframe_interval, 8)
        self.assertEqual(plan.num_scale_frames, 2)
        self.assertIsNone(plan.overlap_keyframes)
        self.assertTrue(plan.keyframes_only_points)

    def test_cli_style_options_override_profile_defaults(self) -> None:
        from app.preview.vendor import lingbot_runtime

        plan = lingbot_runtime.resolve_lingbot_plan(
            torch=FakeTorch(),
            frame_count=64,
            original_frame_count=64,
            frame_budget=0,
            input_mode="offline_video",
            runtime_options={
                "backend": "sdpa",
                "compile": False,
                "camera_num_iterations": 2,
                "keyframe_interval": 5,
                "num_scale_frames": 3,
                "overlap_keyframes": 4,
                "keyframes_only_points": False,
            },
        )

        self.assertEqual(plan.backend, "sdpa")
        self.assertFalse(plan.compile)
        self.assertEqual(plan.camera_num_iterations, 2)
        self.assertEqual(plan.keyframe_interval, 5)
        self.assertEqual(plan.num_scale_frames, 3)
        self.assertEqual(plan.overlap_keyframes, 4)
        self.assertFalse(plan.keyframes_only_points)

    def test_lingbot_inference_reporter_emits_frame_and_window_metrics(self) -> None:
        from app.preview.vendor.lingbot_runtime import LingBotInferenceReporter

        calls = []

        def progress(stage, progress_value, message, metrics):
            calls.append((stage, progress_value, message, metrics))

        reporter = LingBotInferenceReporter(progress, min_interval_seconds=0, min_frame_step=1)
        reporter(
            {
                "type": "streaming_frame",
                "current_frame": 12,
                "total_frames": 24,
                "elapsed_seconds": 6.0,
                "seconds_per_frame": 0.5,
            }
        )
        reporter(
            {
                "type": "windowed_window",
                "current_window": 2,
                "total_windows": 4,
                "window_start": 32,
                "window_end": 64,
                "covered_frames": 64,
                "total_frames": 128,
                "elapsed_seconds": 10.0,
            }
        )

        self.assertEqual(calls[0][0], "lingbot_inference")
        self.assertEqual(calls[0][3]["lingbot_current_frame"], 12)
        self.assertEqual(calls[0][3]["lingbot_total_frames"], 24)
        self.assertEqual(calls[0][3]["lingbot_current_inference_fps"], 2.0)
        self.assertEqual(calls[1][3]["lingbot_current_window"], 2)
        self.assertEqual(calls[1][3]["lingbot_total_windows"], 4)

    def test_lingbot_inference_throughput_metrics(self) -> None:
        from app.preview.vendor.lingbot_runtime import inference_throughput_metrics

        metrics = inference_throughput_metrics(20, 4.0)

        self.assertEqual(metrics["processed_frames"], 20)
        self.assertEqual(metrics["inference_seconds"], 4.0)
        self.assertEqual(metrics["inference_fps"], 5.0)
        self.assertEqual(metrics["lingbot_inference_fps"], 5.0)

    def test_lingbot_configures_persistent_compile_cache(self) -> None:
        from app.preview.vendor.lingbot_runtime import configure_torch_compile_cache

        with patch.dict(os.environ, {}, clear=True):
            configure_torch_compile_cache({"lingbot_compile_cache_dir": "/model-cache/torchinductor"})

            self.assertEqual(os.environ["TORCHINDUCTOR_CACHE_DIR"], "/model-cache/torchinductor")
            self.assertEqual(os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"], "1")
            self.assertEqual(os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"], "1")

    def test_gaussian_splat_ply_has_supersplat_properties(self) -> None:
        from app.preview.io.ply import write_gaussian_splat_ply

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recon.ply"
            points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
            colors = np.array([[255, 0, 0], [0, 255, 128]], dtype=np.uint8)

            count = write_gaussian_splat_ply(path, points, colors)

            self.assertEqual(count, 2)
            data = path.read_bytes()
            header = data.split(b"end_header\n", 1)[0].decode("ascii")
            self.assertIn("property float f_dc_0", header)
            self.assertIn("property float f_rest_44", header)
            self.assertIn("property float opacity", header)
            self.assertIn("property float scale_0", header)
            self.assertIn("property float rot_3", header)
            self.assertNotIn("property uchar red", header)
            self.assertEqual(path.stat().st_size, len(header.encode("ascii")) + len(b"end_header\n") + 2 * 62 * 4)

    def test_point_export_removes_image_batch_dimension(self) -> None:
        from app.preview.vendor.lingbot_runtime import prepare_lingbot_point_export

        world = np.zeros((2, 3, 4, 3), dtype=np.float32)
        conf = np.ones((2, 3, 4), dtype=np.float32)
        images = np.ones((1, 2, 3, 3, 4), dtype=np.float32)

        export = prepare_lingbot_point_export({"world_points": world, "world_points_conf": conf, "images": images}, images)

        self.assertEqual(export.world_points.shape, (2, 3, 4, 3))
        self.assertEqual(export.confidence.shape, (2, 3, 4))
        self.assertEqual(export.colors.shape, (2, 3, 4, 3))

    def test_keyframes_only_point_export_keeps_keyframe_frames(self) -> None:
        from app.preview.vendor.lingbot_runtime import filter_keyframe_point_export, prepare_lingbot_point_export

        world = np.arange(12, dtype=np.float32).reshape(4, 1, 1, 3)
        conf = np.ones((4, 1, 1), dtype=np.float32)
        images = np.ones((4, 1, 1, 3), dtype=np.uint8)
        export = prepare_lingbot_point_export(
            {
                "world_points": world,
                "world_points_conf": conf,
                "images": images,
                "is_keyframe": np.array([[True, False, True, False]]),
            },
            images,
        )

        filtered = filter_keyframe_point_export(export, enabled=True)

        self.assertTrue(filtered.metrics["lingbot_keyframes_only_points_applied"])
        self.assertEqual(filtered.world_points.shape[0], 2)
        np.testing.assert_array_equal(filtered.world_points[:, 0, 0, 0], np.array([0, 6], dtype=np.float32))

    def test_keyframes_only_point_export_falls_back_without_mask(self) -> None:
        from app.preview.vendor.lingbot_runtime import filter_keyframe_point_export, prepare_lingbot_point_export

        world = np.zeros((3, 1, 1, 3), dtype=np.float32)
        conf = np.ones((3, 1, 1), dtype=np.float32)
        images = np.ones((3, 1, 1, 3), dtype=np.uint8)
        export = prepare_lingbot_point_export({"world_points": world, "world_points_conf": conf, "images": images}, images)

        filtered = filter_keyframe_point_export(export, enabled=True)

        self.assertFalse(filtered.metrics["lingbot_keyframes_only_points_applied"])
        self.assertEqual(filtered.metrics["lingbot_keyframes_only_points_fallback"], "missing_keyframe_mask")
        self.assertEqual(filtered.world_points.shape[0], 3)

    def test_point_export_reports_shape_mismatch(self) -> None:
        from app.preview.types import PreviewFailure
        from app.preview.vendor.lingbot_runtime import prepare_lingbot_point_export

        world = np.zeros((2, 3, 4, 3), dtype=np.float32)
        conf = np.ones((1, 3, 4), dtype=np.float32)
        images = np.ones((2, 3, 3, 4), dtype=np.float32)

        with self.assertRaises(PreviewFailure) as ctx:
            prepare_lingbot_point_export({"world_points": world, "world_points_conf": conf, "images": images}, images)

        self.assertEqual(ctx.exception.code, "LINGBOT_EXPORT_SHAPE_MISMATCH")


if __name__ == "__main__":
    unittest.main()
