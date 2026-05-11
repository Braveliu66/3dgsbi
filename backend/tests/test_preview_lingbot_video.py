from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.algorithms import ALGORITHMS, normalize_preview_pipeline  # noqa: E402
from app.preview.types import PreviewContext  # noqa: E402


class LingBotVideoPreviewTests(unittest.TestCase):
    def test_video_preview_defaults_to_lingbot_and_image_preview_stays_litevggt(self) -> None:
        self.assertEqual(normalize_preview_pipeline(None, "video"), "lingbot_map_spz")
        self.assertEqual(normalize_preview_pipeline("", "video"), "lingbot_map_spz")
        self.assertEqual(normalize_preview_pipeline("lingbot", "video"), "lingbot_map_spz")
        self.assertEqual(normalize_preview_pipeline(None, "images"), "litevggt_spz")

    def test_lingbot_algorithm_registration_excludes_heavy_render_dependencies(self) -> None:
        entry = next(item for item in ALGORITHMS if item["name"] == "LingBot-Map Video Preview")

        self.assertEqual(entry["repo_url"], "https://github.com/Robbyant/lingbot-map")
        self.assertEqual(entry["license"], "Apache-2.0")
        self.assertEqual(entry["commit_hash_setting"], "lingbot_map_repo_commit")
        self.assertEqual(entry["weight_paths"], ["lingbot/lingbot-map-long.pt"])
        self.assertEqual(entry["source_type"], "pinned_runtime_package")
        self.assertEqual(entry["commands"], {})
        self.assertIn("excludes optional rendering", entry["license_notice"])

    def test_lingbot_adapter_mock_output_becomes_preview_artifacts(self) -> None:
        from app.preview.adapters import lingbot

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            video_path = input_dir / "clip.mp4"
            video_path.write_bytes(b"video")
            weight = root / "model-cache" / "lingbot" / "lingbot-map-long.pt"
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"weight")
            ctx = PreviewContext(
                task_id="task",
                project_id="project",
                pipeline="lingbot_map_spz",
                input_dir=input_dir,
                work_dir=root / "work",
                output_spz=root / "work" / "preview.spz",
                model_cache_dir=root / "model-cache",
                source_version=3,
                options={},
                progress=lambda *_: None,
            )

            def fake_runtime(**kwargs):
                self.assertEqual(kwargs["fps"], 2)
                self.assertEqual(kwargs["max_frames"], 128)
                self.assertEqual(kwargs["image_size"], 518)
                self.assertEqual(kwargs["mode"], "auto")
                self.assertEqual(kwargs["camera_iterations"], 1)
                self.assertIsNone(kwargs["keyframe_interval"])
                self.assertEqual(kwargs["window_size"], 64)
                self.assertEqual(kwargs["overlap_keyframes"], 4)
                self.assertEqual(kwargs["num_scale_frames"], 2)
                self.assertEqual(kwargs["max_points"], 2_000_000)
                self.assertEqual(kwargs["frame_stride"], 2)
                self.assertEqual(kwargs["pixel_stride"], 6)
                self.assertEqual(kwargs["conf_percentile"], 10.0)
                self.assertEqual(kwargs["min_conf"], 1e-5)
                self.assertFalse(kwargs["compile_model"])
                self.assertFalse(kwargs["save_predictions"])
                self.assertTrue(kwargs["keyframes_only_points"])
                self.assertFalse(kwargs["allow_sdpa_fallback"])
                self.assertEqual(kwargs["min_inference_fps"], 3.0)
                kwargs["output_ply"].parent.mkdir(parents=True, exist_ok=True)
                kwargs["output_ply"].write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
                return {
                    "adapter": "lingbot_map_spz",
                    "lingbot_model": "lingbot-map-long.pt",
                    "lingbot_sampled_frames": 8,
                    "lingbot_inference_mode": "streaming",
                    "lingbot_keyframe_interval": 1,
                    "lingbot_inference_fps": 3.25,
                    "point_count": 42,
                    "cuda_memory_peak_mb": 123.0,
                }

            with patch.object(lingbot, "run_lingbot_video_preview", side_effect=fake_runtime), patch.object(
                lingbot,
                "convert_ply_to_spz",
                side_effect=lambda _ply, spz: (spz.parent.mkdir(parents=True, exist_ok=True), spz.write_bytes(b"spz"), 11)[2],
            ):
                result = lingbot.run(ctx)

            self.assertEqual(result.output_spz.read_bytes(), b"spz")
            self.assertTrue(result.intermediate_ply and result.intermediate_ply.exists())
            self.assertEqual(result.splat_count, 11)
            self.assertEqual(result.metrics["adapter"], "lingbot_map_spz")
            self.assertEqual(result.metrics["lingbot_sampled_frames"], 8)
            self.assertEqual(result.metrics["lingbot_inference_fps"], 3.25)
            self.assertEqual(result.source_commits["LingBot-Map"], "4cd986009b9adeded8a4e740919221940dedeffe")

    def test_lingbot_checkpoint_pos_embed_infers_model_image_size(self) -> None:
        from app.preview.vendor.lingbot_runtime import infer_lingbot_model_image_size_from_state_dict

        long_checkpoint = {"aggregator.patch_embed.pos_embed": np.zeros((1, 1370, 1024), dtype=np.float32)}
        small_checkpoint = {"aggregator.patch_embed.pos_embed": np.zeros((1, 1025, 1024), dtype=np.float32)}

        self.assertEqual(infer_lingbot_model_image_size_from_state_dict(long_checkpoint), 518)
        self.assertEqual(infer_lingbot_model_image_size_from_state_dict(small_checkpoint), 448)
        self.assertEqual(infer_lingbot_model_image_size_from_state_dict({}), 518)

    def test_lingbot_flashinfer_missing_requires_explicit_sdpa_fallback(self) -> None:
        from app.preview.types import PreviewFailure
        from app.preview.vendor.lingbot_runtime import resolve_lingbot_attention_backend

        with self.assertRaises(PreviewFailure) as raised:
            resolve_lingbot_attention_backend(allow_sdpa_fallback=False, flashinfer_probe=lambda: False)

        self.assertEqual(raised.exception.code, "LINGBOT_FLASHINFER_UNAVAILABLE")
        use_sdpa, flashinfer_found = resolve_lingbot_attention_backend(
            allow_sdpa_fallback=True,
            flashinfer_probe=lambda: False,
        )
        self.assertTrue(use_sdpa)
        self.assertFalse(flashinfer_found)

    def test_lingbot_oom_fallback_profile_is_smaller_and_keyframed(self) -> None:
        from app.preview.vendor.lingbot_runtime import (
            LingBotInferenceProfile,
            make_lingbot_oom_fallback_profile,
            resolve_kv_cache_sliding_window,
        )

        self.assertEqual(resolve_kv_cache_sliding_window(64), 16)
        self.assertEqual(resolve_kv_cache_sliding_window(8), 8)

        profile = LingBotInferenceProfile(
            image_size=518,
            max_frames=128,
            mode="auto",
            keyframe_interval=None,
            camera_iterations=4,
            num_scale_frames=8,
            window_size=64,
            kv_cache_sliding_window=16,
            overlap_keyframes=8,
        )

        fallback = make_lingbot_oom_fallback_profile(profile)

        self.assertEqual(fallback.image_size, 448)
        self.assertEqual(fallback.max_frames, 96)
        self.assertEqual(fallback.keyframe_interval, 2)
        self.assertEqual(fallback.camera_iterations, 1)
        self.assertEqual(fallback.num_scale_frames, 2)
        self.assertEqual(fallback.window_size, 32)
        self.assertEqual(fallback.kv_cache_sliding_window, 8)
        self.assertEqual(fallback.overlap_keyframes, 4)
        self.assertTrue(fallback.oom_fallback)

    def test_lingbot_inference_fps_guard_requires_three_fps(self) -> None:
        from app.preview.types import PreviewFailure
        from app.preview.vendor.lingbot_runtime import validate_lingbot_inference_fps

        validate_lingbot_inference_fps(3.0, 3.0)
        with self.assertRaises(PreviewFailure) as raised:
            validate_lingbot_inference_fps(2.99, 3.0)
        self.assertEqual(raised.exception.code, "LINGBOT_INFERENCE_FPS_BELOW_TARGET")

    def test_lingbot_arrays_to_gaussian_splat_ply_stays_in_memory(self) -> None:
        from app.preview.vendor import lingbot_runtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = {
                "world_points_from_depth": np.array(
                    [
                        [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]],
                        [[[5, 5, 5], [6, 6, 6]], [[7, 7, 7], [8, 8, 8]]],
                    ],
                    dtype=np.float32,
                ),
                "images": np.array(
                    [
                        [[[10, 20], [30, 40]], [[50, 60], [70, 80]], [[90, 100], [110, 120]]],
                        [[[11, 21], [31, 41]], [[51, 61], [71, 81]], [[91, 101], [111, 121]]],
                    ],
                    dtype=np.uint8,
                ),
                "depth_conf": np.ones((2, 2, 2), dtype=np.float32),
                "is_keyframe": np.array([True, False], dtype=np.bool_),
            }
            output_ply = root / "preview.ply"

            with patch.object(
                lingbot_runtime,
                "save_predictions_npz",
                side_effect=AssertionError("unexpected NPZ write"),
            ):
                metrics = lingbot_runtime.write_spark_plain_ply_from_arrays(
                    predictions,
                    output_ply,
                    frame_stride=1,
                    pixel_stride=1,
                    conf_percentile=0,
                    min_conf=0,
                    max_points=0,
                    keyframes_only_points=True,
                )

            payload = output_ply.read_bytes()
            header, body = payload.split(b"end_header\n", 1)
            self.assertIn(b"element vertex 4", header)
            self.assertIn(b"property float f_dc_0", header)
            self.assertIn(b"property float opacity", header)
            self.assertIn(b"property float scale_0", header)
            self.assertIn(b"property float rot_0", header)
            self.assertFalse((root / "_predictions_for_ply").exists())
            self.assertEqual(metrics["point_count"], 4)
            self.assertEqual(metrics["lingbot_ply_format"], "gaussian_splat")
            self.assertEqual(metrics["lingbot_point_frame_count"], 1)

            records = np.frombuffer(
                body,
                dtype=lingbot_runtime.gaussian_splat_record_dtype(),
            )
            np.testing.assert_allclose(records["x"], np.array([1, 2, 3, 4], dtype=np.float32))
            np.testing.assert_allclose(records["rot_0"], np.ones(4, dtype=np.float32))

    def test_lingbot_npz_to_gaussian_splat_ply_prefers_depth_points(self) -> None:
        from app.preview.vendor.lingbot_runtime import gaussian_splat_record_dtype, write_spark_plain_ply_from_npz

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions"
            predictions.mkdir()
            world_points_from_depth = np.array(
                [
                    [[1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4]],
                ],
                dtype=np.float32,
            )
            world_points = np.full((2, 2, 3), 99, dtype=np.float32)
            image = np.array(
                [
                    [[10, 20], [30, 40]],
                    [[50, 60], [70, 80]],
                    [[90, 100], [110, 120]],
                ],
                dtype=np.uint8,
            )
            depth_conf = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
            np.savez(
                predictions / "frame_000000.npz",
                world_points_from_depth=world_points_from_depth,
                world_points=world_points,
                images=image,
                depth_conf=depth_conf,
            )

            output_ply = root / "preview.ply"
            metrics = write_spark_plain_ply_from_npz(
                predictions,
                output_ply,
                frame_stride=1,
                pixel_stride=1,
                conf_percentile=0,
                min_conf=0.25,
                max_points=0,
            )

            payload = output_ply.read_bytes()
            header, body = payload.split(b"end_header\n", 1)
            self.assertIn(b"element vertex 2", header)
            self.assertIn(b"property float f_dc_2", header)
            self.assertIn(b"property float scale_2", header)
            self.assertIn(b"property float rot_3", header)
            self.assertEqual(metrics["point_count"], 2)
            self.assertEqual(metrics["lingbot_ply_format"], "gaussian_splat")
            self.assertEqual(metrics["lingbot_point_source"], "world_points_from_depth")

            records = np.frombuffer(
                body,
                dtype=gaussian_splat_record_dtype(),
            )
            self.assertEqual(records.shape[0], 2)
            np.testing.assert_allclose(records["x"], np.array([3, 4], dtype=np.float32))
            np.testing.assert_allclose(records["rot_0"], np.ones(2, dtype=np.float32))

    def test_lingbot_detects_torch_compile_cudagraph_overwrite(self) -> None:
        from app.preview.vendor.lingbot_runtime import is_cudagraph_overwrite_error

        error = RuntimeError("accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run")
        self.assertTrue(is_cudagraph_overwrite_error(error))
        self.assertFalse(is_cudagraph_overwrite_error(RuntimeError("out of memory")))


if __name__ == "__main__":
    unittest.main()
