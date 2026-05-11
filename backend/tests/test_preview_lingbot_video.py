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
                self.assertEqual(kwargs["fps"], 3)
                self.assertEqual(kwargs["max_frames"], 0)
                self.assertEqual(kwargs["mode"], "windowed")
                self.assertEqual(kwargs["camera_iterations"], 1)
                self.assertEqual(kwargs["keyframe_interval"], 6)
                self.assertEqual(kwargs["window_size"], 64)
                self.assertEqual(kwargs["overlap_keyframes"], 4)
                self.assertEqual(kwargs["num_scale_frames"], 2)
                self.assertEqual(kwargs["max_points"], 0)
                self.assertEqual(kwargs["frame_stride"], 1)
                self.assertEqual(kwargs["pixel_stride"], 4)
                self.assertEqual(kwargs["conf_percentile"], 5.0)
                self.assertEqual(kwargs["min_conf"], 1e-5)
                self.assertTrue(kwargs["compile_model"])
                self.assertTrue(kwargs["save_predictions"])
                kwargs["output_ply"].parent.mkdir(parents=True, exist_ok=True)
                kwargs["output_ply"].write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
                return {
                    "adapter": "lingbot_map_spz",
                    "lingbot_model": "lingbot-map-long.pt",
                    "lingbot_sampled_frames": 8,
                    "lingbot_inference_mode": "streaming",
                    "lingbot_keyframe_interval": 1,
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
            self.assertEqual(result.source_commits["LingBot-Map"], "4cd986009b9adeded8a4e740919221940dedeffe")

    def test_lingbot_npz_to_spark_plain_ply_prefers_depth_points(self) -> None:
        from app.preview.vendor.lingbot_runtime import write_spark_plain_ply_from_npz

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
            self.assertIn(b"property uchar blue", header)
            self.assertNotIn(b"property uchar alpha", header)
            self.assertEqual(metrics["point_count"], 2)
            self.assertEqual(metrics["lingbot_point_source"], "world_points_from_depth")

            records = np.frombuffer(
                body,
                dtype=np.dtype(
                    [
                        ("x", "<f4"),
                        ("y", "<f4"),
                        ("z", "<f4"),
                        ("red", "u1"),
                        ("green", "u1"),
                        ("blue", "u1"),
                    ]
                ),
            )
            self.assertEqual(records.shape[0], 2)
            np.testing.assert_allclose(records["x"], np.array([3, 4], dtype=np.float32))
            np.testing.assert_array_equal(records["red"], np.array([30, 40], dtype=np.uint8))
            np.testing.assert_array_equal(records["green"], np.array([70, 80], dtype=np.uint8))
            np.testing.assert_array_equal(records["blue"], np.array([110, 120], dtype=np.uint8))

    def test_lingbot_detects_torch_compile_cudagraph_overwrite(self) -> None:
        from app.preview.vendor.lingbot_runtime import is_cudagraph_overwrite_error

        error = RuntimeError("accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run")
        self.assertTrue(is_cudagraph_overwrite_error(error))
        self.assertFalse(is_cudagraph_overwrite_error(RuntimeError("out of memory")))


if __name__ == "__main__":
    unittest.main()
