from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
                self.assertEqual(kwargs["image_size"], 518)
                self.assertEqual(kwargs["target_width"], 518)
                self.assertEqual(kwargs["target_height"], 378)
                self.assertEqual(kwargs["mode"], "streaming")
                self.assertEqual(kwargs["preprocess_mode"], "crop")
                self.assertEqual(kwargs["camera_iterations"], 4)
                self.assertEqual(kwargs["keyframe_interval"], 6)
                self.assertEqual(kwargs["window_size"], 64)
                self.assertEqual(kwargs["overlap_size"], 16)
                self.assertEqual(kwargs["overlap_keyframes"], 8)
                self.assertEqual(kwargs["num_scale_frames"], 4)
                self.assertEqual(kwargs["max_points"], 2_000_000)
                self.assertEqual(kwargs["frame_stride"], 1)
                self.assertEqual(kwargs["pixel_stride"], 1)
                self.assertEqual(kwargs["conf_percentile"], 50.0)
                self.assertEqual(kwargs["min_conf"], 1e-5)
                self.assertFalse(kwargs["compile_model"])
                self.assertTrue(kwargs["save_predictions"])
                self.assertFalse(kwargs["keyframes_only_points"])
                self.assertFalse(kwargs["allow_sdpa_fallback"])
                self.assertEqual(kwargs["min_inference_fps"], 3.0)
                from app.preview.io.ply import write_point_cloud_ply

                write_point_cloud_ply(
                    np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                    np.array([[10, 20, 30]], dtype=np.uint8),
                    kwargs["output_points_ply"],
                    confidence=np.array([0.9], dtype=np.float32),
                    max_points=0,
                )
                kwargs["output_meta_json"].write_text("{}", encoding="utf-8")
                np.savez(
                    kwargs["output_official_predictions_npz"],
                    images=np.zeros((1, 3, 2, 2), dtype=np.float32),
                    depth=np.ones((1, 2, 2, 1), dtype=np.float32),
                    depth_conf=np.ones((1, 2, 2), dtype=np.float32),
                    extrinsic=np.zeros((1, 3, 4), dtype=np.float32),
                    intrinsic=np.eye(3, dtype=np.float32)[None],
                )
                return {
                    "adapter": "lingbot_map_spz",
                    "lingbot_model": "lingbot-map-long.pt",
                    "lingbot_sampled_frames": 8,
                    "lingbot_inference_mode": "streaming",
                    "lingbot_keyframe_interval": 1,
                    "lingbot_inference_fps": 3.25,
                    "lingbot_point_source": "world_points",
                    "lingbot_preview_point_radius": 0.001,
                    "point_count": 42,
                    "cuda_memory_peak_mb": 123.0,
                }

            with patch.object(lingbot, "run_lingbot_video_preview", side_effect=fake_runtime), patch.object(
                lingbot,
                "convert_ply_to_spz",
                side_effect=lambda ply, spz: (self.assertEqual(ply.name, "preview_splats.ply"), spz.parent.mkdir(parents=True, exist_ok=True), spz.write_bytes(b"spz"), 11)[3],
            ):
                result = lingbot.run(ctx)

            self.assertEqual(result.output_spz.read_bytes(), b"spz")
            self.assertTrue(result.intermediate_ply and result.intermediate_ply.exists())
            self.assertEqual(result.splat_count, 11)
            self.assertEqual(result.metrics["adapter"], "lingbot_map_spz")
            self.assertEqual(result.metrics["point_source"], "world_points")
            self.assertEqual(result.metrics["fixed_splat_ply_count"], 1)
            self.assertEqual(result.metrics["fixed_splat_base_point_radius"], 0.001)
            self.assertAlmostEqual(result.metrics["fixed_splat_point_radius"], 0.00011)
            self.assertEqual(result.metrics["fixed_splat_point_radius_scale"], 0.11)
            self.assertEqual(result.metrics["fixed_splat_opacity"], 0.55)
            self.assertEqual(result.metrics["lingbot_sampled_frames"], 8)
            self.assertEqual(result.metrics["lingbot_inference_fps"], 3.25)
            self.assertTrue(Path(result.metrics["lingbot_official_predictions_npz"]).exists())
            self.assertGreater(result.metrics["lingbot_official_predictions_npz_size"], 0)
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

    def test_lingbot_runtime_no_longer_exposes_experimental_fallback_helpers(self) -> None:
        from app.preview.vendor import lingbot_runtime

        self.assertFalse(hasattr(lingbot_runtime, "make_lingbot_oom_fallback_profile"))
        self.assertFalse(hasattr(lingbot_runtime, "make_lingbot_oom_fallback_profiles"))
        self.assertEqual(lingbot_runtime.resolve_kv_cache_sliding_window(64), 16)
        self.assertEqual(lingbot_runtime.resolve_kv_cache_sliding_window(8), 8)

    def test_lingbot_auto_mode_uses_keyframes_before_very_long_windowed(self) -> None:
        from app.preview.vendor.lingbot_runtime import resolve_keyframe_interval, resolve_mode

        self.assertEqual(resolve_mode("auto", 320), "streaming")
        self.assertEqual(resolve_mode("auto", 321), "streaming")
        self.assertEqual(resolve_mode("auto", 807), "streaming")
        self.assertEqual(resolve_mode("auto", 3000), "streaming")
        self.assertEqual(resolve_mode("auto", 3001), "windowed")
        self.assertEqual(resolve_keyframe_interval(None, "streaming", 800), 3)
        self.assertEqual(resolve_keyframe_interval(None, "streaming", 807), 3)
        self.assertEqual(resolve_keyframe_interval(None, "windowed", 800), 3)

    def test_lingbot_preprocess_auto_uses_official_crop(self) -> None:
        from app.preview.vendor.lingbot_runtime import resolve_preprocess_mode

        self.assertEqual(resolve_preprocess_mode("auto", 720, 1280), "crop")
        self.assertEqual(resolve_preprocess_mode("auto", 1280, 720), "crop")
        self.assertEqual(resolve_preprocess_mode("crop", 720, 1280), "crop")
        self.assertEqual(resolve_preprocess_mode("pad", 1280, 720), "pad")

    def test_lingbot_crop_preprocess_uses_official_loader(self) -> None:
        source = (BACKEND_ROOT / "app" / "preview" / "vendor" / "lingbot_runtime.py").read_text(encoding="utf-8")

        self.assertIn("images = load_and_preprocess_images(", source)
        self.assertNotIn("images = load_and_preprocess_images_to_target_box(", source)
        self.assertNotIn("enable_point=True", source)
        self.assertNotIn("enable_depth=True", source)

    def test_lingbot_windowed_inference_passes_overlap_size(self) -> None:
        from app.preview.vendor.lingbot_runtime import run_lingbot_inference

        class FakeCompiler:
            def cudagraph_mark_step_begin(self) -> None:
                return None

        class FakeTorch:
            compiler = FakeCompiler()

        class FakeModel:
            def __init__(self) -> None:
                self.kwargs = None

            def inference_windowed(self, images, **kwargs):
                self.kwargs = kwargs
                return {"ok": True}

            def inference_streaming(self, images, **kwargs):
                raise AssertionError("unexpected streaming inference")

        model = FakeModel()
        result = run_lingbot_inference(
            model,
            images=object(),
            resolved_mode="windowed",
            window_size=64,
            overlap_size=16,
            overlap_keyframes=8,
            num_scale_frames=4,
            keyframe_interval=6,
            output_device="cpu",
            torch_module=FakeTorch(),
        )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(model.kwargs["overlap_size"], 16)
        self.assertEqual(model.kwargs["overlap_keyframes"], 8)

    def test_lingbot_target_resolution_crops_portrait_before_resize(self) -> None:
        from app.preview.vendor.lingbot_runtime import resolve_lingbot_target_dimensions

        kwargs = {"target_width": 518, "target_height": 378, "patch_size": 14}

        self.assertEqual(resolve_lingbot_target_dimensions(1920, 1080, **kwargs), (518, 294))
        self.assertEqual(resolve_lingbot_target_dimensions(1600, 1200, **kwargs), (504, 378))
        self.assertEqual(resolve_lingbot_target_dimensions(720, 1280, **kwargs), (378, 378))

        square_kwargs = {"target_width": 518, "target_height": 518, "patch_size": 14}
        self.assertEqual(resolve_lingbot_target_dimensions(720, 1280, **square_kwargs), (518, 518))

    def test_lingbot_video_frame_extraction_uses_ffmpeg_autorotate(self) -> None:
        from app.preview.vendor import lingbot_runtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "phone.mov"
            video.write_bytes(b"video")
            output_dir = root / "frames"

            def fake_run(command, check):
                self.assertIn("-vf", command)
                self.assertIn("fps=10", command)
                self.assertNotIn("-noautorotate", command)
                pattern = Path(command[-1])
                pattern.parent.mkdir(parents=True, exist_ok=True)
                for index in range(5):
                    (pattern.parent / f"{index + 1:06d}.jpg").write_bytes(b"jpg")
                return SimpleNamespace(returncode=0)

            fake_cv2 = SimpleNamespace(imread=lambda _: np.zeros((1280, 720, 3), dtype=np.uint8))

            with patch.object(lingbot_runtime.shutil, "which", return_value="ffmpeg"), patch.object(
                lingbot_runtime.subprocess,
                "run",
                side_effect=fake_run,
            ), patch.dict(sys.modules, {"cv2": fake_cv2}):
                frames = lingbot_runtime.extract_video_frames(video, output_dir, fps=10, max_frames=3)

            self.assertEqual(frames.count, 3)
            self.assertIsNone(frames.source_fps)
            self.assertEqual(frames.sampled_fps, 10)
            self.assertEqual((frames.width, frames.height), (720, 1280))
            self.assertEqual([path.name for path in sorted(output_dir.glob("*.jpg"))], ["000000.jpg", "000001.jpg", "000002.jpg"])

    def test_lingbot_video_frame_extraction_can_keep_native_frame_rate(self) -> None:
        from app.preview.vendor import lingbot_runtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            output_dir = root / "frames"

            def fake_run(command, check):
                self.assertNotIn("-vf", command)
                pattern = Path(command[-1])
                pattern.parent.mkdir(parents=True, exist_ok=True)
                for index in range(3):
                    (pattern.parent / f"{index + 1:06d}.jpg").write_bytes(b"jpg")
                return SimpleNamespace(returncode=0)

            fake_cv2 = SimpleNamespace(imread=lambda _: np.zeros((1080, 1920, 3), dtype=np.uint8))

            with patch.object(lingbot_runtime.shutil, "which", return_value="ffmpeg"), patch.object(
                lingbot_runtime.subprocess,
                "run",
                side_effect=fake_run,
            ), patch.dict(sys.modules, {"cv2": fake_cv2}):
                frames = lingbot_runtime.extract_video_frames(video, output_dir, fps=0, max_frames=0)

            self.assertEqual(frames.count, 3)
            self.assertEqual(frames.sampled_fps, 0)
            self.assertEqual((frames.width, frames.height), (1920, 1080))

    def test_lingbot_inference_fps_guard_is_non_blocking(self) -> None:
        from app.preview.vendor.lingbot_runtime import validate_lingbot_inference_fps

        validate_lingbot_inference_fps(3.0, 3.0)
        validate_lingbot_inference_fps(2.99, 3.0)

    def test_lingbot_arrays_to_pointcloud_ply_stays_in_memory(self) -> None:
        from app.preview.vendor import lingbot_runtime
        from app.preview.io.ply import POINT_CLOUD_PLY_DTYPE

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
            self.assertIn(b"property uchar red", header)
            self.assertIn(b"property uchar green", header)
            self.assertIn(b"property uchar blue", header)
            self.assertNotIn(b"property float confidence", header)
            self.assertNotIn(b"property float f_dc_0", header)
            self.assertFalse((root / "_predictions_for_ply").exists())
            self.assertEqual(metrics["point_count"], 4)
            self.assertEqual(metrics["lingbot_ply_format"], "rgb_point_cloud")
            self.assertEqual(metrics["lingbot_point_frame_count"], 1)
            self.assertEqual(metrics["lingbot_point_source_frames"], 2)
            self.assertEqual(metrics["lingbot_point_skipped_frames"], 1)

            records = np.frombuffer(
                body,
                dtype=POINT_CLOUD_PLY_DTYPE,
            )
            np.testing.assert_allclose(records["x"], np.array([1, 2, 3, 4], dtype=np.float32))

    def test_lingbot_official_predictions_npz_preserves_viewer_arrays(self) -> None:
        from app.preview.vendor.lingbot_runtime import save_official_predictions_npz

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "official_predictions.npz"
            save_official_predictions_npz(
                {
                    "images": np.zeros((1, 3, 2, 2), dtype=np.float32),
                    "depth": np.ones((1, 2, 2, 1), dtype=np.float32),
                    "extrinsic": np.zeros((1, 3, 4), dtype=np.float32),
                    "ignore": "not an array",
                },
                output_path,
            )

            with np.load(output_path, allow_pickle=False) as data:
                self.assertEqual(set(data.files), {"images", "depth", "extrinsic"})
                self.assertEqual(data["images"].shape, (1, 3, 2, 2))

    def test_lingbot_depth_reprojection_synthesizes_confidence(self) -> None:
        from app.preview.vendor.lingbot_runtime import attach_depth_world_points

        predictions = {
            "depth": np.ones((1, 2, 2, 1), dtype=np.float32),
            "extrinsic_w2c": np.eye(4, dtype=np.float32)[None, :3, :4],
            "intrinsic": np.eye(3, dtype=np.float32)[None],
        }

        attach_depth_world_points(
            predictions,
            unproject_depth_map_to_point_map=lambda depth, _extrinsic, _intrinsic: np.zeros((*depth.shape[:-1], 3), dtype=np.float32),
        )

        self.assertEqual(predictions["world_points_from_depth"].shape, (1, 2, 2, 3))
        self.assertEqual(predictions["depth_conf"].shape, (1, 2, 2))
        np.testing.assert_allclose(predictions["depth_conf"], 1.0)

    def test_lingbot_depth_reprojection_passes_w2c_extrinsic_to_official_unproject(self) -> None:
        from app.preview.vendor.lingbot_runtime import attach_depth_world_points

        extrinsic_w2c = np.eye(4, dtype=np.float32)
        extrinsic_w2c[:3, 3] = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        predictions = {
            "depth": np.ones((1, 1, 1, 1), dtype=np.float32),
            "extrinsic": np.zeros((1, 3, 4), dtype=np.float32),
            "extrinsic_w2c": extrinsic_w2c[None, :3, :4],
            "intrinsic": np.eye(3, dtype=np.float32)[None],
        }

        captured = {}

        def fake_unproject(depth, extrinsic, intrinsic):
            captured["depth"] = depth
            captured["extrinsic"] = extrinsic
            captured["intrinsic"] = intrinsic
            return np.full((1, 1, 1, 3), 7.0, dtype=np.float32)

        attach_depth_world_points(
            predictions,
            unproject_depth_map_to_point_map=fake_unproject,
        )

        np.testing.assert_allclose(captured["extrinsic"], predictions["extrinsic_w2c"])
        np.testing.assert_allclose(predictions["world_points_from_depth"], 7.0)

    def test_lingbot_depth_reprojection_skips_when_world_points_exist(self) -> None:
        from app.preview.vendor.lingbot_runtime import attach_depth_world_points

        predictions = {
            "world_points": np.zeros((1, 1, 1, 3), dtype=np.float32),
            "depth": np.ones((1, 1, 1, 1), dtype=np.float32),
            "intrinsic": np.eye(3, dtype=np.float32)[None],
        }

        attach_depth_world_points(
            predictions,
            unproject_depth_map_to_point_map=lambda *_: (_ for _ in ()).throw(AssertionError("unexpected depth fallback")),
        )

        self.assertNotIn("world_points_from_depth", predictions)

    def test_lingbot_depth_reprojection_requires_w2c_extrinsic(self) -> None:
        from app.preview.types import PreviewFailure
        from app.preview.vendor.lingbot_runtime import attach_depth_world_points

        predictions = {
            "depth": np.ones((1, 1, 1, 1), dtype=np.float32),
            "extrinsic": np.eye(4, dtype=np.float32)[None, :3, :4],
            "intrinsic": np.eye(3, dtype=np.float32)[None],
        }

        with self.assertRaises(PreviewFailure) as raised:
            attach_depth_world_points(
                predictions,
                unproject_depth_map_to_point_map=lambda *_: np.zeros((1, 1, 1, 3), dtype=np.float32),
            )

        self.assertEqual(raised.exception.code, "LINGBOT_EXTRINSIC_W2C_MISSING")

    def test_lingbot_visualization_preserves_w2c_and_uses_c2w_for_viewer(self) -> None:
        from app.preview.vendor.lingbot_runtime import predictions_to_visualization_np

        class FakeTensor:
            def __init__(self, array):
                self.array = np.asarray(array, dtype=np.float32)
                self.shape = self.array.shape
                self.device = "cpu"
                self.dtype = self.array.dtype

            def __getitem__(self, key):
                return self.array[key]

            def __setitem__(self, key, value):
                self.array[key] = value.array if isinstance(value, FakeTensor) else value

            def detach(self):
                return self

            def to(self, _device):
                return self

            def is_floating_point(self):
                return True

            def float(self):
                return self

            def numpy(self):
                return self.array

        class FakeTorch:
            Tensor = FakeTensor

            @staticmethod
            def zeros(shape, *, device, dtype):
                return FakeTensor(np.zeros(shape, dtype=dtype))

        extrinsic_w2c = FakeTensor(np.array([[[1.0, 0.0, 0.0, -2.0], [0.0, 1.0, 0.0, -3.0], [0.0, 0.0, 1.0, -4.0]]], dtype=np.float32))
        intrinsic = FakeTensor(np.eye(3, dtype=np.float32)[None])

        def fake_pose_encoding_to_extri_intri(_pose_enc, _image_shape):
            return extrinsic_w2c, intrinsic

        def fake_inverse(matrix):
            inverse = np.array(matrix.array, copy=True)
            inverse[..., :3, 3] *= -1.0
            return FakeTensor(inverse)

        visualized = predictions_to_visualization_np(
            {"pose_enc": FakeTensor(np.zeros((1, 1, 1), dtype=np.float32))},
            FakeTensor(np.zeros((1, 3, 2, 2), dtype=np.float32)),
            pose_encoding_to_extri_intri=fake_pose_encoding_to_extri_intri,
            closed_form_inverse_se3_general=fake_inverse,
            torch_module=FakeTorch,
        )

        np.testing.assert_allclose(visualized["extrinsic_w2c"], extrinsic_w2c.array)
        np.testing.assert_allclose(visualized["extrinsic"][..., 3], np.array([[2.0, 3.0, 4.0]], dtype=np.float32))

    def test_lingbot_camera_view_uses_c2w_position(self) -> None:
        from app.preview.vendor.lingbot_runtime import lingbot_camera_view_from_frame

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        view = lingbot_camera_view_from_frame(
            {
                "extrinsic": c2w[:3, :4],
                "intrinsic": np.eye(3, dtype=np.float32),
                "images": np.zeros((3, 10, 20), dtype=np.float32),
            },
            radius_hint=2.0,
        )

        self.assertIsNotNone(view)
        assert view is not None
        np.testing.assert_allclose(view["position"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(view["target"], [1.0, 2.0, 5.0])

    def test_lingbot_official_viewer_tool_uses_upstream_pointcloud_viewer(self) -> None:
        source = (BACKEND_ROOT / "app" / "preview" / "tools" / "lingbot_official_pointcloud_viewer.py").read_text(encoding="utf-8")

        self.assertIn("from lingbot_map.vis import PointCloudViewer", source)
        self.assertIn("official_predictions.npz", source)

    def test_lingbot_stable_export_metrics_count_filtering_and_limit(self) -> None:
        from app.preview.vendor import lingbot_runtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = {
                "world_points_from_depth": np.array(
                    [
                        [[[0, 0, 0], [1, 0, 0]], [[2, 0, 0], [3, 0, 0]]],
                        [[[0, 2, 0], [1, 2, 0]], [[2, 2, 0], [3, 2, 0]]],
                    ],
                    dtype=np.float32,
                ),
                "images": np.full((2, 3, 2, 2), 255, dtype=np.uint8),
                "depth_conf": np.array(
                    [
                        [[0.1, 0.6], [0.7, 0.8]],
                        [[0.2, 0.3], [0.9, 1.0]],
                    ],
                    dtype=np.float32,
                ),
                "is_keyframe": np.array([True, False], dtype=np.bool_),
            }

            metrics = lingbot_runtime.write_spark_plain_ply_from_arrays(
                predictions,
                root / "preview.ply",
                frame_stride=1,
                pixel_stride=1,
                conf_percentile=0,
                min_conf=0.5,
                max_points=0,
                keyframes_only_points=False,
            )

            self.assertEqual(metrics["point_count_raw"], 8)
            self.assertEqual(metrics["lingbot_points_before_confidence_filter"], 8)
            self.assertEqual(metrics["lingbot_points_filtered_by_confidence"], 3)
            self.assertEqual(metrics["lingbot_points_after_confidence_filter"], 5)
            self.assertEqual(metrics["lingbot_points_before_downsample"], 5)
            self.assertEqual(metrics["lingbot_points_after_downsample"], 5)
            self.assertNotIn("lingbot_points_after_voxel", metrics)
            self.assertNotIn("lingbot_points_removed_by_voxel", metrics)
            self.assertEqual(metrics["point_count_exported"], 5)
            self.assertEqual(metrics["lingbot_point_source_frames"], 2)
            self.assertEqual(metrics["lingbot_point_frame_count"], 2)
            self.assertEqual(metrics["lingbot_point_skipped_frames"], 0)

    def test_lingbot_npz_to_pointcloud_ply_prefers_model_world_points(self) -> None:
        from app.preview.io.ply import POINT_CLOUD_PLY_DTYPE
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
                world_points_conf=depth_conf,
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
            self.assertNotIn(b"property float confidence", header)
            self.assertNotIn(b"property float scale_2", header)
            self.assertEqual(metrics["point_count"], 2)
            self.assertEqual(metrics["lingbot_ply_format"], "rgb_point_cloud")
            self.assertEqual(metrics["lingbot_point_source"], "world_points")

            records = np.frombuffer(
                body,
                dtype=POINT_CLOUD_PLY_DTYPE,
            )
            self.assertEqual(records.shape[0], 2)
            np.testing.assert_allclose(records["x"], np.array([99, 99], dtype=np.float32))

    def test_lingbot_depth_points_are_supported_as_export_source(self) -> None:
        from app.preview.io.ply import POINT_CLOUD_PLY_DTYPE
        from app.preview.vendor.lingbot_runtime import write_spark_plain_ply_from_npz

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions"
            predictions.mkdir()
            np.savez(
                predictions / "frame_000000.npz",
                world_points_from_depth=np.ones((2, 2, 3), dtype=np.float32),
                images=np.full((3, 2, 2), 255, dtype=np.uint8),
                depth_conf=np.ones((2, 2), dtype=np.float32),
            )

            output_ply = root / "preview_points.ply"
            metrics = write_spark_plain_ply_from_npz(
                predictions,
                output_ply,
                frame_stride=1,
                pixel_stride=1,
                conf_percentile=0,
                min_conf=0,
                max_points=0,
            )

            header, body = output_ply.read_bytes().split(b"end_header\n", 1)
            self.assertIn(b"element vertex 4", header)
            self.assertNotIn(b"property float confidence", header)
            self.assertEqual(metrics["point_count"], 4)
            self.assertEqual(metrics["lingbot_point_source"], "world_points_from_depth")
            self.assertTrue(metrics["lingbot_depth_reprojection_fallback"])
            records = np.frombuffer(body, dtype=POINT_CLOUD_PLY_DTYPE)
            self.assertEqual(records.shape[0], 4)

    def test_pointcloud_ply_converts_to_fixed_splat_ply_with_log_scale(self) -> None:
        from app.preview.io.ply import (
            FIXED_SPLAT_PLY_DTYPE,
            convert_pointcloud_ply_to_fixed_splat_ply,
            logit,
            write_point_cloud_ply,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            points_ply = root / "points.ply"
            splats_ply = root / "splats.ply"
            write_point_cloud_ply(
                np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]], dtype=np.float32),
                np.array([[0, 128, 255], [255, 128, 0]], dtype=np.uint8),
                points_ply,
                confidence=np.array([0.5, 0.75], dtype=np.float32),
                max_points=0,
                include_confidence=False,
            )

            count = convert_pointcloud_ply_to_fixed_splat_ply(
                points_ply,
                splats_ply,
                point_radius=0.00183,
                opacity=0.75,
            )

            header, body = splats_ply.read_bytes().split(b"end_header\n", 1)
            self.assertEqual(count, 2)
            self.assertIn(b"property float f_dc_0", header)
            self.assertIn(b"property float opacity", header)
            self.assertIn(b"property float scale_0", header)
            self.assertNotIn(b"property float f_rest_0", header)
            records = np.frombuffer(body, dtype=FIXED_SPLAT_PLY_DTYPE)
            np.testing.assert_allclose(records["scale_0"], np.full(2, np.log(0.00183), dtype=np.float32))
            np.testing.assert_allclose(records["opacity"], np.full(2, logit(0.75), dtype=np.float32))
            np.testing.assert_allclose(records["rot_0"], np.ones(2, dtype=np.float32))

    def test_lingbot_detects_torch_compile_cudagraph_overwrite(self) -> None:
        from app.preview.vendor.lingbot_runtime import is_cudagraph_overwrite_error

        error = RuntimeError("accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run")
        self.assertTrue(is_cudagraph_overwrite_error(error))
        self.assertFalse(is_cudagraph_overwrite_error(RuntimeError("out of memory")))

    def test_upload_page_passes_preview_meta_to_viewer(self) -> None:
        source = (BACKEND_ROOT.parent / "frontend" / "app" / "upload" / "page.tsx").read_text(encoding="utf-8")

        self.assertIn("viewerMetaUrl={viewer.viewer_meta_url ?? viewer.preview_meta_url}", source)

    def test_lingbot_preview_logs_diagnostic_fields(self) -> None:
        adapter_source = (BACKEND_ROOT / "app" / "preview" / "adapters" / "lingbot.py").read_text(encoding="utf-8")
        runtime_source = (BACKEND_ROOT / "app" / "preview" / "vendor" / "lingbot_runtime.py").read_text(encoding="utf-8")
        worker_source = (BACKEND_ROOT / "app" / "worker.py").read_text(encoding="utf-8")

        self.assertIn("[lingbot-preview] adapter params", adapter_source)
        self.assertIn("[lingbot-preview] pointcloud summary", adapter_source)
        self.assertIn("resolved inference", runtime_source)
        self.assertIn("export metrics", runtime_source)
        self.assertIn("video sampled", runtime_source)
        self.assertNotIn("preview_lingbot_shape_mode", adapter_source)
        self.assertNotIn("lingbot_shape_mode", runtime_source)
        self.assertNotIn("lingbot_cuda_allocator", runtime_source)
        self.assertNotIn("lingbot_inference_attempt_count", worker_source)

    def test_spark_spz_transcode_defaults_prioritize_clarity(self) -> None:
        source = (BACKEND_ROOT / "app" / "preview" / "tools" / "spark_transcode_spz.mjs").read_text(encoding="utf-8")

        self.assertIn("const DEFAULT_MAX_SH = 3", source)
        self.assertIn("const DEFAULT_FRACTIONAL_BITS = 14", source)
        self.assertIn('readIntEnv("SPARK_SPZ_MAX_SH"', source)
        self.assertIn('readIntEnv("SPARK_SPZ_FRACTIONAL_BITS"', source)


if __name__ == "__main__":
    unittest.main()
