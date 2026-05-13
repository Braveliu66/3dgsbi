from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

try:
    import numpy as np
    from app.preview.adapters import litevggt as litevggt_adapter
    from app.preview.types import PreviewContext, PreviewFailure
    from app.preview.vendor.litevggt_runtime import (
        litevggt_preview_bounds,
        load_litevggt_image_batch,
        load_litevggt_image_tensors,
        load_litevggt_padded_image,
        point_indices_to_frame_indices,
        resolve_litevggt_frame_selection,
        resolve_litevggt_quality_settings,
        select_aligned_frames,
        select_points_by_scene_coverage,
        select_points_by_confidence,
        write_litevggt_preview_meta_json,
    )

    RUNTIME_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - local dependency guard
    np = None
    RUNTIME_IMPORT_ERROR = exc


@unittest.skipIf(RUNTIME_IMPORT_ERROR is not None, f"LiteVGGT runtime dependencies unavailable: {RUNTIME_IMPORT_ERROR}")
class LiteVGGTOfficialPathTests(unittest.TestCase):
    def test_quality_settings_follow_frame_count_profiles(self) -> None:
        cases = [
            (8, 448, 1.0, "small"),
            (32, 392, 1.0, "medium"),
            (80, 336, 1.0, "large"),
            (200, 308, 1.0, "xlarge"),
            (400, 280, 1.0, "huge"),
        ]

        for frame_count, target_size, keep_ratio, profile in cases:
            with self.subTest(frame_count=frame_count):
                settings = resolve_litevggt_quality_settings(frame_count)
                self.assertEqual(settings.target_size, target_size)
                self.assertAlmostEqual(settings.keep_ratio, keep_ratio)
                self.assertEqual(settings.quality_profile, profile)
                self.assertEqual(settings.keep_ratio_source, "auto")
                self.assertEqual(settings.target_size_source, "auto")

    def test_quality_settings_user_overrides_win(self) -> None:
        settings = resolve_litevggt_quality_settings(
            400,
            {
                "target_size": 300,
                "keep_ratio": 0.95,
            },
        )

        self.assertEqual(settings.target_size, 294)
        self.assertAlmostEqual(settings.keep_ratio, 0.95)
        self.assertEqual(settings.quality_profile, "huge")
        self.assertEqual(settings.keep_ratio_source, "user")
        self.assertEqual(settings.target_size_source, "user")

    def test_image_loading_passes_target_size_to_official_loader(self) -> None:
        calls = []

        def fake_load(path, *, target_size):
            calls.append((path, target_size))
            return np.zeros((14, target_size, 3), dtype=np.float32)

        tensors = load_litevggt_image_tensors([Path("a.jpg"), Path("b.jpg")], fake_load, 336)

        self.assertEqual(calls, [("a.jpg", 336), ("b.jpg", 336)])
        self.assertEqual(len(tensors), 2)
        self.assertEqual(tuple(tensors[0].shape), (3, 14, 336))

    def test_crop_image_batch_at_518_keeps_14_aligned_dimensions(self) -> None:
        calls = []

        def fake_load(path, *, target_size):
            calls.append((path, target_size))
            return np.zeros((350, 518, 3), dtype=np.float32)

        batch = load_litevggt_image_batch([Path("crop.jpg")], fake_load, 518, preprocess_mode="crop")

        self.assertEqual(calls, [("crop.jpg", 518)])
        self.assertEqual(batch.preprocess_mode, "crop")
        self.assertEqual(tuple(batch.tensors[0].shape), (3, 350, 518))
        self.assertEqual(batch.valid_masks.shape, (1, 350, 518))
        self.assertEqual(350 % 14, 0)
        self.assertEqual(518 % 14, 0)

    def test_padded_image_loading_preserves_full_image_with_valid_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wide.jpg"
            from PIL import Image

            Image.new("RGB", (40, 20), (20, 120, 200)).save(path)

            image, mask = load_litevggt_padded_image(path, 28)

        self.assertEqual(image.shape, (28, 28, 3))
        self.assertEqual(mask.shape, (28, 28))
        self.assertEqual(int(mask.sum()), 28 * 14)

    def test_select_aligned_frames_keeps_complete_ordered_scene(self) -> None:
        files = [Path(f"{index:03d}.jpg") for index in range(18)]

        selected = select_aligned_frames(files, multiple=8)

        self.assertEqual(len(selected), 16)
        self.assertEqual(selected[0], files[0])
        self.assertEqual(selected[-1], files[-1])
        self.assertEqual(selected, sorted(selected, key=lambda path: path.name))

    def test_select_aligned_frames_respects_max_input_frames_as_upper_bound(self) -> None:
        files = [Path(f"{index:03d}.jpg") for index in range(30)]

        selected = select_aligned_frames(files, multiple=8, max_frames=17)

        self.assertEqual(len(selected), 16)
        self.assertEqual(selected[0], files[0])
        self.assertEqual(selected[-1], files[-1])

    def test_frame_selection_auto_uses_large_inputs_across_full_scene(self) -> None:
        files = [Path(f"{index:03d}.jpg") for index in range(400)]

        selection = resolve_litevggt_frame_selection(files, multiple=8)

        self.assertEqual(len(selection.files), 400)
        self.assertEqual(selection.files[0], files[0])
        self.assertEqual(selection.files[-1], files[-1])
        self.assertEqual(selection.frame_stride, 1)
        self.assertEqual(selection.frame_stride_source, "auto")

    def test_frame_selection_allows_user_stride(self) -> None:
        files = [Path(f"{index:03d}.jpg") for index in range(40)]

        selection = resolve_litevggt_frame_selection(files, multiple=8, frame_stride=3)

        self.assertEqual(len(selection.files), 8)
        self.assertEqual(selection.files[0], files[0])
        self.assertEqual(selection.files[-1], files[-1])
        self.assertEqual(selection.frame_stride, 3)
        self.assertEqual(selection.frame_stride_source, "user")

    def test_point_indices_map_to_original_frame_indices(self) -> None:
        selected = np.array([0, 3, 4, 7, 8, 11], dtype=np.int64)
        mapped = point_indices_to_frame_indices(selected, frame_indices=[10, 20, 30], height=2, width=2)

        np.testing.assert_array_equal(mapped, np.array([10, 10, 20, 20, 30, 30], dtype=np.int32))

    def test_confidence_selection_uses_top_scores_and_max_points(self) -> None:
        points = np.arange(18, dtype=np.float32).reshape(6, 3)
        colors = np.arange(18, dtype=np.uint8).reshape(6, 3)
        confidence = np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.7], dtype=np.float32)

        selected_points, selected_colors, selected_confidence, frame_indices = select_points_by_confidence(
            points,
            colors,
            confidence,
            frame_indices=[100, 200],
            height=1,
            width=3,
            keep_ratio=1.0,
            max_points=3,
        )

        np.testing.assert_array_equal(selected_points, points[[1, 3, 5]])
        np.testing.assert_array_equal(selected_colors, colors[[1, 3, 5]])
        np.testing.assert_array_equal(selected_confidence, confidence[[1, 3, 5]])
        np.testing.assert_array_equal(frame_indices, np.array([100, 200, 200], dtype=np.int32))

    def test_confidence_selection_matches_official_top_confidence_order(self) -> None:
        points = np.arange(18, dtype=np.float32).reshape(6, 3)
        colors = np.arange(18, dtype=np.uint8).reshape(6, 3)
        confidence = np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.7], dtype=np.float32)

        selected_points, selected_colors, selected_confidence, frame_indices = select_points_by_confidence(
            points,
            colors,
            confidence,
            frame_indices=[100, 200],
            height=1,
            width=3,
            keep_ratio=0.5,
            max_points=0,
        )

        np.testing.assert_array_equal(selected_points, points[[1, 3, 5]])
        np.testing.assert_array_equal(selected_colors, colors[[1, 3, 5]])
        np.testing.assert_array_equal(selected_confidence, confidence[[1, 3, 5]])
        np.testing.assert_array_equal(frame_indices, np.array([100, 200, 200], dtype=np.int32))

    def test_scene_coverage_selection_keeps_image_regions(self) -> None:
        points = np.arange(48, dtype=np.float32).reshape(16, 3)
        colors = np.arange(48, dtype=np.uint8).reshape(16, 3)
        confidence = np.array(
            [
                0.99,
                0.98,
                0.10,
                0.09,
                0.97,
                0.96,
                0.08,
                0.07,
                0.06,
                0.05,
                0.04,
                0.03,
                0.02,
                0.01,
                0.20,
                0.19,
            ],
            dtype=np.float32,
        )

        selected_points, _, _, frame_indices = select_points_by_scene_coverage(
            points,
            colors,
            confidence,
            frame_indices=[7],
            height=4,
            width=4,
            keep_ratio=0.25,
            max_points=4,
            grid_size=2,
        )

        selected_pixel_indices = (selected_points[:, 0] / 3).astype(np.int32)
        self.assertEqual(set(selected_pixel_indices.tolist()), {0, 2, 8, 14})
        np.testing.assert_array_equal(frame_indices, np.array([7, 7, 7, 7], dtype=np.int32))

    def test_confidence_selection_rejects_empty_valid_points(self) -> None:
        points = np.full((2, 3), np.nan, dtype=np.float32)
        colors = np.zeros((2, 3), dtype=np.uint8)
        confidence = np.ones((2,), dtype=np.float32)

        with self.assertRaises(PreviewFailure) as cm:
            select_points_by_confidence(
                points,
                colors,
                confidence,
                frame_indices=[0],
                height=1,
                width=2,
                keep_ratio=1.0,
                max_points=10,
            )

        self.assertEqual(cm.exception.code, "LITEVGGT_EMPTY_POINT_CLOUD")

    def test_preview_bounds_ignore_non_finite_points_and_meta_uses_bbox(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 2.0, 3.0],
                [np.nan, 3.0, 4.0],
                [2.0, 4.0, 5.0],
            ],
            dtype=np.float32,
        )

        bounds = litevggt_preview_bounds(points)

        self.assertEqual(len(bounds["bbox_min"]), 3)
        self.assertEqual(len(bounds["bbox_max"]), 3)
        self.assertEqual(len(bounds["bbox_center"]), 3)
        self.assertGreater(bounds["bbox_radius"], 0.0)
        self.assertTrue(np.isfinite(np.array(bounds["bbox_min"])).all())
        self.assertTrue(np.isfinite(np.array(bounds["bbox_max"])).all())

        with tempfile.TemporaryDirectory() as tmp:
            meta_path = Path(tmp) / "preview_meta.json"
            write_litevggt_preview_meta_json(
                meta_path,
                point_count_raw=4,
                point_count_exported=3,
                bounds=bounds,
            )
            payload = json.loads(meta_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["point_source"], "litevggt_depth_unprojected")
        self.assertEqual(payload["point_count_raw"], 4)
        self.assertEqual(payload["point_count_exported"], 3)
        self.assertEqual(payload["bbox_min"], bounds["bbox_min"])
        self.assertEqual(payload["bbox_max"], bounds["bbox_max"])
        self.assertEqual(payload["bbox_center"], bounds["bbox_center"])
        self.assertEqual(payload["bbox_radius"], bounds["bbox_radius"])
        self.assertEqual(payload["center"], bounds["bbox_center"])
        self.assertEqual(payload["radius"], bounds["bbox_radius"])

    def test_preview_bounds_reject_empty_or_non_finite_clouds(self) -> None:
        with self.assertRaises(PreviewFailure) as cm:
            litevggt_preview_bounds(np.full((2, 3), np.nan, dtype=np.float32))

        self.assertEqual(cm.exception.code, "LITEVGGT_EMPTY_POINT_CLOUD")


@unittest.skipIf(RUNTIME_IMPORT_ERROR is not None, f"LiteVGGT runtime dependencies unavailable: {RUNTIME_IMPORT_ERROR}")
class LiteVGGTAdapterTests(unittest.TestCase):
    def _make_context(self, root: Path, options: dict[str, Any] | None = None) -> PreviewContext:
        input_dir = root / "input"
        work_dir = root / "work"
        output_spz = root / "preview.spz"
        model_cache = root / "model-cache"
        input_dir.mkdir()
        work_dir.mkdir()
        weight = model_cache / "litevggt" / "te_dict.pt"
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b"weight")
        for index in range(8):
            (input_dir / f"{index:03d}.jpg").write_bytes(b"image")

        return PreviewContext(
            task_id="task",
            project_id="project",
            pipeline="litevggt_spz",
            input_dir=input_dir,
            work_dir=work_dir,
            output_spz=output_spz,
            model_cache_dir=model_cache,
            source_version=1,
            options=options or {},
            progress=lambda stage, value, message, metrics: None,
        )

    def _fake_runtime_metrics(self, **kwargs):
        kwargs["output_ply"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_ply"].write_bytes(b"points")
        kwargs["output_meta_json"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_meta_json"].write_text("{}", encoding="utf-8")
        return {
            "point_count": 9,
            "point_count_raw": 12,
            "point_count_exported": 9,
            "point_source": "litevggt_depth_unprojected",
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [1.0, 1.0, 1.0],
            "bbox_center": [0.5, 0.5, 0.5],
            "bbox_radius": 0.9,
            "litevggt_preview_point_radius": 0.003,
        }

    def _fake_splat_ply(self, input_ply, output_ply, **kwargs):
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        output_ply.write_bytes(b"splats")
        return 9

    def _fake_spz(self, input_ply, output_spz):
        output_spz.write_bytes(b"spz")
        return 9

    def test_adapter_uses_official_fidelity_preview_defaults_and_outputs_raw_ply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = self._make_context(root)

            with (
                patch.object(litevggt_adapter, "run_litevggt_pointcloud", side_effect=self._fake_runtime_metrics) as run,
                patch.object(
                    litevggt_adapter,
                    "convert_pointcloud_ply_to_fixed_splat_ply",
                    side_effect=self._fake_splat_ply,
                ) as pointcloud_to_splats,
                patch.object(litevggt_adapter, "convert_ply_to_spz", side_effect=self._fake_spz) as splats_to_spz,
            ):
                result = litevggt_adapter.run(ctx)

        self.assertEqual(run.call_args.kwargs["keep_ratio"], 0.42)
        self.assertEqual(run.call_args.kwargs["target_size"], 518)
        self.assertEqual(run.call_args.kwargs["max_input_frames"], 64)
        self.assertEqual(run.call_args.kwargs["max_points"], 15_000_000)
        self.assertIsNone(run.call_args.kwargs["depth_conf_thresh"])
        self.assertEqual(run.call_args.kwargs["preprocess_mode"], "crop")
        self.assertEqual(run.call_args.kwargs["selection_strategy"], "global_confidence")
        self.assertEqual(run.call_args.kwargs["axis_trim_low_quantile"], 0.0)
        self.assertEqual(run.call_args.kwargs["axis_trim_high_quantile"], 1.0)
        self.assertEqual(run.call_args.kwargs["spatial_keep_quantile"], 1.0)
        self.assertEqual(run.call_args.kwargs["output_ply"].name, "preview_points.ply")
        self.assertEqual(run.call_args.kwargs["output_meta_json"].name, "preview_meta.json")

        self.assertEqual(result.intermediate_ply.name, "preview_points.ply")
        self.assertEqual(result.metrics["intermediate_points_ply"], str(result.intermediate_ply))
        self.assertEqual(Path(result.metrics["intermediate_splats_ply"]).name, "preview_splats.ply")
        self.assertEqual(Path(result.metrics["preview_meta_json"]).name, "preview_meta.json")
        self.assertEqual(result.metrics["point_source"], "litevggt_depth_unprojected")
        self.assertEqual(result.metrics["fixed_splat_base_point_radius"], 0.003)
        self.assertEqual(result.metrics["fixed_splat_point_radius"], 0.003)
        self.assertEqual(result.metrics["fixed_splat_count"], 9)
        self.assertEqual(pointcloud_to_splats.call_args.args[0].name, "preview_points.ply")
        self.assertEqual(pointcloud_to_splats.call_args.args[1].name, "preview_splats.ply")
        self.assertEqual(pointcloud_to_splats.call_args.kwargs["point_radius"], 0.003)
        self.assertEqual(splats_to_spz.call_args.args[0].name, "preview_splats.ply")

    def test_adapter_passes_only_minimal_runtime_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = self._make_context(
                root,
                {
                    "litevggt_keep_ratio": 0.95,
                    "litevggt_target_size": 336,
                    "litevggt_frame_stride": 2,
                    "litevggt_max_input_frames": 12,
                    "preview_max_points": 1234,
                    "litevggt_inference_mode": "ignored",
                    "litevggt_chunk_size": 24,
                },
            )

            with (
                patch.object(litevggt_adapter, "run_litevggt_pointcloud", side_effect=self._fake_runtime_metrics) as run,
                patch.object(
                    litevggt_adapter,
                    "convert_pointcloud_ply_to_fixed_splat_ply",
                    side_effect=self._fake_splat_ply,
                ),
                patch.object(litevggt_adapter, "convert_ply_to_spz", side_effect=self._fake_spz),
            ):
                result = litevggt_adapter.run(ctx)

        self.assertEqual(result.splat_count, 9)
        self.assertEqual(run.call_args.kwargs["keep_ratio"], 0.95)
        self.assertEqual(run.call_args.kwargs["target_size"], 336)
        self.assertEqual(run.call_args.kwargs["frame_stride"], 2)
        self.assertIsNone(run.call_args.kwargs["depth_conf_thresh"])
        self.assertEqual(run.call_args.kwargs["preprocess_mode"], "crop")
        self.assertEqual(run.call_args.kwargs["max_points"], 1234)
        self.assertEqual(run.call_args.kwargs["max_input_frames"], 12)
        self.assertEqual(run.call_args.kwargs["selection_strategy"], "global_confidence")
        self.assertEqual(run.call_args.kwargs["axis_trim_low_quantile"], 0.0)
        self.assertEqual(run.call_args.kwargs["axis_trim_high_quantile"], 1.0)
        self.assertEqual(run.call_args.kwargs["spatial_keep_quantile"], 1.0)
        self.assertEqual(
            sorted(run.call_args.kwargs),
            [
                "axis_trim_high_quantile",
                "axis_trim_low_quantile",
                "checkpoint_path",
                "depth_conf_thresh",
                "frame_stride",
                "input_dir",
                "keep_ratio",
                "max_input_frames",
                "max_points",
                "output_meta_json",
                "output_ply",
                "preprocess_mode",
                "progress",
                "selection_strategy",
                "spatial_keep_quantile",
                "target_size",
            ],
        )

    def test_preview_artifact_metrics_carries_litevggt_point_metadata(self) -> None:
        try:
            from app.worker import preview_artifact_metrics
        except Exception as exc:  # pragma: no cover - dependency guard for slim local envs
            self.skipTest(f"worker dependencies unavailable: {exc}")

        metadata = preview_artifact_metrics(
            {
                "point_source": "litevggt_depth_unprojected",
                "bbox_min": [0.0, 0.0, 0.0],
                "bbox_max": [1.0, 2.0, 3.0],
                "bbox_center": [0.5, 1.0, 1.5],
                "bbox_radius": 1.9,
            }
        )

        self.assertEqual(metadata["point_source"], "litevggt_depth_unprojected")
        self.assertEqual(metadata["bbox_min"], [0.0, 0.0, 0.0])
        self.assertEqual(metadata["bbox_max"], [1.0, 2.0, 3.0])
        self.assertEqual(metadata["bbox_center"], [0.5, 1.0, 1.5])
        self.assertEqual(metadata["bbox_radius"], 1.9)


if __name__ == "__main__":
    unittest.main()
