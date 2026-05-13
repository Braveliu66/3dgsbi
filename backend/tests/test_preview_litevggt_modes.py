from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

try:
    import numpy as np
    from app.preview.adapters import litevggt as litevggt_adapter
    from app.preview.types import PreviewContext, PreviewFailure
    from app.preview.vendor.litevggt_runtime import (
        load_litevggt_image_tensors,
        point_indices_to_frame_indices,
        resolve_litevggt_quality_settings,
        select_aligned_frames,
        select_points_by_confidence,
    )

    RUNTIME_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - local dependency guard
    np = None
    RUNTIME_IMPORT_ERROR = exc


@unittest.skipIf(RUNTIME_IMPORT_ERROR is not None, f"LiteVGGT runtime dependencies unavailable: {RUNTIME_IMPORT_ERROR}")
class LiteVGGTOfficialPathTests(unittest.TestCase):
    def test_quality_settings_follow_frame_count_profiles(self) -> None:
        cases = [
            (8, 448, 0.90, "small"),
            (32, 392, 0.85, "medium"),
            (80, 336, 0.75, "large"),
            (200, 308, 0.60, "xlarge"),
            (400, 280, 0.42, "huge"),
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

    def test_select_aligned_frames_keeps_complete_ordered_scene(self) -> None:
        files = [Path(f"{index:03d}.jpg") for index in range(18)]

        selected = select_aligned_frames(files, multiple=8)

        self.assertEqual(selected, files[:16])

    def test_select_aligned_frames_respects_max_input_frames_as_upper_bound(self) -> None:
        files = [Path(f"{index:03d}.jpg") for index in range(30)]

        selected = select_aligned_frames(files, multiple=8, max_frames=17)

        self.assertEqual(selected, files[:16])

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


@unittest.skipIf(RUNTIME_IMPORT_ERROR is not None, f"LiteVGGT runtime dependencies unavailable: {RUNTIME_IMPORT_ERROR}")
class LiteVGGTAdapterTests(unittest.TestCase):
    def test_adapter_passes_only_minimal_runtime_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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

            ctx = PreviewContext(
                task_id="task",
                project_id="project",
                pipeline="litevggt_spz",
                input_dir=input_dir,
                work_dir=work_dir,
                output_spz=output_spz,
                model_cache_dir=model_cache,
                source_version=1,
                options={
                    "litevggt_keep_ratio": 0.95,
                    "litevggt_target_size": 336,
                    "preview_max_points": 1234,
                    "litevggt_inference_mode": "ignored",
                    "litevggt_chunk_size": 24,
                },
                progress=lambda stage, value, message, metrics: None,
            )

            ply_path = work_dir / "litevggt" / "recon.ply"
            ply_path.parent.mkdir(parents=True)
            ply_path.write_bytes(b"ply")
            output_spz.write_bytes(b"spz")

            with (
                patch.object(litevggt_adapter, "run_litevggt_pointcloud", return_value={"point_count": 9}) as run,
                patch.object(litevggt_adapter, "convert_ply_to_spz", return_value=9),
            ):
                result = litevggt_adapter.run(ctx)

        self.assertEqual(result.splat_count, 9)
        self.assertEqual(run.call_args.kwargs["keep_ratio"], 0.95)
        self.assertEqual(run.call_args.kwargs["target_size"], 336)
        self.assertEqual(run.call_args.kwargs["max_points"], 1234)
        self.assertEqual(
            sorted(run.call_args.kwargs),
            [
                "checkpoint_path",
                "input_dir",
                "keep_ratio",
                "max_input_frames",
                "max_points",
                "output_ply",
                "progress",
                "target_size",
            ],
        )


if __name__ == "__main__":
    unittest.main()
