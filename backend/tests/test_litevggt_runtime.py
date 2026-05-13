from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

try:
    import numpy as np
    import app.preview.vendor.litevggt_runtime as litevggt_runtime
    from app.preview.vendor.litevggt_runtime import (
        LiteVGGTWindowFrameMismatch,
        LiteVGGTWindowResult,
        _merge_litevggt_windows,
        build_litevggt_windows,
        resolve_litevggt_window_attempts,
        select_aligned_frames,
    )

    RUNTIME_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - local dependency guard
    np = None
    RUNTIME_IMPORT_ERROR = exc


@unittest.skipIf(RUNTIME_IMPORT_ERROR is not None, f"LiteVGGT runtime dependencies unavailable: {RUNTIME_IMPORT_ERROR}")
class LiteVGGTRuntimeTests(unittest.TestCase):
    def test_window_attempts_are_aligned_to_multiples_of_8(self) -> None:
        attempts = resolve_litevggt_window_attempts(50, [34, 15, 8])

        self.assertEqual(attempts, [48, 32, 8])

    def test_100_images_align_to_96_and_windows_cover_all_frames(self) -> None:
        files = [Path(f"{index:03d}.jpg") for index in range(100)]
        selected = select_aligned_frames(files, multiple=8, mode="uniform")
        windows = build_litevggt_windows(len(selected), 48, 16)

        covered = set()
        for start, end in windows:
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, len(selected))
            self.assertEqual(end - start, 48)
            covered.update(range(start, end))

        self.assertEqual(len(selected), 96)
        self.assertEqual(covered, set(range(96)))

    def test_merge_rejects_malformed_window_result_before_indexing(self) -> None:
        result = LiteVGGTWindowResult(
            start=0,
            end=48,
            images=np.zeros((48, 2, 2, 3), dtype=np.float32),
            valid_masks=np.ones((48, 2, 2), dtype=bool),
            w2c=np.tile(np.eye(4, dtype=np.float32), (36, 1, 1)),
            intrinsics=np.tile(np.eye(3, dtype=np.float32), (48, 1, 1)),
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.zeros((1, 3), dtype=np.uint8),
            confidence=np.ones((1,), dtype=np.float32),
            valid_pixel_count=48,
            point_count_before_filter=48,
            point_count_after_filter=1,
        )

        with self.assertRaises(LiteVGGTWindowFrameMismatch):
            _merge_litevggt_windows([result], 48)

    def test_merge_rejects_short_transformed_w2c_before_indexing(self) -> None:
        result = LiteVGGTWindowResult(
            start=0,
            end=8,
            images=np.zeros((8, 2, 2, 3), dtype=np.float32),
            valid_masks=np.ones((8, 2, 2), dtype=bool),
            w2c=np.tile(np.eye(4, dtype=np.float32), (8, 1, 1)),
            intrinsics=np.tile(np.eye(3, dtype=np.float32), (8, 1, 1)),
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.zeros((1, 3), dtype=np.uint8),
            confidence=np.ones((1,), dtype=np.float32),
            valid_pixel_count=8,
            point_count_before_filter=8,
            point_count_after_filter=1,
        )

        with patch.object(litevggt_runtime, "transform_w2c_sim3", return_value=np.tile(np.eye(4, dtype=np.float32), (6, 1, 1))):
            with self.assertRaises(LiteVGGTWindowFrameMismatch):
                _merge_litevggt_windows([result], 8)

    def test_merge_accepts_3x4_w2c_without_losing_frames(self) -> None:
        w2c = np.zeros((8, 3, 4), dtype=np.float32)
        w2c[:, :3, :3] = np.eye(3, dtype=np.float32)
        w2c[:, 0, 3] = np.arange(8, dtype=np.float32)
        result = LiteVGGTWindowResult(
            start=0,
            end=8,
            images=np.zeros((8, 2, 2, 3), dtype=np.float32),
            valid_masks=np.ones((8, 2, 2), dtype=bool),
            w2c=w2c,
            intrinsics=np.tile(np.eye(3, dtype=np.float32), (8, 1, 1)),
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.zeros((1, 3), dtype=np.uint8),
            confidence=np.ones((1,), dtype=np.float32),
            valid_pixel_count=8,
            point_count_before_filter=8,
            point_count_after_filter=1,
        )

        _, _, merged_w2c, _, _, _, _ = _merge_litevggt_windows([result], 8)

        self.assertEqual(merged_w2c.shape, (8, 4, 4))
        np.testing.assert_allclose(merged_w2c[:, 3, :], np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (8, 1)))


@unittest.skipIf(RUNTIME_IMPORT_ERROR is not None, f"LiteVGGT runtime dependencies unavailable: {RUNTIME_IMPORT_ERROR}")
class LiteVGGTSceneTests(unittest.TestCase):
    def test_fine_litevggt_scene_uses_windowed_defaults(self) -> None:
        from app.fine import litevggt_scene

        reconstruction = SimpleNamespace(
            images=np.zeros((8, 2, 2, 3), dtype=np.float32),
            w2c=np.tile(np.eye(4, dtype=np.float32), (8, 1, 1)),
            intrinsics=np.tile(np.eye(3, dtype=np.float32), (8, 1, 1)),
            points=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            colors=np.array([[255, 255, 255]], dtype=np.uint8),
            metrics={},
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            model_cache_dir = root / "model-cache"
            scene_dir = root / "scene"
            weight = model_cache_dir / "litevggt" / "te_dict.pt"
            input_dir.mkdir()
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"weight")

            with patch("app.fine.litevggt_scene.run_litevggt_reconstruction", return_value=reconstruction) as run:
                litevggt_scene.build_litevggt_scene(
                    input_dir,
                    scene_dir,
                    model_cache_dir=model_cache_dir,
                    options={},
                    progress=lambda stage, value, message: None,
                )

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["inference_mode"], "windowed")
        self.assertEqual(kwargs["window_size"], 48)
        self.assertEqual(kwargs["window_overlap"], 16)
        self.assertEqual(kwargs["oom_window_sizes"], [32, 16, 8])


if __name__ == "__main__":
    unittest.main()
