from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

try:
    import numpy as np
    from app.preview.types import PreviewFailure
    from app.preview.vendor.litevggt_runtime import (
        LiteVGGTWindowResult,
        _merge_litevggt_windows,
        align_indices_to_multiple_of_8,
        build_litevggt_chunks,
        compute_sim3_alignment_metrics,
        nearest_anchor_indices,
        point_indices_to_frame_indices,
        resolve_litevggt_effective_mode,
        validate_sim3_alignment,
    )

    RUNTIME_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - local dependency guard
    np = None
    RUNTIME_IMPORT_ERROR = exc


@unittest.skipIf(RUNTIME_IMPORT_ERROR is not None, f"LiteVGGT runtime dependencies unavailable: {RUNTIME_IMPORT_ERROR}")
class LiteVGGTModeTests(unittest.TestCase):
    def test_auto_uses_single_within_limit(self) -> None:
        mode = resolve_litevggt_effective_mode(
            inference_mode="auto",
            aligned_count=144,
            single_frame_limit=192,
            hierarchical_enable=False,
        )

        self.assertEqual(mode, "single")

    def test_auto_uses_global_keyframe_above_limit(self) -> None:
        mode = resolve_litevggt_effective_mode(
            inference_mode="auto",
            aligned_count=1200,
            single_frame_limit=192,
            hierarchical_enable=False,
        )

        self.assertEqual(mode, "global_keyframe")

    def test_auto_uses_hierarchical_only_when_enabled(self) -> None:
        mode = resolve_litevggt_effective_mode(
            inference_mode="auto",
            aligned_count=1200,
            single_frame_limit=192,
            hierarchical_enable=True,
        )

        self.assertEqual(mode, "hierarchical")

    def test_auto_mode_regression_matrix(self) -> None:
        cases = [
            (20, 192, False, "single"),
            (144, 192, False, "single"),
            (144, 64, False, "global_keyframe"),
            (1200, 192, False, "global_keyframe"),
            (1200, 192, True, "hierarchical"),
        ]

        for frame_count, limit, hierarchical_enable, expected in cases:
            with self.subTest(frame_count=frame_count, limit=limit, hierarchical_enable=hierarchical_enable):
                self.assertEqual(
                    resolve_litevggt_effective_mode(
                        inference_mode="auto",
                        aligned_count=frame_count,
                        single_frame_limit=limit,
                        hierarchical_enable=hierarchical_enable,
                    ),
                    expected,
                )

    def test_hierarchical_builds_chunks_and_aligned_batches(self) -> None:
        chunks = build_litevggt_chunks(160, chunk_size=64, overlap=16)
        batch = align_indices_to_multiple_of_8(chunks[0] + [0, 32, 64], list(range(160)))

        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(len(batch) % 8, 0)

    def test_nearest_anchor_indices_returns_at_least_three_when_available(self) -> None:
        anchors = nearest_anchor_indices([48, 49, 50, 51], [0, 32, 48, 64, 96], anchor_count=3)

        self.assertGreaterEqual(len(anchors), 3)
        self.assertEqual(anchors, sorted(anchors))

    def test_identity_sim3_metrics_are_zero(self) -> None:
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float32)
        metrics = compute_sim3_alignment_metrics(
            source,
            source,
            1.0,
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )

        self.assertLess(metrics["median"], 1e-6)
        self.assertLess(metrics["p90"], 1e-6)

    def test_scale_out_of_range_is_rejected(self) -> None:
        with self.assertRaises(PreviewFailure) as cm:
            validate_sim3_alignment(
                scale=8.0,
                metrics={"rel_median": 0.0, "rel_p90": 0.0},
                alignment_min_scale=0.25,
                alignment_max_scale=4.0,
                alignment_max_rel_median=0.05,
                alignment_max_rel_p90=0.12,
                code="LITEVGGT_HIERARCHICAL_ALIGNMENT_UNSTABLE",
                message="unstable hierarchical chunk alignment",
            )

        self.assertEqual(cm.exception.code, "LITEVGGT_HIERARCHICAL_ALIGNMENT_UNSTABLE")

    def test_point_indices_map_to_original_frame_indices(self) -> None:
        selected = np.array([0, 3, 4, 7, 8, 11], dtype=np.int64)
        mapped = point_indices_to_frame_indices(selected, frame_indices=[10, 20, 30], height=2, width=2)

        np.testing.assert_array_equal(mapped, np.array([10, 10, 20, 20, 30, 30], dtype=np.int32))

    def test_windowed_alignment_residual_over_threshold_is_rejected(self) -> None:
        first_centers = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [2, 0, 1], [3, 1, 1]],
            dtype=np.float32,
        )
        second_centers = np.array(
            [[2, 0, 0], [2, 1, 0], [2, 0, 1], [30, 30, 30], [4, 1, 1], [5, 1, 1]],
            dtype=np.float32,
        )
        first = _window_result(0, 6, first_centers)
        second = _window_result(2, 8, second_centers)

        with self.assertRaises(PreviewFailure) as cm:
            _merge_litevggt_windows(
                [first, second],
                8,
                alignment_max_rel_median=0.001,
                alignment_max_rel_p90=0.001,
            )

        self.assertEqual(cm.exception.code, "LITEVGGT_WINDOW_ALIGNMENT_UNSTABLE")


def _window_result(start: int, end: int, centers: "np.ndarray") -> LiteVGGTWindowResult:
    w2c = np.tile(np.eye(4, dtype=np.float32), (end - start, 1, 1))
    w2c[:, :3, 3] = -centers
    point_count = max(1, end - start)
    return LiteVGGTWindowResult(
        start=start,
        end=end,
        images=np.zeros((end - start, 2, 2, 3), dtype=np.float32),
        valid_masks=np.ones((end - start, 2, 2), dtype=bool),
        w2c=w2c,
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (end - start, 1, 1)),
        points=np.zeros((point_count, 3), dtype=np.float32),
        colors=np.zeros((point_count, 3), dtype=np.uint8),
        confidence=np.ones((point_count,), dtype=np.float32),
        point_frame_indices=np.arange(start, start + point_count, dtype=np.int32),
        valid_pixel_count=point_count,
        point_count_before_filter=point_count,
        point_count_after_filter=point_count,
    )


if __name__ == "__main__":
    unittest.main()
