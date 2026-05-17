from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fine.colmap_cli import choose_max_num_matches, merge_gaussian_ply_chunks, resolve_colmap_policy  # noqa: E402
from app.fine.types import FineFailure  # noqa: E402


class ColmapCliPolicyTests(unittest.TestCase):
    def test_choose_max_num_matches_respects_scene_caps(self) -> None:
        indoor = choose_max_num_matches(12.0, "indoor", 200)
        outdoor_large = choose_max_num_matches(12.0, "outdoor", 4000)

        self.assertGreaterEqual(indoor, 10000)
        self.assertLessEqual(indoor, 24000)
        self.assertGreaterEqual(outdoor_large, 6000)
        self.assertLessEqual(outdoor_large, 14000)

    def test_policy_selects_indoor_quality_and_outdoor_scale(self) -> None:
        indoor = resolve_colmap_policy(
            scene_type="indoor",
            input_type="video",
            n_images=240,
            free_vram_gb=12,
            quality_mode="auto",
            capture_order="auto",
            prefer_gpu=True,
            gpu_index="0",
        )
        outdoor = resolve_colmap_policy(
            scene_type="outdoor",
            input_type="images",
            n_images=2000,
            free_vram_gb=16,
            quality_mode="auto",
            capture_order="unordered",
            prefer_gpu=True,
            gpu_index="0",
        )

        self.assertIn("sequential", indoor.matchers)
        self.assertTrue(indoor.guided_matching)
        self.assertTrue(indoor.estimate_affine_shape)
        self.assertEqual(indoor.mapper, "mapper")
        self.assertIn("vocab_tree", outdoor.matchers)
        self.assertFalse(outdoor.guided_matching)
        self.assertEqual(outdoor.mapper, "global")
        self.assertEqual(outdoor.fastgs_resolution, 1280)

    def test_merge_gaussian_ply_chunks_concatenates_matching_vertex_schema(self) -> None:
        try:
            import numpy as np
            from plyfile import PlyData, PlyElement
        except Exception as exc:
            raise unittest.SkipTest(f"PLY dependencies unavailable: {exc}") from exc

        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left.ply"
            right = root / "right.ply"
            out = root / "merged.ply"
            PlyData([PlyElement.describe(np.zeros(2, dtype=dtype), "vertex")]).write(left)
            PlyData([PlyElement.describe(np.ones(3, dtype=dtype), "vertex")]).write(right)

            count = merge_gaussian_ply_chunks([left, right], out)

            merged = PlyData.read(out)
            self.assertEqual(count, 5)
            self.assertEqual(len(merged["vertex"].data), 5)

    def test_merge_gaussian_ply_chunks_rejects_schema_mismatch(self) -> None:
        try:
            import numpy as np
            from plyfile import PlyData, PlyElement
        except Exception as exc:
            raise unittest.SkipTest(f"PLY dependencies unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left.ply"
            right = root / "right.ply"
            out = root / "merged.ply"
            PlyData([PlyElement.describe(np.zeros(1, dtype=np.dtype([("x", "f4"), ("y", "f4")])), "vertex")]).write(left)
            PlyData([PlyElement.describe(np.zeros(1, dtype=np.dtype([("x", "f4"), ("z", "f4")])), "vertex")]).write(right)

            with self.assertRaises(FineFailure) as raised:
                merge_gaussian_ply_chunks([left, right], out)

        self.assertEqual(raised.exception.code, "FASTGS_PLY_SCHEMA_MISMATCH")


if __name__ == "__main__":
    unittest.main()
