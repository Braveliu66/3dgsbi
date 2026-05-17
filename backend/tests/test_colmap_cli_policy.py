from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fine.colmap_cli import choose_max_num_matches, resolve_colmap_policy  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
