from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fine.colmap_cli import (  # noqa: E402
    ColmapCapabilities,
    choose_max_num_matches,
    detect_colmap_capabilities,
    retry_policies,
    resolve_colmap_policy,
)


STANDARD_COLMAP_HELP = """
Usage:
  colmap [command]

Available commands:
    feature_extractor
    exhaustive_matcher
    sequential_matcher
    vocab_tree_matcher
    mapper
    hierarchical_mapper
    image_undistorter
    model_analyzer
"""

GLOBAL_COLMAP_HELP = """
Usage:
  colmap [command]

Available commands:
    feature_extractor
    exhaustive_matcher
    sequential_matcher
    vocab_tree_matcher
    mapper
    global_mapper
    image_undistorter
    model_analyzer
"""


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
            matcher_policy="auto",
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

    def test_outdoor_uses_mapper_before_large_scale_global_threshold(self) -> None:
        outdoor = resolve_colmap_policy(
            scene_type="outdoor",
            input_type="images",
            n_images=1200,
            free_vram_gb=16,
            quality_mode="auto",
            capture_order="unordered",
            matcher_policy="auto",
            prefer_gpu=True,
            gpu_index="0",
        )

        self.assertEqual(outdoor.mapper, "mapper")

    def test_explicit_matcher_policy_overrides_auto_matchers(self) -> None:
        outdoor = resolve_colmap_policy(
            scene_type="outdoor",
            input_type="images",
            n_images=4000,
            free_vram_gb=24,
            quality_mode="auto",
            capture_order="unordered",
            matcher_policy="spatial",
            prefer_gpu=True,
            gpu_index="0",
        )

        self.assertEqual(outdoor.matchers, ["spatial"])

    def test_standard_colmap_without_global_mapper_is_supported(self) -> None:
        with patch("app.fine.colmap_cli.shutil.which", return_value="/usr/bin/colmap"), patch(
            "app.fine.colmap_cli.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout=STANDARD_COLMAP_HELP),
        ):
            capabilities = detect_colmap_capabilities()

        self.assertIn("mapper", capabilities.commands)
        self.assertIn("hierarchical_mapper", capabilities.commands)
        self.assertNotIn("global_mapper", capabilities.commands)

    def test_retry_policy_falls_back_when_global_mapper_is_unavailable(self) -> None:
        capabilities = ColmapCapabilities(
            executable="/usr/bin/colmap",
            help_text=STANDARD_COLMAP_HELP,
            commands={
                "feature_extractor",
                "exhaustive_matcher",
                "sequential_matcher",
                "vocab_tree_matcher",
                "mapper",
                "hierarchical_mapper",
                "image_undistorter",
                "model_analyzer",
            },
            has_aliked_lightglue=False,
            has_sift_lightglue=False,
        )

        policies = retry_policies(
            scene_type="outdoor",
            input_type="images",
            n_images=2000,
            free_vram_gb=16,
            quality_mode="auto",
            capture_order="unordered",
            matcher_policy="auto",
            prefer_gpu=True,
            gpu_index="0",
            capabilities=capabilities,
        )

        self.assertTrue(policies)
        self.assertTrue(all(policy.mapper == "mapper" for policy in policies))

    def test_retry_policy_uses_global_mapper_when_available_without_calibrator(self) -> None:
        capabilities = ColmapCapabilities(
            executable="/usr/bin/colmap",
            help_text=GLOBAL_COLMAP_HELP,
            commands={
                "feature_extractor",
                "exhaustive_matcher",
                "sequential_matcher",
                "vocab_tree_matcher",
                "mapper",
                "global_mapper",
                "image_undistorter",
                "model_analyzer",
            },
            has_aliked_lightglue=False,
            has_sift_lightglue=False,
        )

        policies = retry_policies(
            scene_type="outdoor",
            input_type="images",
            n_images=2000,
            free_vram_gb=16,
            quality_mode="auto",
            capture_order="unordered",
            matcher_policy="auto",
            prefer_gpu=True,
            gpu_index="0",
            capabilities=capabilities,
        )

        self.assertTrue(policies)
        self.assertTrue(all(policy.mapper == "global" for policy in policies))
        self.assertTrue(all(not policy.use_view_graph_calibrator for policy in policies))

    def test_colmap_help_detection_handles_unindented_global_mapper(self) -> None:
        help_text = "Available commands: feature_extractor mapper global_mapper image_undistorter model_analyzer"
        with patch("app.fine.colmap_cli.shutil.which", return_value="/usr/bin/colmap"), patch(
            "app.fine.colmap_cli.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout=help_text),
        ):
            capabilities = detect_colmap_capabilities()

        self.assertIn("global_mapper", capabilities.commands)


if __name__ == "__main__":
    unittest.main()
