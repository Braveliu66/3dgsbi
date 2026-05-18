from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.litevggt_defaults import (  # noqa: E402
    LITEVGGT_VIDEO_QUALITY_DEFAULTS_BY_SCENE,
    apply_litevggt_video_speed_defaults,
    litevggt_system_defaults,
)

try:
    from app.preview.video_preprocess import _resolve_video_sample_fps  # noqa: E402
except Exception as exc:  # pragma: no cover - local dependency guard
    _resolve_video_sample_fps = None
    VIDEO_PREPROCESS_IMPORT_ERROR = exc
else:
    VIDEO_PREPROCESS_IMPORT_ERROR = None


class PreviewVideoSpeedDefaultsTests(unittest.TestCase):
    def test_image_defaults_use_quality_presets_by_scene(self) -> None:
        indoor = litevggt_system_defaults("indoor")
        outdoor = litevggt_system_defaults("outdoor")

        self.assertEqual(indoor["preview_image_max_side"], 1024)
        self.assertEqual(indoor["litevggt_target_size"], 420)
        self.assertAlmostEqual(indoor["litevggt_keep_ratio"], 0.46)
        self.assertEqual(indoor["preview_max_points"], 3_200_000)
        self.assertEqual(indoor["litevggt_point_selection_strategy"], "scene_coverage")
        self.assertAlmostEqual(indoor["litevggt_spatial_keep_quantile"], 0.995)
        self.assertAlmostEqual(indoor["preview_fixed_splat_radius_scale"], 0.14)
        self.assertAlmostEqual(indoor["preview_fixed_splat_opacity"], 0.46)

        self.assertEqual(outdoor["preview_image_max_side"], 1280)
        self.assertEqual(outdoor["litevggt_target_size"], 448)
        self.assertAlmostEqual(outdoor["litevggt_keep_ratio"], 0.55)
        self.assertEqual(outdoor["preview_max_points"], 5_000_000)
        self.assertEqual(outdoor["litevggt_point_selection_strategy"], "scene_coverage")
        self.assertAlmostEqual(outdoor["litevggt_axis_trim_low_quantile"], 0.0)
        self.assertAlmostEqual(outdoor["litevggt_axis_trim_high_quantile"], 0.999)
        self.assertAlmostEqual(outdoor["litevggt_spatial_keep_quantile"], 0.999)
        self.assertAlmostEqual(outdoor["preview_fixed_splat_radius_scale"], 0.10)
        self.assertAlmostEqual(outdoor["preview_fixed_splat_opacity"], 0.42)

    def test_video_speed_defaults_replace_non_request_litevggt_values(self) -> None:
        adjusted = apply_litevggt_video_speed_defaults(
            {
                "scene_type": "indoor",
                "preview_image_max_side": 768,
                "preview_image_jpeg_quality": 90,
                "litevggt_keep_ratio": 0.46,
                "preview_max_points": 3_200_000,
                "litevggt_max_input_frames": None,
                "litevggt_target_size": 420,
                "litevggt_parameter_sources": {
                    "preview_image_max_side": "system_default",
                    "litevggt_target_size": "system_default",
                },
            }
        )

        self.assertEqual(adjusted["preview_image_max_side"], 336)
        self.assertEqual(adjusted["preview_image_jpeg_quality"], 82)
        self.assertAlmostEqual(adjusted["preview_video_fps"], 1.5)
        self.assertEqual(adjusted["preview_video_max_frames"], 48)
        self.assertAlmostEqual(adjusted["litevggt_keep_ratio"], 0.20)
        self.assertEqual(adjusted["preview_max_points"], 1_200_000)
        self.assertEqual(adjusted["litevggt_max_input_frames"], 48)
        self.assertEqual(adjusted["litevggt_target_size"], 280)
        self.assertIsNone(adjusted["litevggt_depth_conf_thresh"])
        self.assertEqual(adjusted["litevggt_inference_mode"], "single")
        self.assertEqual(adjusted["litevggt_chunk_size"], 48)
        self.assertEqual(adjusted["litevggt_overlap"], 8)
        self.assertFalse(adjusted["litevggt_loop_closure"])
        self.assertEqual(adjusted["litevggt_point_selection_strategy"], "scene_coverage")
        self.assertAlmostEqual(adjusted["litevggt_axis_trim_low_quantile"], 0.01)
        self.assertAlmostEqual(adjusted["litevggt_axis_trim_high_quantile"], 0.985)
        self.assertAlmostEqual(adjusted["litevggt_spatial_keep_quantile"], 0.99)
        self.assertAlmostEqual(adjusted["preview_fixed_splat_radius_scale"], 0.13)
        self.assertAlmostEqual(adjusted["preview_fixed_splat_opacity"], 0.42)
        self.assertEqual(adjusted["litevggt_parameter_sources"]["litevggt_target_size"], "video_speed_default")

    def test_outdoor_video_speed_defaults_keep_wide_ranges_and_smaller_splats(self) -> None:
        adjusted = apply_litevggt_video_speed_defaults(
            {
                "preview_scene_profile": "outdoor_fast_clean",
                "litevggt_keep_ratio": 0.55,
                "preview_max_points": 5_000_000,
                "litevggt_target_size": 448,
            }
        )

        self.assertAlmostEqual(adjusted["preview_video_fps"], 1.5)
        self.assertEqual(adjusted["preview_video_max_frames"], 48)
        self.assertEqual(adjusted["litevggt_max_input_frames"], 48)
        self.assertEqual(adjusted["litevggt_target_size"], 308)
        self.assertAlmostEqual(adjusted["litevggt_keep_ratio"], 0.22)
        self.assertEqual(adjusted["preview_max_points"], 1_500_000)
        self.assertEqual(adjusted["litevggt_inference_mode"], "single")
        self.assertFalse(adjusted["litevggt_loop_closure"])
        self.assertEqual(adjusted["litevggt_point_selection_strategy"], "scene_coverage")
        self.assertAlmostEqual(adjusted["litevggt_axis_trim_low_quantile"], 0.002)
        self.assertAlmostEqual(adjusted["litevggt_axis_trim_high_quantile"], 0.998)
        self.assertAlmostEqual(adjusted["litevggt_spatial_keep_quantile"], 0.998)
        self.assertAlmostEqual(adjusted["preview_fixed_splat_radius_scale"], 0.11)
        self.assertAlmostEqual(adjusted["preview_fixed_splat_opacity"], 0.40)

    def test_quality_video_presets_document_slow_high_coverage_tradeoff(self) -> None:
        indoor = LITEVGGT_VIDEO_QUALITY_DEFAULTS_BY_SCENE["indoor"]
        outdoor = LITEVGGT_VIDEO_QUALITY_DEFAULTS_BY_SCENE["outdoor"]

        self.assertEqual(indoor["preview_video_max_frames"], 96)
        self.assertEqual(indoor["litevggt_max_input_frames"], 96)
        self.assertEqual(indoor["litevggt_inference_mode"], "windowed")
        self.assertEqual(indoor["litevggt_overlap"], 16)
        self.assertTrue(indoor["litevggt_loop_closure"])

        self.assertEqual(outdoor["preview_video_max_frames"], 128)
        self.assertIsNone(outdoor["litevggt_max_input_frames"])
        self.assertEqual(outdoor["litevggt_inference_mode"], "windowed")
        self.assertEqual(outdoor["litevggt_overlap"], 24)
        self.assertFalse(outdoor["litevggt_loop_closure"])
        self.assertEqual(outdoor["litevggt_keyframe_target"], 80)
        self.assertAlmostEqual(outdoor["litevggt_window_voxel_diag_ratio"], 1 / 900)
        self.assertAlmostEqual(outdoor["litevggt_final_voxel_diag_ratio"], 1 / 1000)

    def test_video_speed_defaults_replace_stale_non_request_300_frame_cap(self) -> None:
        adjusted = apply_litevggt_video_speed_defaults(
            {
                "scene_type": "outdoor",
                "preview_video_max_frames": 300,
                "litevggt_max_input_frames": 300,
                "litevggt_parameter_sources": {
                    "preview_video_max_frames": "system_default",
                    "litevggt_max_input_frames": "system_default",
                },
            }
        )

        self.assertEqual(adjusted["preview_video_max_frames"], 48)
        self.assertEqual(adjusted["litevggt_max_input_frames"], 48)

    def test_video_speed_defaults_preserve_request_overrides(self) -> None:
        adjusted = apply_litevggt_video_speed_defaults(
            {
                "preview_image_max_side": 512,
                "preview_video_max_frames": 48,
                "litevggt_target_size": 280,
                "litevggt_parameter_sources": {
                    "preview_image_max_side": "request",
                    "preview_video_max_frames": "request",
                    "litevggt_target_size": "request",
                },
            }
        )

        self.assertEqual(adjusted["preview_image_max_side"], 512)
        self.assertEqual(adjusted["preview_video_max_frames"], 48)
        self.assertEqual(adjusted["litevggt_target_size"], 280)
        self.assertAlmostEqual(adjusted["litevggt_keep_ratio"], 0.20)
        self.assertEqual(adjusted["litevggt_max_input_frames"], 48)

    @unittest.skipIf(_resolve_video_sample_fps is None, f"video preprocess dependencies unavailable: {VIDEO_PREPROCESS_IMPORT_ERROR}")
    def test_auto_video_fps_targets_frame_cap_without_starving_short_clips(self) -> None:
        assert _resolve_video_sample_fps is not None
        self.assertAlmostEqual(_resolve_video_sample_fps(None, 640.0, 64, 8), 0.1)
        self.assertAlmostEqual(_resolve_video_sample_fps(None, 10.0, 64, 8), 2.0)
        self.assertAlmostEqual(_resolve_video_sample_fps(None, 4.0, 64, 8), 2.0)
        self.assertAlmostEqual(_resolve_video_sample_fps(0.25, 640.0, 64, 8), 0.25)


if __name__ == "__main__":
    unittest.main()
