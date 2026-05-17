from __future__ import annotations

from copy import deepcopy
from typing import Any


LITEVGGT_OFFICIAL_PAD_PRESET = "scene_defaults_2026_05_17_v1"
LITEVGGT_DEFAULTS_PRESET_KEY = "_litevggt_defaults_preset"
LITEVGGT_DEFAULT_PREVIEW_SCENE_PROFILE = "mixed_balanced"

LITEVGGT_OFFICIAL_RUNTIME_DEFAULTS: dict[str, Any] = {
    "litevggt_keep_ratio": 0.42,
    "preview_max_points": 15_000_000,
    "litevggt_max_input_frames": None,
    "litevggt_target_size": 518,
    "litevggt_frame_stride": None,
    "litevggt_depth_conf_thresh": None,
    "litevggt_preprocess_mode": "pad",
    "litevggt_inference_mode": "single",
    "litevggt_chunk_size": 64,
    "litevggt_overlap": 16,
    "litevggt_loop_closure": False,
    "litevggt_keyframe_target": None,
    "litevggt_min_frame_gap": 1,
    "litevggt_min_scene_change": 0.0,
    "litevggt_window_voxel_diag_ratio": 0.0,
    "litevggt_final_voxel_diag_ratio": 0.0,
    "litevggt_point_selection_strategy": "global_confidence",
    "litevggt_axis_trim_low_quantile": 0.0,
    "litevggt_axis_trim_high_quantile": 1.0,
    "litevggt_spatial_keep_quantile": 1.0,
}

LITEVGGT_INDOOR_RUNTIME_DEFAULTS: dict[str, Any] = {
    "litevggt_keep_ratio": 0.38,
    "preview_max_points": 2_200_000,
    "litevggt_max_input_frames": None,
    "litevggt_target_size": 336,
    "litevggt_frame_stride": None,
    "litevggt_depth_conf_thresh": None,
    "litevggt_preprocess_mode": "pad",
    "litevggt_inference_mode": "auto",
    "litevggt_chunk_size": 48,
    "litevggt_overlap": 8,
    "litevggt_loop_closure": True,
    "litevggt_keyframe_target": None,
    "litevggt_min_frame_gap": 2,
    "litevggt_min_scene_change": 0.055,
    "litevggt_window_voxel_diag_ratio": 1 / 600,
    "litevggt_final_voxel_diag_ratio": 1 / 650,
    "litevggt_point_selection_strategy": "scene_coverage",
    "litevggt_axis_trim_low_quantile": 0.015,
    "litevggt_axis_trim_high_quantile": 0.94,
    "litevggt_spatial_keep_quantile": 0.97,
    "preview_fixed_splat_radius_scale": 0.20,
    "preview_fixed_splat_opacity": 0.58,
}

LITEVGGT_OUTDOOR_RUNTIME_DEFAULTS: dict[str, Any] = {
    "litevggt_keep_ratio": 0.18,
    "preview_max_points": 1_500_000,
    "litevggt_max_input_frames": None,
    "litevggt_target_size": 320,
    "litevggt_frame_stride": None,
    "litevggt_depth_conf_thresh": None,
    "litevggt_preprocess_mode": "pad",
    "litevggt_inference_mode": "auto",
    "litevggt_chunk_size": 48,
    "litevggt_overlap": 4,
    "litevggt_loop_closure": False,
    "litevggt_keyframe_target": None,
    "litevggt_min_frame_gap": 3,
    "litevggt_min_scene_change": 0.09,
    "litevggt_window_voxel_diag_ratio": 1 / 320,
    "litevggt_final_voxel_diag_ratio": 1 / 360,
    "litevggt_point_selection_strategy": "global_confidence",
    "litevggt_axis_trim_low_quantile": 0.01,
    "litevggt_axis_trim_high_quantile": 0.985,
    "litevggt_spatial_keep_quantile": 0.965,
    "preview_fixed_splat_radius_scale": 0.18,
    "preview_fixed_splat_opacity": 0.55,
}

LITEVGGT_PREVIEW_IMAGE_DEFAULTS_BY_SCENE: dict[str, dict[str, Any]] = {
    "indoor": {
        "preview_image_max_side": 768,
        "preview_image_jpeg_quality": 88,
    },
    "outdoor": {
        "preview_image_max_side": 640,
        "preview_image_jpeg_quality": 86,
    },
}

LITEVGGT_SCENE_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "mixed_balanced": deepcopy(LITEVGGT_OFFICIAL_RUNTIME_DEFAULTS),
    "indoor_full": deepcopy(LITEVGGT_INDOOR_RUNTIME_DEFAULTS),
    "outdoor_fast_clean": deepcopy(LITEVGGT_OUTDOOR_RUNTIME_DEFAULTS),
}


def litevggt_system_defaults(scene_type: str) -> dict[str, Any]:
    scene = "outdoor" if str(scene_type).strip().lower() == "outdoor" else "indoor"
    profile = "outdoor_fast_clean" if scene == "outdoor" else "indoor_full"
    return {
        "scene_type": scene,
        "preview_scene_profile": profile,
        **deepcopy(LITEVGGT_PREVIEW_IMAGE_DEFAULTS_BY_SCENE[scene]),
        **deepcopy(LITEVGGT_SCENE_PROFILE_DEFAULTS[profile]),
    }


def litevggt_stored_defaults(options: dict[str, Any]) -> dict[str, Any]:
    return {**dict(options or {}), LITEVGGT_DEFAULTS_PRESET_KEY: LITEVGGT_OFFICIAL_PAD_PRESET}


def litevggt_effective_saved_defaults(options: dict[str, Any] | None) -> dict[str, Any]:
    stored = dict(options or {})
    if stored.get(LITEVGGT_DEFAULTS_PRESET_KEY) != LITEVGGT_OFFICIAL_PAD_PRESET:
        return {}
    stored.pop(LITEVGGT_DEFAULTS_PRESET_KEY, None)
    return stored
