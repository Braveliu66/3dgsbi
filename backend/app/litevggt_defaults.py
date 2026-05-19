from __future__ import annotations

from copy import deepcopy
from typing import Any


LITEVGGT_OFFICIAL_PAD_PRESET = "scene_defaults_2026_05_18_v2"
LITEVGGT_DEFAULTS_PRESET_KEY = "_litevggt_defaults_preset"
LITEVGGT_DEFAULT_PREVIEW_SCENE_PROFILE = "indoor_full"

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
    "litevggt_keep_ratio": 0.46,
    "preview_max_points": 3_200_000,
    "litevggt_max_input_frames": None,
    "litevggt_target_size": 420,
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
    "litevggt_axis_trim_low_quantile": 0.005,
    "litevggt_axis_trim_high_quantile": 0.992,
    "litevggt_spatial_keep_quantile": 0.995,
    "preview_fixed_splat_radius_scale": 0.14,
    "preview_fixed_splat_opacity": 0.46,
}

LITEVGGT_OUTDOOR_RUNTIME_DEFAULTS: dict[str, Any] = {
    "litevggt_keep_ratio": 0.55,
    "preview_max_points": 5_000_000,
    "litevggt_max_input_frames": None,
    "litevggt_target_size": 448,
    "litevggt_frame_stride": None,
    "litevggt_depth_conf_thresh": None,
    "litevggt_preprocess_mode": "pad",
    "litevggt_inference_mode": "auto",
    "litevggt_chunk_size": 48,
    "litevggt_overlap": 16,
    "litevggt_loop_closure": False,
    "litevggt_keyframe_target": None,
    "litevggt_min_frame_gap": 2,
    "litevggt_min_scene_change": 0.035,
    "litevggt_window_voxel_diag_ratio": 1 / 900,
    "litevggt_final_voxel_diag_ratio": 1 / 1000,
    "litevggt_point_selection_strategy": "scene_coverage",
    "litevggt_axis_trim_low_quantile": 0.0,
    "litevggt_axis_trim_high_quantile": 0.999,
    "litevggt_spatial_keep_quantile": 0.999,
    "preview_fixed_splat_radius_scale": 0.10,
    "preview_fixed_splat_opacity": 0.42,
}

LITEVGGT_PREVIEW_IMAGE_DEFAULTS_BY_SCENE: dict[str, dict[str, Any]] = {
    "indoor": {
        "preview_image_max_side": 1024,
        "preview_image_jpeg_quality": 88,
    },
    "outdoor": {
        "preview_image_max_side": 1280,
        "preview_image_jpeg_quality": 86,
    },
}

LITEVGGT_SCENE_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "mixed_balanced": deepcopy(LITEVGGT_OFFICIAL_RUNTIME_DEFAULTS),
    "indoor_full": deepcopy(LITEVGGT_INDOOR_RUNTIME_DEFAULTS),
    "outdoor_fast_clean": deepcopy(LITEVGGT_OUTDOOR_RUNTIME_DEFAULTS),
}

LITEVGGT_VIDEO_SPEED_DEFAULTS_BY_SCENE: dict[str, dict[str, Any]] = {
    "indoor": {
        "preview_image_max_side": 336,
        "preview_image_jpeg_quality": 82,
        "preview_video_fps": 3,
        "preview_video_max_frames": 256,
        "litevggt_keep_ratio": 0.77,
        "preview_max_points": 2_200_000,
        "litevggt_max_input_frames": 256,
        "litevggt_target_size": 280,
        "litevggt_frame_stride": None,
        "litevggt_depth_conf_thresh": None,
        "litevggt_preprocess_mode": "pad",
        "litevggt_inference_mode": "single",
        "litevggt_chunk_size": 256,
        "litevggt_overlap": 8,
        "litevggt_loop_closure": True,
        "litevggt_keyframe_target": None,
        "litevggt_min_frame_gap": 2,
        "litevggt_min_scene_change": 0.055,
        "litevggt_window_voxel_diag_ratio": 1 / 260,
        "litevggt_final_voxel_diag_ratio": 1 / 250,
        "litevggt_point_selection_strategy": "scene_coverage",
        "litevggt_axis_trim_low_quantile": 0.01,
        "litevggt_axis_trim_high_quantile": 0.985,
        "litevggt_spatial_keep_quantile": 0.97,
        "preview_fixed_splat_radius_scale": 0.13,
        "preview_fixed_splat_opacity": 0.42,
    },
    "outdoor": {
        "preview_image_max_side": 336,
        "preview_image_jpeg_quality": 82,
        "preview_video_fps": 3,
        "preview_video_max_frames": 256,
        "litevggt_keep_ratio": 0.77,
        "preview_max_points": 2_500_000,
        "litevggt_max_input_frames": 256,
        "litevggt_target_size": 308,
        "litevggt_frame_stride": None,
        "litevggt_depth_conf_thresh": None,
        "litevggt_preprocess_mode": "pad",
        "litevggt_inference_mode": "single",
        "litevggt_chunk_size": 256,
        "litevggt_overlap": 16,
        "litevggt_loop_closure": True,
        "litevggt_keyframe_target": None,
        "litevggt_min_frame_gap": 2,
        "litevggt_min_scene_change": 0.035,
        "litevggt_window_voxel_diag_ratio": 1 / 450,
        "litevggt_final_voxel_diag_ratio": 1 / 250,
        "litevggt_point_selection_strategy": "scene_coverage",
        "litevggt_axis_trim_low_quantile": 0.002,
        "litevggt_axis_trim_high_quantile": 0.998,
        "litevggt_spatial_keep_quantile": 0.998,
        "preview_fixed_splat_radius_scale": 0.11,
        "preview_fixed_splat_opacity": 0.40,
    },
}

LITEVGGT_VIDEO_QUALITY_DEFAULTS_BY_SCENE: dict[str, dict[str, Any]] = {
    "indoor": {
        **deepcopy(LITEVGGT_VIDEO_SPEED_DEFAULTS_BY_SCENE["indoor"]),
        "preview_video_fps": 5.0,
        "preview_video_max_frames": 96,
        "litevggt_keep_ratio": 0.30,
        "preview_max_points": 2_400_000,
        "litevggt_max_input_frames": 96,
        "litevggt_target_size": 336,
        "litevggt_inference_mode": "windowed",
        "litevggt_overlap": 16,
        "litevggt_loop_closure": True,
        "litevggt_axis_trim_low_quantile": 0.005,
        "litevggt_axis_trim_high_quantile": 0.992,
        "litevggt_spatial_keep_quantile": 0.995,
        "preview_fixed_splat_radius_scale": 0.11,
        "preview_fixed_splat_opacity": 0.40,
    },
    "outdoor": {
        **deepcopy(LITEVGGT_VIDEO_SPEED_DEFAULTS_BY_SCENE["outdoor"]),
        "preview_video_fps": 3.0,
        "preview_video_max_frames": 128,
        "litevggt_keep_ratio": 0.78,
        "preview_max_points": 3_500_000,
        "litevggt_max_input_frames": None,
        "litevggt_target_size": 392,
        "litevggt_inference_mode": "single",
        "litevggt_overlap": 8,
        "litevggt_loop_closure": False,
        "litevggt_keyframe_target": 80,
        "litevggt_min_frame_gap": 2,
        "litevggt_min_scene_change": 0.035,
        "litevggt_window_voxel_diag_ratio": 1 / 900,
        "litevggt_final_voxel_diag_ratio": 1 / 1000,
        "litevggt_axis_trim_low_quantile": 0.0,
        "litevggt_axis_trim_high_quantile": 0.999,
        "litevggt_spatial_keep_quantile": 0.999,
        "preview_fixed_splat_radius_scale": 0.10,
        "preview_fixed_splat_opacity": 0.40,
    },
}

LITEVGGT_VIDEO_SPEED_DEFAULTS: dict[str, Any] = deepcopy(LITEVGGT_VIDEO_SPEED_DEFAULTS_BY_SCENE["indoor"])

LITEVGGT_VIDEO_QUALITY_DEFAULTS: dict[str, Any] = deepcopy(LITEVGGT_VIDEO_QUALITY_DEFAULTS_BY_SCENE["indoor"])


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


def apply_litevggt_video_speed_defaults(options: dict[str, Any] | None) -> dict[str, Any]:
    adjusted = dict(options or {})
    sources = adjusted.get("litevggt_parameter_sources")
    parameter_sources = dict(sources) if isinstance(sources, dict) else {}
    scene = _litevggt_scene_from_options(adjusted)
    video_defaults = LITEVGGT_VIDEO_SPEED_DEFAULTS_BY_SCENE[scene]

    for key, value in video_defaults.items():
        if parameter_sources.get(key) == "request":
            continue
        adjusted[key] = value
        parameter_sources[key] = "video_speed_default"

    adjusted["litevggt_parameter_sources"] = parameter_sources
    return adjusted


def _litevggt_scene_from_options(options: dict[str, Any]) -> str:
    raw = options.get("scene_type") or options.get("preview_scene_type") or options.get("preview_scene_profile")
    value = str(raw or "indoor").strip().lower()
    return "outdoor" if value in {"outdoor", "outside", "outdoor_fast_clean"} else "indoor"
