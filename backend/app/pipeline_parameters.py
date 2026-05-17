from __future__ import annotations

from copy import deepcopy
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.fine.fastgs_defaults import (
    FASTGS_DATA_DEVICE,
    FASTGS_DENSE,
    FASTGS_DENSIFICATION_INTERVAL,
    FASTGS_DENSIFY_FROM_ITER,
    FASTGS_DENSIFY_UNTIL_ITER,
    FASTGS_FEATURE_LR,
    FASTGS_GRAD_ABS_THRESH,
    FASTGS_GRAD_THRESH,
    FASTGS_HIGHFEATURE_LR,
    FASTGS_LAMBDA_DSSIM,
    FASTGS_LATE_PRUNE_ENABLED,
    FASTGS_LATE_PRUNE_FROM_ITER,
    FASTGS_LATE_PRUNE_INTERVAL,
    FASTGS_LATE_PRUNE_MAX_FRACTION,
    FASTGS_LATE_PRUNE_MAX_WORLD_SCALE_RATIO,
    FASTGS_LATE_PRUNE_MIN_OPACITY,
    FASTGS_LATE_PRUNE_SCORE_THRESH,
    FASTGS_LATE_PRUNE_UNTIL_ITER,
    FASTGS_LOSS_THRESH,
    FASTGS_LOWFEATURE_LR,
    FASTGS_MULT,
    FASTGS_OPACITY_LR,
    FASTGS_OPACITY_RESET_INTERVAL,
    FASTGS_PERCENT_DENSE,
    FASTGS_POSITION_LR_DELAY_MULT,
    FASTGS_POSITION_LR_FINAL,
    FASTGS_POSITION_LR_INIT,
    FASTGS_ROTATION_LR,
    FASTGS_SAMPLE_CAMERAS,
    FASTGS_SCALING_LR,
    FASTGS_SHFEATURE_LR,
    FASTGS_SIZE_PRUNE_FROM_ITER,
    FASTGS_SIZE_PRUNE_MAX_SCREEN_SIZE,
    FASTGS_SIZE_PRUNE_MAX_WORLD_SCALE_RATIO,
    FASTGS_DEBLUR_AUTO_SCHEDULE,
    FASTGS_DEBLUR_BLURRED_VIEWS_ONLY,
    FASTGS_DEBLUR_ENABLED,
    FASTGS_DEBLUR_EXTRA_POINTS_ENABLED,
    FASTGS_DEBLUR_GTNET_LR,
    FASTGS_DEBLUR_HIDDEN,
    FASTGS_DEBLUR_LAMBDA_P,
    FASTGS_DEBLUR_LAMBDA_S,
    FASTGS_DEBLUR_LATE_DENSIFY_ENABLED,
    FASTGS_DEBLUR_MAX_CLAMP,
    FASTGS_DEBLUR_MAX_POSITION_DELTA,
    FASTGS_DEBLUR_MODE,
    FASTGS_DEBLUR_SCHEDULE_PROFILE,
    FASTGS_DEBLUR_SHARP_REFINE_CLEAR_ONLY,
    FASTGS_DEBLUR_SHARP_REFINE_ENABLED,
    FASTGS_DEBLUR_SHARP_REFINE_FROM_ITER,
    FASTGS_DEBLUR_TOPOLOGY_SHARP_ONLY,
    FASTGS_DEBLUR_TRANSFORM_REG_WEIGHT,
    FASTGS_DEBLUR_WIDTH,
    FASTGS_DEBLUR_XYZ_LR_SCALE,
    FINE_IMAGE_MAX_SIDE,
    FINE_ITERATIONS,
)
from app.models import PipelineParameterDefault


VALID_PIPELINES = {"litevggt_spz", "lingbot_video_pointcloud_fast", "official_fastgs_big"}
VALID_SCENE_TYPES = {"indoor", "outdoor"}

SCENE_PROFILES = {
    "indoor": {"preview_scene_profile": "indoor_full", "fine_scene_profile": "indoor_full"},
    "outdoor": {"preview_scene_profile": "outdoor_fast_clean", "fine_scene_profile": "outdoor_fast_clean"},
}

LITEVGGT_DEFAULTS: dict[str, dict[str, Any]] = {
    "indoor": {
        "scene_type": "indoor",
        "preview_scene_profile": "indoor_full",
        "preview_image_max_side": 518,
        "preview_image_jpeg_quality": 90,
        "litevggt_keep_ratio": 0.55,
        "preview_max_points": 3_000_000,
        "litevggt_max_input_frames": 240,
        "litevggt_target_size": 240,
        "litevggt_frame_stride": None,
        "litevggt_depth_conf_thresh": None,
        "litevggt_preprocess_mode": "pad",
        "litevggt_inference_mode": "single",
        "litevggt_chunk_size": 48,
        "litevggt_overlap": 8,
        "litevggt_loop_closure": True,
        "litevggt_keyframe_target": None,
        "litevggt_min_frame_gap": 2,
        "litevggt_min_scene_change": 0.065,
        "litevggt_window_voxel_diag_ratio": 1 / 700,
        "litevggt_final_voxel_diag_ratio": 1 / 700,
        "litevggt_point_selection_strategy": "scene_coverage",
        "litevggt_axis_trim_low_quantile": 0.02,
        "litevggt_axis_trim_high_quantile": 0.90,
        "litevggt_spatial_keep_quantile": 0.975,
        "preview_fixed_splat_radius_scale": 0.22,
        "preview_fixed_splat_opacity": 0.55,
    },
    "outdoor": {
        "scene_type": "outdoor",
        "preview_scene_profile": "outdoor_fast_clean",
        "preview_image_max_side": 518,
        "preview_image_jpeg_quality": 90,
        "litevggt_keep_ratio": 0.25,
        "preview_max_points": 4_000_000,
        "litevggt_max_input_frames": 240,
        "litevggt_target_size": 476,
        "litevggt_frame_stride": None,
        "litevggt_depth_conf_thresh": None,
        "litevggt_preprocess_mode": "pad",
        "litevggt_inference_mode": "single",
        "litevggt_chunk_size": 48,
        "litevggt_overlap": 8,
        "litevggt_loop_closure": True,
        "litevggt_keyframe_target": 240,
        "litevggt_min_frame_gap": 3,
        "litevggt_min_scene_change": 3.0,
        "litevggt_window_voxel_diag_ratio": 1 / 350,
        "litevggt_final_voxel_diag_ratio": 1 / 450,
        "litevggt_point_selection_strategy": "global_confidence",
        "litevggt_axis_trim_low_quantile": 0.001,
        "litevggt_axis_trim_high_quantile": 0.995,
        "litevggt_spatial_keep_quantile": 0.99,
        "preview_fixed_splat_radius_scale": 0.22,
        "preview_fixed_splat_opacity": 0.55,
    },
}

LINGBOT_DEFAULTS: dict[str, dict[str, Any]] = {
    "indoor": {
        "scene_type": "indoor",
        "preview_lingbot_fps": 10.0,
        "preview_lingbot_image_size": 518,
        "preview_lingbot_target_width": 518,
        "preview_lingbot_target_height": 378,
        "preview_lingbot_window_size": 128,
        "preview_lingbot_keyframe_interval": 1,
        "preview_lingbot_overlap_keyframes": 16,
        "preview_lingbot_num_scale_frames": 8,
        "preview_lingbot_camera_iterations": 4,
        "preview_lingbot_camera_iterations_retry": 4,
        "preview_lingbot_pixel_stride_fast": 4,
        "preview_lingbot_pixel_stride_full": 2,
        "preview_lingbot_conf_percentile_fast": 55.0,
        "preview_lingbot_conf_percentile_full": 40.0,
        "preview_lingbot_min_conf": 1.2,
        "preview_lingbot_use_sdpa": False,
        "preview_lingbot_allow_sdpa_fallback": False,
        "preview_lingbot_compile": False,
        "preview_lingbot_voxel_target_fast": 4200,
        "preview_lingbot_voxel_target_full": 7000,
        "preview_lingbot_coverage_keyframes": True,
        "preview_lingbot_mask_sky": False,
    },
    "outdoor": {
        "scene_type": "outdoor",
        "preview_lingbot_fps": 10.0,
        "preview_lingbot_image_size": 518,
        "preview_lingbot_target_width": 518,
        "preview_lingbot_target_height": 378,
        "preview_lingbot_window_size": 128,
        "preview_lingbot_keyframe_interval": 2,
        "preview_lingbot_overlap_keyframes": 16,
        "preview_lingbot_num_scale_frames": 8,
        "preview_lingbot_camera_iterations": 4,
        "preview_lingbot_camera_iterations_retry": 4,
        "preview_lingbot_pixel_stride_fast": 6,
        "preview_lingbot_pixel_stride_full": 4,
        "preview_lingbot_conf_percentile_fast": 70.0,
        "preview_lingbot_conf_percentile_full": 55.0,
        "preview_lingbot_min_conf": 1.5,
        "preview_lingbot_use_sdpa": False,
        "preview_lingbot_allow_sdpa_fallback": False,
        "preview_lingbot_compile": False,
        "preview_lingbot_voxel_target_fast": 3000,
        "preview_lingbot_voxel_target_full": 5200,
        "preview_lingbot_coverage_keyframes": True,
        "preview_lingbot_mask_sky": True,
    },
}

FASTGS_DEFAULTS: dict[str, dict[str, Any]] = {
    scene_type: {
        "scene_type": scene_type,
        "fine_scene_type": scene_type,
        "fine_scene_profile": SCENE_PROFILES[scene_type]["fine_scene_profile"],
        "fine_image_max_side": FINE_IMAGE_MAX_SIDE,
        "fine_iterations": FINE_ITERATIONS,
        "fine_train_resolution": FINE_IMAGE_MAX_SIDE,
        "fine_data_device": FASTGS_DATA_DEVICE,
        "fine_position_lr_init": FASTGS_POSITION_LR_INIT,
        "fine_position_lr_final": FASTGS_POSITION_LR_FINAL,
        "fine_position_lr_delay_mult": FASTGS_POSITION_LR_DELAY_MULT,
        "fine_position_lr_max_steps": FINE_ITERATIONS,
        "fine_feature_lr": FASTGS_FEATURE_LR,
        "fine_shfeature_lr": FASTGS_SHFEATURE_LR,
        "fine_highfeature_lr": FASTGS_HIGHFEATURE_LR,
        "fine_lowfeature_lr": FASTGS_LOWFEATURE_LR,
        "fine_opacity_lr": FASTGS_OPACITY_LR,
        "fine_scaling_lr": FASTGS_SCALING_LR,
        "fine_rotation_lr": FASTGS_ROTATION_LR,
        "fine_percent_dense": FASTGS_PERCENT_DENSE,
        "fine_grad_thresh": FASTGS_GRAD_THRESH,
        "fine_grad_abs_thresh": FASTGS_GRAD_ABS_THRESH,
        "fine_densify_grad_threshold": 0.0002,
        "fine_densification_interval": FASTGS_DENSIFICATION_INTERVAL,
        "fine_densify_from_iter": FASTGS_DENSIFY_FROM_ITER,
        "fine_densify_until_iter": FASTGS_DENSIFY_UNTIL_ITER,
        "fine_opacity_reset_interval": FASTGS_OPACITY_RESET_INTERVAL,
        "fine_dense": FASTGS_DENSE,
        "fine_mult": FASTGS_MULT,
        "fine_lambda_dssim": FASTGS_LAMBDA_DSSIM,
        "fine_fastgs_loss_thresh": FASTGS_LOSS_THRESH,
        "fine_fastgs_sample_cameras": FASTGS_SAMPLE_CAMERAS,
        "fine_fastgs_size_prune_from_iter": FASTGS_SIZE_PRUNE_FROM_ITER,
        "fine_fastgs_size_prune_max_screen_size": FASTGS_SIZE_PRUNE_MAX_SCREEN_SIZE,
        "fine_fastgs_size_prune_max_world_scale_ratio": FASTGS_SIZE_PRUNE_MAX_WORLD_SCALE_RATIO,
        "fine_fastgs_late_prune_enabled": FASTGS_LATE_PRUNE_ENABLED,
        "fine_fastgs_late_prune_interval": FASTGS_LATE_PRUNE_INTERVAL,
        "fine_fastgs_late_prune_from_iter": FASTGS_LATE_PRUNE_FROM_ITER,
        "fine_fastgs_late_prune_until_iter": FASTGS_LATE_PRUNE_UNTIL_ITER,
        "fine_fastgs_late_prune_min_opacity": FASTGS_LATE_PRUNE_MIN_OPACITY,
        "fine_fastgs_late_prune_score_thresh": FASTGS_LATE_PRUNE_SCORE_THRESH,
        "fine_fastgs_late_prune_max_world_scale_ratio": FASTGS_LATE_PRUNE_MAX_WORLD_SCALE_RATIO,
        "fine_fastgs_late_prune_max_fraction": FASTGS_LATE_PRUNE_MAX_FRACTION,
        "fine_fastgs_final_prune_min_opacity": 0.005,
        "fine_fastgs_final_prune_score_thresh": 0.90,
        "fine_fastgs_final_prune_max_world_scale_ratio": 0.10,
        "fine_deblur_enabled": FASTGS_DEBLUR_ENABLED,
        "fine_deblur_mode": FASTGS_DEBLUR_MODE,
        "fine_deblur_auto_schedule": FASTGS_DEBLUR_AUTO_SCHEDULE,
        "fine_deblur_schedule_profile": FASTGS_DEBLUR_SCHEDULE_PROFILE,
        "fine_deblur_late_densify_enabled": FASTGS_DEBLUR_LATE_DENSIFY_ENABLED,
        "fine_deblur_warmup_iters": 5_000,
        "fine_deblur_extra_points_enabled": FASTGS_DEBLUR_EXTRA_POINTS_ENABLED,
        "fine_deblur_sharp_refine_enabled": FASTGS_DEBLUR_SHARP_REFINE_ENABLED,
        "fine_deblur_sharp_refine_from_iter": FASTGS_DEBLUR_SHARP_REFINE_FROM_ITER,
        "fine_deblur_sharp_refine_clear_only": FASTGS_DEBLUR_SHARP_REFINE_CLEAR_ONLY,
        "fine_deblur_topology_sharp_only": FASTGS_DEBLUR_TOPOLOGY_SHARP_ONLY,
        "fine_deblur_num_moments": 5,
        "fine_deblur_gtnet_lr": FASTGS_DEBLUR_GTNET_LR,
        "fine_deblur_hidden": FASTGS_DEBLUR_HIDDEN,
        "fine_deblur_width": FASTGS_DEBLUR_WIDTH,
        "fine_deblur_lambda_s": FASTGS_DEBLUR_LAMBDA_S,
        "fine_deblur_lambda_p": FASTGS_DEBLUR_LAMBDA_P,
        "fine_deblur_max_clamp": FASTGS_DEBLUR_MAX_CLAMP,
        "fine_deblur_max_position_delta": FASTGS_DEBLUR_MAX_POSITION_DELTA,
        "fine_deblur_transform_reg_weight": FASTGS_DEBLUR_TRANSFORM_REG_WEIGHT,
        "fine_deblur_xyz_lr_scale": FASTGS_DEBLUR_XYZ_LR_SCALE,
        "fine_deblur_blurred_views_only": FASTGS_DEBLUR_BLURRED_VIEWS_ONLY,
    }
    for scene_type in VALID_SCENE_TYPES
}

SYSTEM_DEFAULTS = {
    "litevggt_spz": LITEVGGT_DEFAULTS,
    "lingbot_video_pointcloud_fast": LINGBOT_DEFAULTS,
    "official_fastgs_big": FASTGS_DEFAULTS,
}


def pipeline_parameter_schema() -> dict[str, Any]:
    return {
        "scene_types": [
            {"value": "indoor", "label": "室内"},
            {"value": "outdoor", "label": "户外"},
        ],
        "pipelines": [
            pipeline_schema("litevggt_spz", "LiteVGGT 图片预览", litevggt_fields()),
            pipeline_schema("lingbot_video_pointcloud_fast", "LingBot 视频预览", lingbot_fields()),
            pipeline_schema("official_fastgs_big", "FastGS-Big 精修", fastgs_fields()),
        ],
    }


def pipeline_schema(pipeline: str, label: str, fields: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pipeline": pipeline,
        "label": label,
        "defaults": deepcopy(SYSTEM_DEFAULTS[pipeline]),
        "fields": fields,
    }


def field(key: str, label: str, kind: str, group: str, description: str, **extra: Any) -> dict[str, Any]:
    return {"key": key, "label": label, "type": kind, "group": group, "description": description, **extra}


def litevggt_fields() -> list[dict[str, Any]]:
    return [
        field("preview_scene_profile", "预览场景 profile", "select", "场景策略", "控制 LiteVGGT 的场景默认策略。前端只保存 indoor_full 或 outdoor_fast_clean。", options=["indoor_full", "outdoor_fast_clean"]),
        field("preview_image_max_side", "预处理最大边长", "number", "输入预处理", "worker 下载图片后缩放到的最大边长，影响速度和显存。", min=128, max=4096, step=1),
        field("preview_image_jpeg_quality", "JPEG 质量", "number", "输入预处理", "预处理 JPEG 写出质量，越高越保留细节但文件更大。", min=1, max=100, step=1),
        field("litevggt_keep_ratio", "点云保留比例", "number", "点云输出", "按质量/覆盖策略保留的点比例，越大点越多、加载更慢。", min=0.01, max=1, step=0.01),
        field("preview_max_points", "最大点数", "number", "点云输出", "最终预览点云点数上限，直接影响 PLY/SPZ 大小和浏览器加载。", min=1, max=20_000_000, step=1000),
        field("litevggt_max_input_frames", "最大输入帧数", "nullable_number", "输入抽帧", "最多送入 LiteVGGT 的图片数；空值表示使用策略默认。", min=1, max=2000, step=1),
        field("litevggt_target_size", "模型输入尺寸", "number", "输入预处理", "LiteVGGT 输入目标尺寸，越大细节越好但显存和时间更高。", min=128, max=1024, step=1),
        field("litevggt_frame_stride", "固定抽帧步长", "nullable_number", "输入抽帧", "例如 2 表示隔帧取样；空值表示交给关键帧策略。", min=1, max=10000, step=1),
        field("litevggt_depth_conf_thresh", "深度置信度阈值", "nullable_number", "点云过滤", "丢弃低置信度深度点；空值表示使用运行时默认逻辑。", min=-100, max=100, step=0.01),
        field("litevggt_preprocess_mode", "预处理模式", "select", "输入预处理", "pad 保留完整画面，crop 裁剪到目标比例。", options=["pad", "crop"]),
        field("litevggt_inference_mode", "推理模式", "select", "推理策略", "single 一次处理全部帧；chunk/window 用于长序列或显存不足场景。", options=["single", "chunk", "window"]),
        field("litevggt_chunk_size", "chunk 帧数", "number", "推理策略", "分块推理时每块帧数，越大上下文更完整但更占显存。", min=1, max=512, step=1),
        field("litevggt_overlap", "chunk 重叠帧数", "number", "推理策略", "相邻 chunk 的重叠帧数，越大连续性更好但计算更多。", min=0, max=512, step=1),
        field("litevggt_loop_closure", "回环一致性", "boolean", "推理策略", "开启后有利于环绕拍摄/闭环场景结构一致。"),
        field("litevggt_keyframe_target", "关键帧目标数", "nullable_number", "输入抽帧", "目标关键帧数量；空值表示不额外指定。", min=1, max=2000, step=1),
        field("litevggt_min_frame_gap", "关键帧最小间隔", "number", "输入抽帧", "关键帧之间至少间隔多少帧。", min=1, max=10000, step=1),
        field("litevggt_min_scene_change", "最小场景变化", "number", "输入抽帧", "场景变化超过该阈值更可能被选为关键帧。", min=0, max=100, step=0.001),
        field("litevggt_window_voxel_diag_ratio", "窗口体素比例", "number", "点云过滤", "窗口阶段 voxel size = 场景对角线 * 该比例；越大下采样越强。", min=0, max=1, step=0.0001),
        field("litevggt_final_voxel_diag_ratio", "最终体素比例", "number", "点云过滤", "最终输出前统一体素下采样比例。", min=0, max=1, step=0.0001),
        field("litevggt_point_selection_strategy", "点选择策略", "select", "点云输出", "scene_coverage 保覆盖；global_confidence 保高置信点；per_frame 保每帧贡献。", options=["scene_coverage", "global_confidence", "per_frame"]),
        field("litevggt_axis_trim_low_quantile", "轴向低分位裁剪", "number", "点云过滤", "按 x/y/z 低分位裁剪离群点。", min=0, max=1, step=0.001),
        field("litevggt_axis_trim_high_quantile", "轴向高分位裁剪", "number", "点云过滤", "按 x/y/z 高分位裁剪离群点。", min=0, max=1, step=0.001),
        field("litevggt_spatial_keep_quantile", "空间保留分位", "number", "点云过滤", "按空间距离保留主体区域，越小裁剪越强。", min=0, max=1, step=0.001),
        field("preview_fixed_splat_radius_scale", "Splat 半径倍率", "number", "SPZ 转换", "固定 Gaussian 半径相对基础点半径的倍率。", min=0.05, max=20, step=0.01),
        field("preview_fixed_splat_opacity", "Splat 不透明度", "number", "SPZ 转换", "固定 Gaussian 输出透明度。", min=0.05, max=0.99, step=0.01),
    ]


def lingbot_fields() -> list[dict[str, Any]]:
    return [
        field("scene_type", "场景类型", "select", "场景策略", "控制 LingBot 点云预览的室内/户外默认参数。", options=["indoor", "outdoor"]),
        field("preview_lingbot_fps", "采样 FPS", "number", "视频采样", "从视频抽帧的目标帧率。", min=0.1, max=60, step=0.1),
        field("preview_lingbot_image_size", "模型图像尺寸", "number", "输入预处理", "LingBot 模型输入图像尺寸。", min=224, max=1024, step=1),
        field("preview_lingbot_target_width", "目标宽度", "number", "输入预处理", "送入运行时的目标宽度。", min=14, max=2048, step=1),
        field("preview_lingbot_target_height", "目标高度", "number", "输入预处理", "送入运行时的目标高度。", min=14, max=2048, step=1),
        field("preview_lingbot_window_size", "窗口大小", "number", "窗口推理", "每个 LingBot 滑动窗口包含的帧数。", min=8, max=512, step=1),
        field("preview_lingbot_keyframe_interval", "关键帧间隔", "number", "窗口推理", "相邻关键帧间隔，越大速度越快但连续性下降。", min=1, max=100000, step=1),
        field("preview_lingbot_overlap_keyframes", "重叠关键帧", "number", "窗口推理", "相邻窗口保留的重叠关键帧数量。", min=1, max=128, step=1),
        field("preview_lingbot_num_scale_frames", "尺度估计帧数", "number", "相机估计", "用于尺度估计的帧数。", min=1, max=64, step=1),
        field("preview_lingbot_camera_iterations", "快速相机迭代", "number", "相机估计", "快速阶段相机优化迭代次数。", min=1, max=8, step=1),
        field("preview_lingbot_camera_iterations_retry", "重试相机迭代", "number", "相机估计", "重试阶段相机优化迭代次数。", min=1, max=8, step=1),
        field("preview_lingbot_pixel_stride_fast", "快速点云像素步长", "number", "点云输出", "快速预览点云采样步长，越大点越少。", min=1, max=512, step=1),
        field("preview_lingbot_pixel_stride_full", "完整点云像素步长", "number", "点云输出", "完整导出点云采样步长。", min=1, max=512, step=1),
        field("preview_lingbot_conf_percentile_fast", "快速置信度分位", "number", "点云过滤", "快速预览按置信度分位过滤点。", min=0, max=100, step=0.1),
        field("preview_lingbot_conf_percentile_full", "完整置信度分位", "number", "点云过滤", "完整点云按置信度分位过滤点。", min=0, max=100, step=0.1),
        field("preview_lingbot_min_conf", "最小置信度", "number", "点云过滤", "点云输出的最低置信度阈值。", min=-100, max=100, step=0.01),
        field("preview_lingbot_use_sdpa", "使用 SDPA", "boolean", "运行时", "是否请求 SDPA 注意力路径。"),
        field("preview_lingbot_allow_sdpa_fallback", "允许 SDPA fallback", "boolean", "运行时", "缺少 flashinfer 时是否允许降级到 SDPA fallback。"),
        field("preview_lingbot_compile", "torch compile", "boolean", "运行时", "是否启用 torch.compile。首次运行可能更慢。"),
        field("preview_lingbot_voxel_target_fast", "快速 voxel 目标", "number", "点云输出", "快速预览体素目标数量。", min=1, max=100000, step=1),
        field("preview_lingbot_voxel_target_full", "完整 voxel 目标", "number", "点云输出", "完整点云体素目标数量。", min=1, max=100000, step=1),
        field("preview_lingbot_coverage_keyframes", "覆盖关键帧", "boolean", "点云输出", "是否强制加入覆盖性更好的关键帧点。"),
        field("preview_lingbot_mask_sky", "天空过滤", "boolean", "点云过滤", "户外视频中开启可减少天空远点和噪声。"),
    ]


def fastgs_fields() -> list[dict[str, Any]]:
    return [
        field("fine_scene_profile", "精修场景 profile", "select", "场景策略", "控制 FastGS-Big 的场景默认策略。", options=["indoor_full", "outdoor_fast_clean"]),
        field("fine_image_max_side", "输入最大边长", "number", "输入预处理", "精修输入图像缩放最大边长。", min=256, max=4096, step=1),
        field("fine_iterations", "训练轮数", "number", "训练基础", "FastGS-Big 总训练轮数。", min=5000, max=60000, step=100),
        field("fine_train_resolution", "训练分辨率", "number", "训练基础", "FastGS 训练分辨率参数 -r。", min=1, max=16384, step=1),
        field("fine_data_device", "数据设备", "select", "训练基础", "训练数据放置设备。cuda 更快但显存占用更高。", options=["cuda", "cpu"]),
        field("fine_position_lr_init", "位置初始学习率", "number", "学习率", "Gaussian xyz 初始学习率。", min=1e-8, max=1, step=0.00001),
        field("fine_position_lr_final", "位置最终学习率", "number", "学习率", "训练后期位置微调学习率。", min=1e-9, max=1, step=0.000001),
        field("fine_position_lr_delay_mult", "位置 LR delay", "number", "学习率", "前期位置学习率延迟倍率。", min=0, max=1, step=0.001),
        field("fine_position_lr_max_steps", "位置 LR 衰减步数", "number", "学习率", "位置学习率衰减总步数，通常等于训练轮数。", min=1, max=100000, step=1),
        field("fine_feature_lr", "DC 特征学习率", "number", "学习率", "基础颜色特征学习率。", min=1e-7, max=1, step=0.0001),
        field("fine_shfeature_lr", "SH 特征学习率", "number", "学习率", "高阶球谐颜色特征学习率。", min=1e-7, max=1, step=0.0001),
        field("fine_highfeature_lr", "高频特征学习率", "number", "学习率", "FastGS-Big 高频特征学习率。", min=1e-7, max=1, step=0.0001),
        field("fine_lowfeature_lr", "低频特征学习率", "number", "学习率", "FastGS-Big 低频特征学习率。", min=1e-7, max=1, step=0.0001),
        field("fine_opacity_lr", "Opacity 学习率", "number", "学习率", "透明度学习率。", min=1e-7, max=1, step=0.0001),
        field("fine_scaling_lr", "Scale 学习率", "number", "学习率", "Gaussian 尺寸学习率。", min=1e-7, max=1, step=0.0001),
        field("fine_rotation_lr", "Rotation 学习率", "number", "学习率", "Gaussian 旋转学习率。", min=1e-7, max=1, step=0.0001),
        field("fine_percent_dense", "percent_dense", "number", "Densify", "3DGS/FastGS densify 场景比例参数。", min=0, max=1, step=0.0001),
        field("fine_grad_thresh", "clone 梯度阈值", "number", "Densify", "clone 类 densify 梯度阈值。", min=1e-7, max=0.1, step=0.0001),
        field("fine_grad_abs_thresh", "split 绝对梯度阈值", "number", "Densify", "split 类 densify 绝对梯度阈值。", min=1e-7, max=0.1, step=0.0001),
        field("fine_densify_grad_threshold", "densify 梯度阈值", "number", "Densify", "FastGS densify_grad_threshold。", min=1e-7, max=0.1, step=0.0001),
        field("fine_densification_interval", "densify 间隔", "number", "Densify", "每隔多少轮执行 densify 检查。", min=1, max=10000, step=1),
        field("fine_densify_from_iter", "densify 开始轮数", "number", "Densify", "从第几轮开始 densify。", min=0, max=100000, step=1),
        field("fine_densify_until_iter", "densify 结束轮数", "number", "Densify", "到第几轮停止 densify。", min=0, max=100000, step=1),
        field("fine_opacity_reset_interval", "opacity reset 间隔", "number", "Prune", "opacity reset 间隔。大值相当于禁用中途 reset。", min=1, max=100000, step=1),
        field("fine_dense", "dense", "number", "Densify", "FastGS dense 参数。", min=0, max=1, step=0.0001),
        field("fine_mult", "compact 倍率", "number", "FastGS", "FastGS compact box / splat tile 倍率。", min=0.01, max=10, step=0.01),
        field("fine_lambda_dssim", "DSSIM 权重", "number", "Loss", "总 loss 中 DSSIM 权重。", min=0, max=1, step=0.01),
        field("fine_fastgs_loss_thresh", "FastGS loss 阈值", "number", "FastGS", "VCD/VCP loss 阈值。", min=0, max=1, step=0.01),
        field("fine_fastgs_sample_cameras", "采样相机数", "number", "FastGS", "VCD/VCP 多视角 score 使用的相机数。", min=1, max=32, step=1),
        field("fine_fastgs_size_prune_from_iter", "size prune 开始轮数", "number", "Prune", "大 splat 尺寸裁剪开始轮数。", min=0, max=100000, step=1),
        field("fine_fastgs_size_prune_max_screen_size", "屏幕尺寸裁剪阈值", "number", "Prune", "大 splat 屏幕尺寸裁剪阈值。", min=1, max=10000, step=1),
        field("fine_fastgs_size_prune_max_world_scale_ratio", "世界尺寸裁剪比例", "number", "Prune", "大 splat 世界尺度裁剪比例。", min=0, max=1, step=0.01),
        field("fine_fastgs_late_prune_enabled", "启用 late prune", "boolean", "Late prune", "训练后期是否执行轻量裁剪。"),
        field("fine_fastgs_late_prune_interval", "late prune 间隔", "number", "Late prune", "late prune 执行间隔。", min=1, max=100000, step=1),
        field("fine_fastgs_late_prune_from_iter", "late prune 开始", "number", "Late prune", "late prune 开始轮数。", min=0, max=100000, step=1),
        field("fine_fastgs_late_prune_until_iter", "late prune 结束", "number", "Late prune", "late prune 结束轮数。", min=0, max=100000, step=1),
        field("fine_fastgs_late_prune_min_opacity", "late prune opacity", "number", "Late prune", "late prune 最小 opacity 阈值。", min=0.001, max=0.2, step=0.001),
        field("fine_fastgs_late_prune_score_thresh", "late prune score", "number", "Late prune", "late prune score 阈值。", min=0.5, max=1, step=0.01),
        field("fine_fastgs_late_prune_max_world_scale_ratio", "late prune 世界比例", "number", "Late prune", "late prune 世界尺度比例阈值。", min=0, max=1, step=0.01),
        field("fine_fastgs_late_prune_max_fraction", "late prune 最大比例", "number", "Late prune", "单次 late prune 最多裁剪比例。", min=0, max=1, step=0.01),
        field("fine_fastgs_final_prune_min_opacity", "final prune opacity", "number", "Final prune", "最终裁剪最小 opacity 阈值。", min=0.001, max=0.2, step=0.001),
        field("fine_fastgs_final_prune_score_thresh", "final prune score", "number", "Final prune", "最终裁剪 score 阈值。", min=0.5, max=1, step=0.01),
        field("fine_fastgs_final_prune_max_world_scale_ratio", "final prune 世界比例", "number", "Final prune", "最终裁剪世界尺度比例阈值。", min=0, max=1, step=0.01),
        field("fine_deblur_enabled", "Deblur", "deblur_switch", "Deblur", "开启保存为 auto，由后端根据模糊检测和模式自动启用；关闭保存为 false。"),
        field("fine_deblur_mode", "Deblur 模式", "select", "Deblur", "sharp 表示不做模糊建模；defocus/motion/mixed 对应失焦、运动和混合模糊。", options=["sharp", "defocus", "motion", "mixed"]),
        field("fine_deblur_auto_schedule", "自动调度", "select", "Deblur", "是否由调度器控制 warmup、deblur loss、densify、prune。", options=["true", "false"]),
        field("fine_deblur_schedule_profile", "调度 profile", "select", "Deblur", "quality 质量优先，balanced 折中，fast 速度优先。", options=["quality", "balanced", "fast"]),
        field("fine_deblur_late_densify_enabled", "后期二次 densify", "select", "Deblur", "Deblur 后期是否再额外加密一段。", options=["true", "false"]),
        field("fine_deblur_warmup_iters", "warmup 轮数", "number", "Deblur", "前多少轮不启用 Deblur，先稳定普通 FastGS 几何。", min=0, max=100000, step=1),
        field("fine_deblur_extra_points_enabled", "额外点", "select", "Deblur", "是否允许 Deblur 梯度额外克隆点。", options=["true", "false"]),
        field("fine_deblur_sharp_refine_enabled", "sharp refine", "select", "Deblur", "最后阶段是否关闭 Deblur 并用普通 Gaussian 细调。", options=["true", "false"]),
        field("fine_deblur_sharp_refine_from_iter", "sharp refine 开始", "number", "Deblur", "sharp refine 开始轮数。", min=0, max=100000, step=1),
        field("fine_deblur_sharp_refine_clear_only", "仅清晰帧 refine", "select", "Deblur", "sharp refine 阶段是否只使用清晰帧。", options=["true", "false"]),
        field("fine_deblur_topology_sharp_only", "拓扑用 sharp 渲染", "select", "Deblur", "densify/prune 拓扑决策是否只使用普通 sharp render。", options=["true", "false"]),
        field("fine_deblur_num_moments", "motion moments", "number", "Deblur", "motion/mixed 模式位置扰动 moments 数量。", min=1, max=8, step=1),
        field("fine_deblur_gtnet_lr", "GTnet 学习率", "number", "Deblur", "GTnet MLP 学习率。", min=0.000001, max=0.1, step=0.0001),
        field("fine_deblur_hidden", "GTnet 隐藏层", "number", "Deblur", "GTnet MLP 隐藏层数。", min=1, max=8, step=1),
        field("fine_deblur_width", "GTnet 宽度", "number", "Deblur", "GTnet MLP 每层宽度。", min=16, max=256, step=1),
        field("fine_deblur_lambda_s", "scale blur 强度", "number", "Deblur", "scale/covariance blur 强度。", min=0, max=0.1, step=0.001),
        field("fine_deblur_lambda_p", "position blur 强度", "number", "Deblur", "position/motion blur 强度。", min=0, max=0.1, step=0.001),
        field("fine_deblur_max_clamp", "scale clamp", "number", "Deblur", "GTnet 对 scale/rotation delta 的最大 clamp。", min=1, max=1.8, step=0.01),
        field("fine_deblur_max_position_delta", "最大位置位移", "number", "Deblur", "position delta 最大位移。", min=0, max=1, step=0.001),
        field("fine_deblur_transform_reg_weight", "transform 正则", "number", "Deblur", "GTnet transform 正则权重。", min=0, max=1, step=0.0001),
        field("fine_deblur_xyz_lr_scale", "Deblur xyz LR 缩放", "number", "Deblur", "Deblur 阶段位置学习率缩放。", min=0, max=1, step=0.01),
        field("fine_deblur_blurred_views_only", "仅模糊帧 Deblur", "select", "Deblur", "是否只让 registry 中 blurred=true 的图片走 Deblur。", options=["true", "false"]),
    ]


def normalize_parameter_scene_type(value: Any, default: str = "indoor") -> str:
    normalized = str(value or default).strip().lower()
    if normalized in {"outdoor", "outside", "outdoor_fast_clean"}:
        return "outdoor"
    if normalized in {"indoor", "inside", "indoor_full"}:
        return "indoor"
    return default


def system_defaults_for(pipeline: str, scene_type: str) -> dict[str, Any]:
    if pipeline not in VALID_PIPELINES:
        raise ValueError(f"Unsupported pipeline: {pipeline}")
    scene = normalize_parameter_scene_type(scene_type)
    return deepcopy(SYSTEM_DEFAULTS[pipeline][scene])


def sanitize_pipeline_options(pipeline: str, options: dict[str, Any]) -> dict[str, Any]:
    pipeline_item = next(item for item in pipeline_parameter_schema()["pipelines"] if item["pipeline"] == pipeline)
    known_keys = {item["key"] for item in pipeline_item["fields"]}
    return {key: value for key, value in (options or {}).items() if key in known_keys and is_json_scalar(value)}


def is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def saved_defaults_for(db: Session, pipeline: str, scene_type: str) -> dict[str, Any]:
    scene = normalize_parameter_scene_type(scene_type)
    row = db.scalar(
        select(PipelineParameterDefault).where(
            PipelineParameterDefault.pipeline == pipeline,
            PipelineParameterDefault.scene_type == scene,
        )
    )
    return dict(row.options or {}) if row else {}


def merged_task_options(db: Session, pipeline: str, scene_type: str, payload_options: dict[str, Any]) -> dict[str, Any]:
    scene = normalize_parameter_scene_type(scene_type)
    return {
        **system_defaults_for(pipeline, scene),
        **saved_defaults_for(db, pipeline, scene),
        **(payload_options or {}),
    }


def defaults_payload(db: Session) -> dict[str, Any]:
    rows = db.scalars(select(PipelineParameterDefault)).all()
    values = {
        pipeline: {
            scene_type: system_defaults_for(pipeline, scene_type)
            for scene_type in sorted(VALID_SCENE_TYPES)
        }
        for pipeline in sorted(VALID_PIPELINES)
    }
    for row in rows:
        if row.pipeline in values and row.scene_type in values[row.pipeline]:
            values[row.pipeline][row.scene_type] = {
                **system_defaults_for(row.pipeline, row.scene_type),
                **dict(row.options or {}),
            }
    return {"defaults": values}
