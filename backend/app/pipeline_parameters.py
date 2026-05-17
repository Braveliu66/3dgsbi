from __future__ import annotations

from copy import deepcopy
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.fine.colmap_defaults import (
    COLMAP_GUIDED_MATCHING,
    COLMAP_MATCHER,
    COLMAP_MAX_IMAGE_SIZE,
    COLMAP_MIN_REGISTERED_RATIO,
    COLMAP_SIFT_EDGE_THRESHOLD,
    COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE,
    COLMAP_SIFT_DOMAIN_SIZE_POOLING,
    COLMAP_SIFT_MATCH_MAX_RATIO,
    COLMAP_SIFT_MAX_NUM_FEATURES,
    COLMAP_SIFT_PEAK_THRESHOLD,
    COLMAP_THREADS,
    FINE_IMAGE_MAX_SIDE,
    FINE_ITERATIONS,
    FINE_PIPELINE_NAME,
)
from app.litevggt_defaults import (
    LITEVGGT_DEFAULTS_PRESET_KEY,
    litevggt_effective_saved_defaults,
    litevggt_stored_defaults,
    litevggt_system_defaults,
)
from app.models import PipelineParameterDefault


VALID_PIPELINES = {"litevggt_spz", "lingbot_video_pointcloud_fast", FINE_PIPELINE_NAME}
VALID_SCENE_TYPES = {"indoor", "outdoor"}
PIPELINE_DEFAULTS_PRESET_KEY = "_pipeline_defaults_preset"
COLMAP_DEFAULTS_PRESET = "colmap_scene_defaults_2026_05_18_v1"

SCENE_PROFILES = {
    "indoor": {"preview_scene_profile": "indoor_full", "fine_scene_profile": "indoor_full"},
    "outdoor": {"preview_scene_profile": "outdoor_fast_clean", "fine_scene_profile": "outdoor_fast_clean"},
}

LITEVGGT_DEFAULTS: dict[str, dict[str, Any]] = {
    "indoor": litevggt_system_defaults("indoor"),
    "outdoor": litevggt_system_defaults("outdoor"),
}

LINGBOT_DEFAULTS: dict[str, dict[str, Any]] = {
    "indoor": {
        "scene_type": "indoor",
        "preview_lingbot_fps": 10.0,
        "preview_lingbot_image_size": 518,
        "preview_lingbot_target_width": 518,
        "preview_lingbot_target_height": 378,
        "preview_lingbot_preprocess_mode": "crop",
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
        "preview_lingbot_preprocess_mode": "crop",
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

COLMAP_DEFAULTS: dict[str, dict[str, Any]] = {
    scene_type: {
        "scene_type": scene_type,
        "fine_scene_type": scene_type,
        "fine_scene_profile": SCENE_PROFILES[scene_type]["fine_scene_profile"],
        "fine_sfm_backend": "colmap_cli",
        "quality_mode": "auto",
        "camera_distortion": "undistorted",
        "prefer_gpu": True,
        "fine_capture_order": "auto",
        "fine_image_max_side": FINE_IMAGE_MAX_SIDE,
        "fine_iterations": FINE_ITERATIONS,
        "fine_colmap_max_image_size": COLMAP_MAX_IMAGE_SIZE,
        "fine_sift_max_num_features": COLMAP_SIFT_MAX_NUM_FEATURES,
        "fine_colmap_matcher": COLMAP_MATCHER,
        "fine_colmap_threads": COLMAP_THREADS,
        "fine_colmap_sift_peak_threshold": COLMAP_SIFT_PEAK_THRESHOLD,
        "fine_colmap_sift_edge_threshold": COLMAP_SIFT_EDGE_THRESHOLD,
        "fine_colmap_estimate_affine_shape": COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE,
        "fine_colmap_domain_size_pooling": COLMAP_SIFT_DOMAIN_SIZE_POOLING,
        "fine_colmap_guided_matching": COLMAP_GUIDED_MATCHING,
        "fine_colmap_sift_match_max_ratio": COLMAP_SIFT_MATCH_MAX_RATIO,
        "fine_min_registered_ratio": COLMAP_MIN_REGISTERED_RATIO,
        "fine_blur_reject_ratio": 0.10,
    }
    for scene_type in VALID_SCENE_TYPES
}

COLMAP_DEFAULTS["indoor"].update(
    {
        "fine_scene_profile": "indoor_full",
        "fine_colmap_max_image_size": 1500,
        "fine_sift_max_num_features": 24_000,
        "fine_colmap_matcher": "auto",
        "fine_min_registered_ratio": 0.60,
        "fine_blur_reject_ratio": 0.08,
    }
)
COLMAP_DEFAULTS["outdoor"].update(
    {
        "fine_scene_profile": "outdoor_fast_clean",
        "fine_image_max_side": 1080,
        "fine_colmap_max_image_size": 1080,
        "fine_sift_max_num_features": 16_000,
        "fine_colmap_matcher": "auto",
        "fine_min_registered_ratio": 0.55,
        "fine_blur_reject_ratio": 0.12,
    }
)

SYSTEM_DEFAULTS = {
    "litevggt_spz": LITEVGGT_DEFAULTS,
    "lingbot_video_pointcloud_fast": LINGBOT_DEFAULTS,
    FINE_PIPELINE_NAME: COLMAP_DEFAULTS,
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
            pipeline_schema(FINE_PIPELINE_NAME, "COLMAP fine reconstruction", colmap_fields()),
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
        field("litevggt_inference_mode", "推理模式", "select", "推理策略", "auto 使用运行时默认路径；single 一次处理全部帧；windowed 用于长序列或显存不足场景。", options=["auto", "single", "windowed"]),
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
        field("preview_lingbot_preprocess_mode", "预处理模式", "select", "输入预处理", "crop 裁剪到目标比例，pad 保留完整画面。", options=["crop", "pad"]),
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


def colmap_fields() -> list[dict[str, Any]]:
    return [
        field("fine_sfm_backend", "SfM backend", "select", "COLMAP", "COLMAP implementation to use.", options=["colmap_cli", "colmap", "pycolmap"]),
        field("quality_mode", "Quality mode", "select", "COLMAP", "COLMAP quality policy.", options=["auto", "quality", "speed"]),
        field("camera_distortion", "Camera distortion", "select", "COLMAP", "Fine input camera distortion policy.", options=["undistorted"]),
        field("prefer_gpu", "Prefer GPU", "boolean", "COLMAP", "Use GPU feature extraction and matching when available."),
        field("fine_capture_order", "Capture order", "select", "COLMAP", "Capture order hint.", options=["auto", "unordered", "sequential"]),
        field("fine_scene_profile", "Fine scene profile", "select", "Scene", "Fine scene profile.", options=["indoor_full", "outdoor_fast_clean"]),
        field("fine_image_max_side", "Input max side", "number", "Input", "Fine input image max side.", min=256, max=4096, step=1),
        field("fine_colmap_max_image_size", "COLMAP max image size", "number", "COLMAP", "COLMAP feature extraction max image size.", min=512, max=4096, step=1),
        field("fine_sift_max_num_features", "SIFT max features", "number", "COLMAP", "Maximum SIFT features per image.", min=1024, max=65536, step=1),
        field("fine_colmap_matcher", "Matcher", "select", "COLMAP", "COLMAP matcher policy.", options=["auto", "exhaustive", "sequential", "vocab_tree", "spatial"]),
        field("fine_colmap_threads", "Threads", "number", "COLMAP", "COLMAP worker threads.", min=1, max=32, step=1),
        field("fine_colmap_sift_peak_threshold", "SIFT peak threshold", "number", "COLMAP", "SIFT peak threshold.", min=0.0001, max=0.1, step=0.0001),
        field("fine_colmap_sift_edge_threshold", "SIFT edge threshold", "number", "COLMAP", "SIFT edge threshold.", min=1, max=100, step=0.1),
        field("fine_colmap_estimate_affine_shape", "Affine shape", "boolean", "COLMAP", "Enable affine shape estimation."),
        field("fine_colmap_domain_size_pooling", "Domain size pooling", "boolean", "COLMAP", "Enable domain size pooling."),
        field("fine_colmap_guided_matching", "Guided matching", "boolean", "COLMAP", "Enable guided matching."),
        field("fine_colmap_sift_match_max_ratio", "SIFT max ratio", "number", "COLMAP", "SIFT match max ratio.", min=0.1, max=1, step=0.01),
        field("fine_min_registered_ratio", "Min registered ratio", "nullable_number", "COLMAP", "Minimum registered image ratio.", min=0.30, max=0.95, step=0.01),
        field("fine_blur_reject_ratio", "Low-quality reject ratio", "number", "Input", "Ratio of lowest-quality frames to filter before COLMAP.", min=0, max=0.45, step=0.01),
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
    if not row:
        return {}
    return effective_saved_defaults_for(pipeline, row.options)


def stored_defaults_for(pipeline: str, options: dict[str, Any]) -> dict[str, Any]:
    if pipeline == "litevggt_spz":
        return litevggt_stored_defaults(options)
    if pipeline == FINE_PIPELINE_NAME:
        return {**dict(options or {}), PIPELINE_DEFAULTS_PRESET_KEY: COLMAP_DEFAULTS_PRESET}
    return dict(options or {})


def effective_saved_defaults_for(pipeline: str, options: dict[str, Any] | None) -> dict[str, Any]:
    if pipeline == "litevggt_spz":
        return litevggt_effective_saved_defaults(options)
    stored = dict(options or {})
    if pipeline == FINE_PIPELINE_NAME:
        if stored.get(PIPELINE_DEFAULTS_PRESET_KEY) != COLMAP_DEFAULTS_PRESET:
            return {}
        stored.pop(PIPELINE_DEFAULTS_PRESET_KEY, None)
    return stored


def merged_task_options(db: Session, pipeline: str, scene_type: str, payload_options: dict[str, Any]) -> dict[str, Any]:
    merged, _sources = merged_task_options_with_sources(db, pipeline, scene_type, payload_options)
    return merged


def merged_task_options_with_sources(
    db: Session,
    pipeline: str,
    scene_type: str,
    payload_options: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    scene = normalize_parameter_scene_type(scene_type)
    system = system_defaults_for(pipeline, scene)
    saved = saved_defaults_for(db, pipeline, scene)
    payload = dict(payload_options or {})
    merged = {**system, **saved, **payload}
    sources = {
        key: ("request" if key in payload else "admin_saved" if key in saved else "system_default")
        for key in merged
        if key != LITEVGGT_DEFAULTS_PRESET_KEY
    }
    return merged, sources


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
                **effective_saved_defaults_for(row.pipeline, row.options),
            }
    return {"defaults": values}
