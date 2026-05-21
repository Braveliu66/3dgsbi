from __future__ import annotations

from copy import deepcopy
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.fine.colmap_defaults import (
    COLMAP_GUIDED_MATCHING,
    COLMAP_MATCHER,
    COLMAP_MAX_IMAGE_SIZE,
    COLMAP_MAX_NUM_MATCHES,
    COLMAP_MIN_REGISTERED_RATIO,
    COLMAP_SEQUENTIAL_OVERLAP,
    COLMAP_SIFT_EDGE_THRESHOLD,
    COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE,
    COLMAP_SIFT_DOMAIN_SIZE_POOLING,
    COLMAP_SIFT_MATCH_MAX_RATIO,
    COLMAP_SIFT_MAX_NUM_FEATURES,
    COLMAP_SIFT_PEAK_THRESHOLD,
    COLMAP_THREADS,
    FINE_DEFAULT_IMAGE_MAX_SIDE,
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


VALID_PIPELINES = {"litevggt_spz", FINE_PIPELINE_NAME}
VALID_SCENE_TYPES = {"indoor", "outdoor"}
PIPELINE_DEFAULTS_PRESET_KEY = "_pipeline_defaults_preset"
COLMAP_DEFAULTS_PRESET = "dash_deblur_group_eap_gsplat_stable_density_2026_05_20_v2"
BLUR_CODE_DIM = 8

SCENE_PROFILES = {
    "indoor": {"preview_scene_profile": "indoor_full", "fine_scene_profile": "indoor_full"},
    "outdoor": {"preview_scene_profile": "outdoor_fast_clean", "fine_scene_profile": "outdoor_fast_clean"},
}

LITEVGGT_DEFAULTS: dict[str, dict[str, Any]] = {
    "indoor": litevggt_system_defaults("indoor"),
    "outdoor": litevggt_system_defaults("outdoor"),
}

COLMAP_DEFAULTS: dict[str, dict[str, Any]] = {
    scene_type: {
        "scene_type": scene_type,
        "fine_scene_type": scene_type,
        "fine_scene_profile": SCENE_PROFILES[scene_type]["fine_scene_profile"],
        "fine_sfm_backend": "colmap_global",
        "quality_mode": "auto",
        "camera_distortion": "undistorted",
        "prefer_gpu": True,
        "fine_capture_order": "auto",
        "fine_image_max_side": FINE_DEFAULT_IMAGE_MAX_SIDE,
        "fine_iterations": FINE_ITERATIONS,
        "resolution": -1,
        "fine_colmap_max_image_size": COLMAP_MAX_IMAGE_SIZE,
        "fine_sift_max_num_features_auto": True,
        "fine_sift_max_num_features": COLMAP_SIFT_MAX_NUM_FEATURES,
        "fine_colmap_max_num_matches": COLMAP_MAX_NUM_MATCHES,
        "fine_colmap_sequential_overlap": COLMAP_SEQUENTIAL_OVERLAP,
        "fine_colmap_matcher": COLMAP_MATCHER,
        "fine_colmap_threads": COLMAP_THREADS,
        "fine_colmap_sift_peak_threshold": COLMAP_SIFT_PEAK_THRESHOLD,
        "fine_colmap_sift_edge_threshold": COLMAP_SIFT_EDGE_THRESHOLD,
        "fine_colmap_estimate_affine_shape": COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE,
        "fine_colmap_domain_size_pooling": COLMAP_SIFT_DOMAIN_SIZE_POOLING,
        "fine_colmap_guided_matching": COLMAP_GUIDED_MATCHING,
        "fine_colmap_sift_match_max_ratio": COLMAP_SIFT_MATCH_MAX_RATIO,
        "fine_min_registered_ratio": COLMAP_MIN_REGISTERED_RATIO,
        "fine_blur_reject_ratio": 0.0,
        "fine_eap_enabled": True,
        "fine_eap_dbscan_eps": 30,
        "fine_eap_min_samples": 10,
        "fine_eap_mask_radius": 20,
        "fine_eap_max_point_multiplier": 10,
        "fine_deblur_enabled": True,
        "fine_deblur_mode": "motion",
        "fine_gsplat_enabled": True,
        "fine_spz_enabled": True,
        "fine_training_flavor": "auto",
        "fine_train_entrypoint": "",
        "fine_train_python": "",
        "fine_trainer_repo": "",
        "fine_data_device": "cpu",
        "use_pos": True,
        "blur_code_dim": BLUR_CODE_DIM,
        "num_moments": 4,
        "hidden": 3,
        "width": 64,
        "gtnet_lr": 0.001,
        "position_lr_final": 0.000016,
        "percent_dense": 0.01,
        "lambda_dssim": 0.2,
        "lambda_s": 0.01,
        "lambda_p": 0.01,
        "max_clamp": 1.10,
        "densify_from_iter": 500,
        "densify_until_iter": 3000,
        "densification_interval": 100,
        "densify_grad_threshold": 0.0005,
        "densify_prune_threshold": 0.01,
        "densify_with_depth": True,
        "prune_range": 3,
        "pts_iter": 999999,
        "pts_rate": 0.0,
        "pts_dist": 2,
        "pts_N_intpl": 4,
        "pts_N_pts": 0,
        "pts_add_bound": 10,
    }
    for scene_type in VALID_SCENE_TYPES
}

COLMAP_DEFAULTS["indoor"].update(
    {
        "fine_scene_profile": "indoor_full",
        "fine_image_max_side": FINE_DEFAULT_IMAGE_MAX_SIDE,
        "fine_colmap_max_image_size": COLMAP_MAX_IMAGE_SIZE,
        "fine_sift_max_num_features": 32_768,
        "fine_colmap_max_num_matches": 32_768,
        "fine_colmap_sequential_overlap": 60,
        "fine_colmap_matcher": "auto",
        "fine_blur_reject_ratio": 0.0,
    }
)
COLMAP_DEFAULTS["outdoor"].update(
    {
        "fine_scene_profile": "outdoor_fast_clean",
        "fine_image_max_side": FINE_DEFAULT_IMAGE_MAX_SIDE,
        "fine_colmap_max_image_size": COLMAP_MAX_IMAGE_SIZE,
        "fine_sift_max_num_features": 65_536,
        "fine_colmap_max_num_matches": 65_536,
        "fine_colmap_sequential_overlap": 30,
        "fine_colmap_matcher": "auto",
        "fine_blur_reject_ratio": 0.0,
    }
)

SYSTEM_DEFAULTS = {
    "litevggt_spz": LITEVGGT_DEFAULTS,
    FINE_PIPELINE_NAME: COLMAP_DEFAULTS,
}


def pipeline_parameter_schema() -> dict[str, Any]:
    return {
        "scene_types": [
            {"value": "indoor", "label": "室内"},
            {"value": "outdoor", "label": "室外"},
        ],
        "pipelines": [
            pipeline_schema("litevggt_spz", "LiteVGGT 图片预览", litevggt_fields()),
            pipeline_schema(FINE_PIPELINE_NAME, "Deblur3DGS 精细重建", colmap_fields()),
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
        field("litevggt_keep_ratio", "点云保留比例", "number", "点云输出", "按质量或覆盖策略保留的点比例，越大点越多、加载更慢。", min=0.01, max=1, step=0.01),
        field("preview_max_points", "最大点数", "number", "点云输出", "最终预览点云点数上限，直接影响 PLY/SPZ 大小和浏览器加载。", min=1, max=20_000_000, step=1000),
        field("litevggt_max_input_frames", "最大输入帧数", "nullable_number", "输入抽帧", "最多送入 LiteVGGT 的图片数；空值表示使用策略默认。", min=1, max=2000, step=1),
        field("litevggt_target_size", "模型输入尺寸", "number", "输入预处理", "LiteVGGT 输入目标尺寸，越大细节越好但显存和时间更高。", min=128, max=1024, step=1),
        field("litevggt_frame_stride", "固定抽帧步长", "nullable_number", "输入抽帧", "例如 2 表示隔帧采样；空值表示交给关键帧策略。", min=1, max=10000, step=1),
        field("litevggt_depth_conf_thresh", "深度置信度阈值", "nullable_number", "点云过滤", "丢弃低置信度深度点；空值表示使用运行时默认逻辑。", min=-100, max=100, step=0.01),
        field("litevggt_preprocess_mode", "预处理模式", "select", "输入预处理", "pad 保留完整画面；crop 裁剪到目标比例。", options=["pad", "crop"]),
        field("litevggt_inference_mode", "推理模式", "select", "推理策略", "auto 使用运行时默认路径；single 一次处理全部帧；windowed 用于长序列或显存不足场景。", options=["auto", "single", "windowed"]),
        field("litevggt_chunk_size", "chunk 帧数", "number", "推理策略", "分块推理时每块帧数，越大上下文更完整但更占显存。", min=1, max=512, step=1),
        field("litevggt_overlap", "chunk 重叠帧数", "number", "推理策略", "相邻 chunk 的重叠帧数，越大连续性更好但计算更多。", min=0, max=512, step=1),
        field("litevggt_loop_closure", "回环一致性", "boolean", "推理策略", "开启后有利于环绕拍摄或闭环场景结构一致。"),
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


def colmap_fields() -> list[dict[str, Any]]:
    return [
        field("fine_sfm_backend", "SfM 后端", "select", "COLMAP", "精细重建使用的 SfM 实现；colmap_global/gcolmap 使用 COLMAP global_mapper。", options=["colmap_global", "gcolmap", "colmap_cli", "colmap", "pycolmap"], option_labels={"colmap_global": "COLMAP Global Mapper", "gcolmap": "GColmap", "colmap_cli": "COLMAP 命令行", "colmap": "COLMAP 命令行", "pycolmap": "PyCOLMAP"}),
        field("quality_mode", "质量模式", "select", "COLMAP", "COLMAP 质量策略。", options=["auto", "quality", "speed"], option_labels={"auto": "自动", "quality": "质量优先", "speed": "速度优先"}),
        field("camera_distortion", "相机畸变策略", "select", "COLMAP", "精细重建输入图像的相机畸变处理策略。", options=["undistorted"], option_labels={"undistorted": "已去畸变"}),
        field("prefer_gpu", "优先使用 GPU", "boolean", "COLMAP", "可用时使用 GPU 提取和匹配特征。"),
        field("fine_capture_order", "拍摄顺序", "select", "COLMAP", "输入素材的拍摄顺序提示。", options=["auto", "unordered", "sequential"], option_labels={"auto": "自动", "unordered": "无序", "sequential": "顺序拍摄"}),
        field("fine_scene_profile", "精细场景策略", "select", "场景", "精细重建使用的场景默认策略。", options=["indoor_full", "outdoor_fast_clean"], option_labels={"indoor_full": "室内完整", "outdoor_fast_clean": "室外快速清理"}),
        field("fine_image_max_side", "训练输入最大边长", "number", "输入", "精细训练默认使用原图分辨率；0 表示不缩放。", min=0, max=4096, step=1),
        field("fine_colmap_max_image_size", "COLMAP 特征最大边长", "number", "COLMAP", "COLMAP 特征提取阶段的最大图像尺寸；默认 1080px，训练图像使用原图分辨率。", min=512, max=4096, step=1),
        field("fine_sift_max_num_features_auto", "SIFT 特征数自适应", "boolean", "COLMAP", "开启后根据图像清晰度、模糊比例和图集规模自动选择 SIFT 上限；小图集默认保留更多特征以改善几何初始化。关闭后使用固定上限。"),
        field("fine_sift_max_num_features", "SIFT 最大特征数", "number", "COLMAP", "关闭自适应后使用的固定 SIFT 特征上限。旧默认 32768/65536 会被视为自动模式。", min=1024, max=65536, step=1),
        field("fine_colmap_max_num_matches", "最大匹配数", "number", "COLMAP", "每对图像最多保留的 COLMAP 特征匹配数。", min=1024, max=65536, step=1),
        field("fine_colmap_sequential_overlap", "顺序匹配重叠帧", "number", "COLMAP", "顺序拍摄时前后参与匹配的邻近帧数量。", min=4, max=200, step=1),
        field("fine_colmap_matcher", "匹配器", "select", "COLMAP", "COLMAP 图像匹配策略。", options=["auto", "exhaustive", "sequential", "vocab_tree", "spatial"], option_labels={"auto": "自动", "exhaustive": "穷举匹配", "sequential": "顺序匹配", "vocab_tree": "词袋树匹配", "spatial": "空间匹配"}),
        field("fine_colmap_threads", "线程数", "number", "COLMAP", "COLMAP 可使用的工作线程数。", min=1, max=32, step=1),
        field("fine_colmap_sift_peak_threshold", "SIFT 峰值阈值", "number", "COLMAP", "SIFT 特征峰值阈值。", min=0.0001, max=0.1, step=0.0001),
        field("fine_colmap_sift_edge_threshold", "SIFT 边缘阈值", "number", "COLMAP", "SIFT 边缘响应过滤阈值。", min=1, max=100, step=0.1),
        field("fine_colmap_estimate_affine_shape", "估计仿射形状", "boolean", "COLMAP", "开启 SIFT 仿射形状估计。"),
        field("fine_colmap_domain_size_pooling", "域尺寸池化", "boolean", "COLMAP", "开启 SIFT 域尺寸池化。"),
        field("fine_colmap_guided_matching", "引导匹配", "boolean", "COLMAP", "开启几何引导匹配。"),
        field("fine_colmap_sift_match_max_ratio", "SIFT 匹配比例阈值", "number", "COLMAP", "SIFT 最近邻匹配的最大比例阈值。", min=0.1, max=1, step=0.01),
        field("fine_min_registered_ratio", "最小注册比例", "nullable_number", "COLMAP", "图像成功注册比例低于该值时判定重建质量不足；空值表示不强制。", min=0.30, max=0.95, step=0.01),
        field("fine_blur_reject_ratio", "低质量帧剔除比例", "number", "输入", "进入 COLMAP 前剔除质量最低的帧比例；默认 0 只记录模糊分析，不剔除模糊帧。", min=0, max=0.45, step=0.01),
        field("fine_eap_enabled", "启用 EAP 初始化", "boolean", "EAP 初始化", "在 COLMAP 后、训练前运行 EAP/APA 点云增强，生成 points3D_eap 作为训练初始点云。"),
        field("fine_eap_dbscan_eps", "EAP 聚类半径", "number", "EAP 初始化", "投影点密集区域聚类半径；用于生成增强图像的遮罩区域。", min=1, max=256, step=1),
        field("fine_eap_min_samples", "EAP 最小聚类点数", "number", "EAP 初始化", "密集区域聚类所需的最小投影点数量。", min=1, max=512, step=1),
        field("fine_eap_mask_radius", "EAP 遮罩半径", "number", "EAP 初始化", "对密集投影点生成遮罩时使用的像素半径。", min=1, max=256, step=1),
        field("fine_eap_max_point_multiplier", "EAP 点数倍率上限", "number", "EAP 初始化", "增强后 sparse 点数相对原始点数的安全上限，超过则任务失败。", min=1, max=100, step=1),
        field("fine_trainer_repo", "训练器目录", "text", "训练运行时", "DashDeblurGroupGS 训练器仓库路径；留空使用 worker 内置训练器。"),
        field("fine_training_flavor", "训练器兼容模式", "select", "训练运行时", "训练器兼容模式。", options=["auto", "dash_deblur_group"], option_labels={"auto": "自动", "dash_deblur_group": "Deblur3DGS"}),
        field("fine_train_python", "训练 Python", "text", "训练运行时", "训练器使用的 Python 可执行文件；留空使用 worker Python。"),
        field("fine_train_entrypoint", "训练入口脚本", "text", "训练运行时", "训练器仓库内的训练脚本；留空使用 train.py。"),
        field("fine_data_device", "图像张量设备", "select", "训练运行时", "兼容训练器中图像张量存放的设备。", options=["cpu", "cuda"], option_labels={"cpu": "CPU", "cuda": "CUDA"}),
        field("fine_spz_enabled", "导出 SPZ", "boolean", "训练运行时", "把最终 Gaussian PLY 转成 Spark SPZ 供网页查看器使用。"),
        field("fine_gsplat_enabled", "启用 gsplat 后端", "boolean", "训练运行时", "只在 sharp/canonical 渲染路径使用 gsplat rasterizer；motion/defocus 仍使用原始后端。"),
        field("fine_iterations", "训练迭代数", "number", "去模糊训练", "DashDeblurGroupGS 总训练迭代数。", min=1, max=100000, step=100),
        field("resolution", "训练下采样倍率", "number", "去模糊训练", "输入图像进入 trainer 的下采样倍率；-1 使用训练器默认策略。", min=-1, max=8, step=1),
        field("fine_deblur_enabled", "启用去模糊", "boolean", "去模糊训练", "启用 Deblurring-3DGS 的 GTnet 训练分支。"),
        field("fine_deblur_mode", "去模糊模式", "select", "去模糊训练", "训练使用的去模糊物理分支；默认使用 motion，不再根据模糊分析自动切换。", options=["motion", "defocus", "sharp"], option_labels={"motion": "运动模糊", "defocus": "失焦模糊", "sharp": "清晰"}),
        field("blur_code_dim", "每图模糊向量维度", "select", "去模糊训练", "每张图的 blur embedding 维度；当前固定为 8。", options=["8"], option_labels={"8": "8"}),
        field("use_pos", "启用位置偏移", "boolean", "去模糊训练", "启用 GTnet 的位置偏移分支。"),
        field("num_moments", "运动矩数量", "number", "去模糊训练", "运动模糊渲染使用的虚拟时刻数量。", min=1, max=16, step=1),
        field("hidden", "GTnet 隐藏层数", "number", "去模糊训练", "GTnet 网络深度。", min=1, max=8, step=1),
        field("width", "GTnet 宽度", "number", "去模糊训练", "GTnet 隐藏层宽度。", min=16, max=512, step=16),
        field("gtnet_lr", "GTnet 学习率", "number", "去模糊训练", "GTnet 学习率。", min=0, max=1, step=0.0001),
        field("position_lr_final", "最终位置学习率", "number", "去模糊训练", "Gaussian 位置参数的最终学习率。", min=0, max=1, step=0.0000001),
        field("percent_dense", "密集阈值比例", "number", "加点与剪枝", "按场景范围区分 clone 和 split 的比例阈值。", min=0, max=1, step=0.001),
        field("lambda_dssim", "DSSIM 权重", "number", "去模糊训练", "训练损失中的 DSSIM 权重。", min=0, max=1, step=0.01),
        field("lambda_s", "尺度正则权重", "number", "去模糊训练", "去模糊尺度正则权重。", min=0, max=1, step=0.001),
        field("lambda_p", "位置正则权重", "number", "去模糊训练", "去模糊位置正则权重。", min=0, max=1, step=0.001),
        field("max_clamp", "模糊变换截断", "number", "去模糊训练", "预测模糊变换的截断上限。", min=1, max=2, step=0.01),
        field("densify_from_iter", "开始加点迭代", "number", "加点与剪枝", "允许 Gaussian densification 的起始迭代。", min=0, max=100000, step=100),
        field("densify_until_iter", "停止加点迭代", "number", "加点与剪枝", "允许 Gaussian densification 的最后迭代。", min=0, max=100000, step=100),
        field("densification_interval", "加点间隔", "number", "加点与剪枝", "两次 densification 之间的迭代间隔。", min=1, max=10000, step=1),
        field("densify_grad_threshold", "加点梯度阈值", "number", "加点与剪枝", "候选点进入 densification 的梯度阈值。", min=0, max=1, step=0.00001),
        field("densify_prune_threshold", "剪枝不透明度阈值", "number", "加点与剪枝", "Deblur-safe pruning 使用的不透明度阈值。", min=0, max=1, step=0.0001),
        field("densify_with_depth", "启用深度剪枝", "boolean", "加点与剪枝", "按深度提高远端背景点的剪枝力度。"),
        field("prune_range", "深度剪枝范围", "number", "加点与剪枝", "传给训练器的深度剪枝范围。", min=0, max=32, step=1),
        field("pts_iter", "随机补点迭代", "number", "随机补点", "Deblurring-3DGS add_points 的触发迭代；默认 999999 表示禁用。", min=0, max=1000000, step=100),
        field("pts_rate", "随机补点密度", "number", "随机补点", "当 pts_N_pts=0 时按包围盒体积估算随机补点数的密度参数；0 表示不按体积估算。", min=0, max=10, step=0.1),
        field("pts_dist", "随机补点插值距离", "number", "随机补点", "随机补点颜色插值使用的邻近距离。", min=0, max=64, step=1),
        field("pts_N_intpl", "随机补点插值邻居数", "number", "随机补点", "随机补点颜色插值使用的邻居数量。", min=1, max=32, step=1),
        field("pts_N_pts", "随机补点数量", "number", "随机补点", "add_points 最多新增的随机点数；默认 0 表示禁用。", min=0, max=5000000, step=10000),
        field("pts_add_bound", "随机补点边界裁剪", "number", "随机补点", "add_points 采样包围盒的边界裁剪数量。", min=0, max=1000, step=1),
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
        stored = {**dict(options or {}), PIPELINE_DEFAULTS_PRESET_KEY: COLMAP_DEFAULTS_PRESET}
        stored["blur_code_dim"] = BLUR_CODE_DIM
        return stored
    return dict(options or {})


def effective_saved_defaults_for(pipeline: str, options: dict[str, Any] | None) -> dict[str, Any]:
    if pipeline == "litevggt_spz":
        return litevggt_effective_saved_defaults(options)
    stored = dict(options or {})
    if pipeline == FINE_PIPELINE_NAME:
        if stored.get(PIPELINE_DEFAULTS_PRESET_KEY) != COLMAP_DEFAULTS_PRESET:
            return {}
        stored.pop(PIPELINE_DEFAULTS_PRESET_KEY, None)
        stored = normalize_fine_legacy_options(stored)
        stored["blur_code_dim"] = BLUR_CODE_DIM
    return stored


def normalize_fine_legacy_options(options: dict[str, Any]) -> dict[str, Any]:
    adjusted = dict(options or {})
    for key in ("fine_deblur_mode", "fine_deblur_mode_requested"):
        if str(adjusted.get(key) or "").strip().lower() in {"mix", "auto", "automatic"}:
            adjusted[key] = "motion"
    return adjusted


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
    if pipeline == FINE_PIPELINE_NAME:
        payload = normalize_fine_legacy_options(payload)
    merged = {**system, **saved, **payload}
    if pipeline == FINE_PIPELINE_NAME:
        merged["blur_code_dim"] = BLUR_CODE_DIM
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
