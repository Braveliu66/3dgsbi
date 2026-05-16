from __future__ import annotations

from typing import Any

from app.preview.io.ply import convert_pointcloud_ply_to_fixed_splat_ply
from app.preview.io.spz import convert_ply_to_spz
from app.preview.types import PreviewContext, PreviewResult, SOURCE_COMMITS
from app.preview.utils import StageTimer, image_files, require_file
from app.preview.vendor.litevggt_runtime import run_litevggt_pointcloud


DEFAULT_PREVIEW_SCENE_PROFILE = "mixed_balanced"
LITEVGGT_SCENE_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "mixed_balanced": {
        # 点云保留比例；越大点越多，结构更完整，但后处理、SPZ转换、前端加载更慢
        "litevggt_keep_ratio": 0.45,

        # 最终预览最大点数；直接影响 PLY/SPZ 文件大小、转换速度和浏览器加载速度
        "preview_max_points": 2_000_000,

        # 最多送入 LiteVGGT 的输入帧数；这是影响推理速度和显存的核心参数之一
        # 500+ 图片时不建议全量输入，建议把原始图片当候选池，只选关键帧
        "litevggt_max_input_frames": 240,

        # LiteVGGT 输入目标尺寸；越大细节越好，但 token 数更多，推理更慢、更占显存
        # 常见档位：336/392 极速，448 均衡，476/518 质量优先
        "litevggt_target_size": 336,

        # 固定抽帧步长；None 表示不按固定 stride 抽帧，而交给关键帧/场景变化策略
        # 例如设为 2 表示每隔 2 帧取 1 帧，速度更快但可能漏掉关键视角
        "litevggt_frame_stride": None,

        # 深度置信度阈值；None 表示使用模型/后处理默认逻辑
        # 设置后会丢弃低置信度深度点，点云更干净，但可能变稀疏
        "litevggt_depth_conf_thresh": None,

        # 图片预处理模式；pad 表示保持完整画面，用 padding 补到目标尺寸
        # 优点是不裁掉边缘；缺点是有效区域可能变小
        "litevggt_preprocess_mode": "pad",

        # 推理模式；single 表示一次性推理选中的全部帧
        # 如果支持 chunk/window 模式，长序列可分块推理以降低显存
        "litevggt_inference_mode": "single",

        # 分块推理时每个 chunk 的帧数；single 模式下可能不生效
        # 越大上下文更完整，但显存和时间更高
        "litevggt_chunk_size": 64,

        # 分块推理时相邻 chunk 的重叠帧数；single 模式下可能不生效
        # 越大 chunk 之间更连续，但重复计算更多
        "litevggt_overlap": 16,

        # 是否启用回环/首尾一致性处理；有助于闭环场景结构一致
        # 会增加一些计算和后处理成本
        "litevggt_loop_closure": True,

        # 关键帧目标数量；None 表示不额外限制关键帧目标数
        # 如果设置为 240，表示场景选帧尽量选到约 240 张关键帧
        "litevggt_keyframe_target": 240,

        # 关键帧之间的最小帧间隔；越大抽帧越稀疏，速度更快但连续性下降
        "litevggt_min_frame_gap": 1,

        # 最小场景变化阈值；变化超过该值才更可能被选为关键帧
        # 值越大，选帧越少、更激进；值越小，选帧越密
        # 注意：6.0 这种值通常表示另一种特征/直方图尺度，需和实现里的计算尺度一致
        "litevggt_min_scene_change": 6.0,

        # 分窗口/中间点云体素下采样比例；按场景包围盒对角线计算 voxel size
        # 1/700 表示 voxel size = scene_diag / 700；值越大，下采样越强，速度越快，细节越少
        "litevggt_window_voxel_diag_ratio": 1 / 700,

        # 最终点云体素下采样比例；用于最终输出前统一降采样
        # 1/800 比 1/450 保留更多细节，但点数更多
        "litevggt_final_voxel_diag_ratio": 1 / 800,

        # 点选择策略：
        # scene_coverage：尽量保证不同帧/不同区域都有覆盖，适合结构预览
        # global_confidence：全局按置信度选点，点更干净但可能局部区域被丢失
        # per_frame：每帧保留一定点，避免某些帧完全没贡献
        "litevggt_point_selection_strategy": "scene_coverage",

        # 单轴低分位裁剪；去掉 x/y/z 方向极小端的离群点
        # 值越大裁剪越狠，室外噪声可适当调大
        "litevggt_axis_trim_low_quantile": 0.003,

        # 单轴高分位裁剪；去掉 x/y/z 方向极大端的离群点
        # 0.995 表示保留到 99.5% 分位，去掉最外侧 0.5% 高端离群点
        "litevggt_axis_trim_high_quantile": 0.990,

        # 空间距离保留分位；按距离中心/主体区域的空间分布去掉远端离群点
        # 值越小越干净但可能裁掉真实远处结构
        "litevggt_spatial_keep_quantile": 0.985,
    },

    "indoor_full": {
        # 室内完整模式保留更多点；结构更完整，但速度慢于 fast
        "litevggt_keep_ratio": 0.55,

        # 最终最大点数；室内 full 仍限制在 500 万，避免浏览器压力过大
        "preview_max_points": 3_000_000,

        # 最多输入帧数；室内细节多，240 帧可以获得更完整覆盖，但速度会明显慢于 64/96 帧
        "litevggt_max_input_frames": 240,

        # 输入目标尺寸；476 偏质量优先，适合室内结构线条和细节
        "litevggt_target_size": 240,

        # 固定抽帧步长；None 表示由关键帧选择策略决定
        "litevggt_frame_stride": None,

        # 深度置信度阈值；None 表示不过早裁掉低置信度点
        "litevggt_depth_conf_thresh": None,

        # pad 模式保留完整画面，适合室内避免裁掉墙角、天花、地面边缘
        "litevggt_preprocess_mode": "pad",

        # 单次推理模式；如果 240 帧显存不够，应切到 chunk/window 模式
        "litevggt_inference_mode": "single",

        # chunk 大小；室内 full 用 48，单块显存压力低于 64
        "litevggt_chunk_size": 48,

        # chunk 重叠帧数；24 表示 50% 重叠，连续性好，但重复计算更多
        "litevggt_overlap": 8,

        # 室内闭环/回环常见，开启有助于房间结构一致
        "litevggt_loop_closure": True,

        # None 表示不额外指定关键帧目标数量，主要受 max_input_frames 和场景变化控制
        "litevggt_keyframe_target": None,

        # 最小帧间隔 1，允许相邻帧都被选中；室内慢速移动时能保留连续性
        "litevggt_min_frame_gap": 2,

        # 室内场景变化阈值；0.045 适合归一化图像差异/场景变化分数
        # 值越小，选帧越多；值越大，选帧越少
        "litevggt_min_scene_change": 0.065,

        # 窗口阶段不做体素下采样；保留更多室内细节，但中间点数会更多
        "litevggt_window_voxel_diag_ratio": 1 / 700,

        # 最终体素下采样；1/800 保留较多细节
        "litevggt_final_voxel_diag_ratio": 1 / 700,

        # 室内结构覆盖优先，避免只保留高置信度墙面而丢掉角落/通道
        "litevggt_point_selection_strategy": "scene_coverage",

        # 低端离群点裁剪；室内噪声相对可控，裁剪不要太狠
        "litevggt_axis_trim_low_quantile": 0.02,

        # 高端离群点裁剪；去掉最外侧 0.5% 高端离群点
        "litevggt_axis_trim_high_quantile": 0.90,

        # 空间离群点裁剪；0.985 会去掉较远的异常点，让室内点云更干净
        "litevggt_spatial_keep_quantile": 0.975,
    },

    "outdoor_fast_clean": {
        # 室外快速干净模式保留较少点；速度更快、噪声更少，但细节更稀疏
        "litevggt_keep_ratio": 0.25,

        # 最终最大点数；虽然 keep_ratio 低，但仍允许最多 500 万点
        # 若目标是极速网页预览，可以进一步降到 1_500_000 ~ 3_000_000
        "preview_max_points": 4_000_000,

        # 最多输入帧数；240 对室外大场景覆盖更好，但速度不是最极速
        # 500+ 图片极速预览建议 64/96/128
        "litevggt_max_input_frames": 240,

        # 输入目标尺寸；476 质量较好，但室外 fast 若追求速度可降到 392/448
        "litevggt_target_size": 476,

        # 固定抽帧步长；None 表示用关键帧选择策略
        "litevggt_frame_stride": None,

        # 深度置信度阈值；None 表示不手动设置
        # 室外天空/反光/远景多，可以考虑设置阈值过滤低置信度点
        "litevggt_depth_conf_thresh": None,

        # pad 模式保留完整画面；室外可避免裁掉建筑边缘
        "litevggt_preprocess_mode": "pad",

        # 单次推理；长序列或显存不足时建议切 chunk/window
        "litevggt_inference_mode": "single",

        # chunk 大小；48 相对保守，适合降低显存峰值
        "litevggt_chunk_size": 48,

        # chunk 重叠帧数；8 比室内小，重复计算少，速度更快
        "litevggt_overlap": 8,

        # 开启回环；室外环绕建筑/街区时有帮助
        "litevggt_loop_closure": True,

        # 关键帧目标数量；室外 fast 目标 160，比 max_input_frames 240 更激进
        "litevggt_keyframe_target": 240,

        # 关键帧最小间隔 2，减少相邻重复帧，提升速度和多样性
        "litevggt_min_frame_gap": 3,

        # 场景变化阈值；8.0 表示更激进地筛选变化明显的帧
        # 注意要确认实现里的 scene_change 分数是否也是 0~10 这类尺度
        "litevggt_min_scene_change": 3.0,

        # 窗口阶段体素下采样更强；1/350 比 1/700 voxel 更大，点更少、更干净
        "litevggt_window_voxel_diag_ratio": 1 / 350,

        # 最终体素下采样较强；1/450 会明显减少室外噪声和点数
        "litevggt_final_voxel_diag_ratio": 1 / 450,

        # 全局置信度优先；更适合室外去噪，但可能牺牲边缘/远处覆盖
        "litevggt_point_selection_strategy": "global_confidence",

        # 低端裁剪较强；室外离群点多，0.02 会裁掉更多异常区域
        "litevggt_axis_trim_low_quantile": 0.001,

        # 高端裁剪较强；0.98 会裁掉最高端 2% 空间分布点，点云更干净但可能损失远景
        "litevggt_axis_trim_high_quantile": 0.995,

        # 空间裁剪较强；0.98 去掉最远 2% 空间离群点
        "litevggt_spatial_keep_quantile": 0.99,
    },
}

def run(ctx: PreviewContext) -> PreviewResult:
    timer = StageTimer()
    weight = require_file(
        ctx.model_path("litevggt", "te_dict.pt"),
        "LITEVGGT_WEIGHT_MISSING",
        "LiteVGGT weight",
    )
    points_ply_path = ctx.work_dir / "litevggt" / "preview_points.ply"
    splats_ply_path = ctx.work_dir / "litevggt" / "preview_splats.ply"
    meta_path = ctx.work_dir / "litevggt" / "preview_meta.json"
    files = image_files(ctx.input_dir)

    scene_profile, profile_defaults = _resolve_scene_profile(ctx.options)
    keep_ratio = _read_float(ctx.options.get("litevggt_keep_ratio"), profile_defaults["litevggt_keep_ratio"])
    max_points = _read_int(ctx.options.get("preview_max_points"), profile_defaults["preview_max_points"])
    max_input_frames_value = ctx.options.get("litevggt_max_input_frames")
    if max_input_frames_value is not None:
        max_input_frames = _read_optional_int(max_input_frames_value)
    else:
        max_input_frames = profile_defaults["litevggt_max_input_frames"]
    target_size = _read_int(ctx.options.get("litevggt_target_size"), profile_defaults["litevggt_target_size"])
    frame_stride = _read_optional_int(ctx.options.get("litevggt_frame_stride", profile_defaults["litevggt_frame_stride"]))
    depth_conf_thresh = _read_optional_float(
        ctx.options.get("litevggt_depth_conf_thresh"),
        profile_defaults["litevggt_depth_conf_thresh"],
    )
    preprocess_mode = str(ctx.options.get("litevggt_preprocess_mode") or profile_defaults["litevggt_preprocess_mode"])
    point_selection_strategy = str(
        ctx.options.get("litevggt_point_selection_strategy") or profile_defaults["litevggt_point_selection_strategy"]
    )
    inference_mode = str(ctx.options.get("litevggt_inference_mode") or profile_defaults["litevggt_inference_mode"])
    chunk_size = _read_int(ctx.options.get("litevggt_chunk_size"), profile_defaults["litevggt_chunk_size"])
    overlap = _read_int(ctx.options.get("litevggt_overlap"), profile_defaults["litevggt_overlap"])
    loop_closure = _read_bool(ctx.options.get("litevggt_loop_closure"), profile_defaults["litevggt_loop_closure"])
    keyframe_target = _read_optional_int(
        ctx.options.get("litevggt_keyframe_target", profile_defaults["litevggt_keyframe_target"])
    )
    min_frame_gap = _read_int(ctx.options.get("litevggt_min_frame_gap"), profile_defaults["litevggt_min_frame_gap"])
    min_scene_change = _read_float(
        ctx.options.get("litevggt_min_scene_change"),
        profile_defaults["litevggt_min_scene_change"],
    )
    window_voxel_diag_ratio = _read_float(
        ctx.options.get("litevggt_window_voxel_diag_ratio"),
        profile_defaults["litevggt_window_voxel_diag_ratio"],
    )
    final_voxel_diag_ratio = _read_float(
        ctx.options.get("litevggt_final_voxel_diag_ratio"),
        profile_defaults["litevggt_final_voxel_diag_ratio"],
    )
    axis_trim_low_quantile = _read_float(
        ctx.options.get("litevggt_axis_trim_low_quantile"),
        profile_defaults["litevggt_axis_trim_low_quantile"],
    )
    axis_trim_high_quantile = _read_float(
        ctx.options.get("litevggt_axis_trim_high_quantile"),
        profile_defaults["litevggt_axis_trim_high_quantile"],
    )
    spatial_keep_quantile = _read_float(
        ctx.options.get("litevggt_spatial_keep_quantile"),
        profile_defaults["litevggt_spatial_keep_quantile"],
    )
    params: dict[str, Any] = {
        "preview_scene_profile": scene_profile,
        "keep_ratio": keep_ratio,
        "max_points": max_points,
        "max_input_frames": max_input_frames,
        "target_size": target_size,
        "frame_stride": frame_stride if frame_stride is not None else "auto",
        "depth_conf_thresh": depth_conf_thresh,
        "preprocess_mode": preprocess_mode,
        "inference_mode": inference_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "loop_closure": loop_closure,
        "keyframe_target": keyframe_target,
        "min_frame_gap": min_frame_gap,
        "min_scene_change": min_scene_change,
        "window_voxel_diag_ratio": window_voxel_diag_ratio,
        "final_voxel_diag_ratio": final_voxel_diag_ratio,
        "point_selection_strategy": point_selection_strategy,
        "axis_trim_low_quantile": axis_trim_low_quantile,
        "axis_trim_high_quantile": axis_trim_high_quantile,
        "spatial_keep_quantile": spatial_keep_quantile,
    }
    print(
        "[litevggt-preview] adapter params "
        f"task_id={ctx.task_id} project_id={ctx.project_id} input_dir={ctx.input_dir} "
        f"image_count={len(files)} first_images={_first_names(files)} weight={weight} "
        f"weight_bytes={weight.stat().st_size} output_points_ply={points_ply_path} "
        f"output_splats_ply={splats_ply_path} output_spz={ctx.output_spz} "
        + " ".join(f"{key}={value}" for key, value in params.items()),
        flush=True,
    )
    ctx.report(
        "litevggt_preflight",
        22,
        f"LiteVGGT preview path: images={len(files)} mode={inference_mode} chunk={chunk_size} overlap={overlap} keep_ratio={params['keep_ratio']} max_points={max_points}",
    )

    def report(stage: str, progress: int, message: str) -> None:
        ctx.report(stage, progress, message)

    metrics = run_litevggt_pointcloud(
        input_dir=ctx.input_dir,
        checkpoint_path=weight,
        output_ply=points_ply_path,
        output_meta_json=meta_path,
        keep_ratio=keep_ratio,
        max_points=max_points,
        max_input_frames=max_input_frames,
        target_size=target_size,
        frame_stride=frame_stride,
        depth_conf_thresh=depth_conf_thresh,
        preprocess_mode=preprocess_mode,
        inference_mode=inference_mode,
        chunk_size=chunk_size,
        overlap=overlap,
        loop_closure=loop_closure,
        scene_profile=scene_profile,
        keyframe_target=keyframe_target,
        min_frame_gap=min_frame_gap,
        min_scene_change=min_scene_change,
        window_voxel_diag_ratio=window_voxel_diag_ratio,
        final_voxel_diag_ratio=final_voxel_diag_ratio,
        selection_strategy=point_selection_strategy,
        axis_trim_low_quantile=axis_trim_low_quantile,
        axis_trim_high_quantile=axis_trim_high_quantile,
        spatial_keep_quantile=spatial_keep_quantile,
        progress=report,
    )
    timer.mark("litevggt_inference")
    print(
        "[litevggt-preview] inference metrics "
        f"task_id={ctx.task_id} output_points_ply={points_ply_path} "
        f"points_ply_exists={points_ply_path.exists()} "
        f"points_ply_bytes={points_ply_path.stat().st_size if points_ply_path.exists() else None} "
        f"metrics={_format_metrics(metrics)}",
        flush=True,
    )

    base_point_radius = _read_bounded_float(metrics.get("litevggt_preview_point_radius"), 0.002, 1e-8, 1.0)
    point_radius_scale = _read_bounded_float(
        ctx.options.get("preview_fixed_splat_radius_scale", ctx.options.get("litevggt_point_radius_scale")),
        0.22,
        0.05,
        20.0,
    )
    fixed_splat_opacity = _read_bounded_float(ctx.options.get("preview_fixed_splat_opacity"), 0.55, 0.05, 0.99)
    point_radius = base_point_radius * point_radius_scale
    ctx.report("splat_conversion", 82, "converting LiteVGGT point cloud PLY to fixed Gaussian PLY")
    converted_splat_count = convert_pointcloud_ply_to_fixed_splat_ply(
        points_ply_path,
        splats_ply_path,
        point_radius=point_radius,
        opacity=fixed_splat_opacity,
    )
    timer.mark("fixed_splat_conversion")

    ctx.report("spz_conversion", 86, "converting LiteVGGT fixed Gaussian PLY to Spark SPZ")
    splat_count = convert_ply_to_spz(splats_ply_path, ctx.output_spz)
    timer.mark("spz_conversion")
    print(
        "[litevggt-preview] conversion complete "
        f"task_id={ctx.task_id} output_splats_ply={splats_ply_path} output_spz={ctx.output_spz} "
        f"splats_ply_exists={splats_ply_path.exists()} "
        f"splats_ply_bytes={splats_ply_path.stat().st_size if splats_ply_path.exists() else None} "
        f"spz_exists={ctx.output_spz.exists()} spz_bytes={ctx.output_spz.stat().st_size if ctx.output_spz.exists() else None} "
        f"splat_count={splat_count} stage_durations={timer.metrics().get('stage_durations')}",
        flush=True,
    )

    return PreviewResult(
        output_spz=ctx.output_spz,
        intermediate_ply=points_ply_path,
        splat_count=splat_count,
        metrics={
            **metrics,
            **timer.metrics(),
            "adapter": "litevggt_spz",
            "preview_scene_profile": scene_profile,
            "intermediate_ply_size": points_ply_path.stat().st_size,
            "intermediate_points_ply": str(points_ply_path),
            "intermediate_splats_ply": str(splats_ply_path),
            "preview_meta_json": str(meta_path),
            "intermediate_points_ply_size": points_ply_path.stat().st_size,
            "intermediate_splats_ply_size": splats_ply_path.stat().st_size,
            "preview_meta_json_size": meta_path.stat().st_size,
            "fixed_splat_base_point_radius": base_point_radius,
            "fixed_splat_point_radius_scale": point_radius_scale,
            "fixed_splat_point_radius": point_radius,
            "fixed_splat_opacity": fixed_splat_opacity,
            "fixed_splat_count": converted_splat_count,
        },
        source_commits={"LiteVGGT": SOURCE_COMMITS["LiteVGGT"], "Spark": SOURCE_COMMITS["Spark"]},
    )


def _resolve_scene_profile(options: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    profile = str(options.get("preview_scene_profile") or DEFAULT_PREVIEW_SCENE_PROFILE).strip().lower()
    if profile not in LITEVGGT_SCENE_PROFILE_DEFAULTS:
        profile = DEFAULT_PREVIEW_SCENE_PROFILE
    return profile, LITEVGGT_SCENE_PROFILE_DEFAULTS[profile]


def _first_names(paths, limit: int = 8) -> str:
    names = [path.name for path in paths[:limit]]
    suffix = "" if len(paths) <= limit else f", ... +{len(paths) - limit}"
    return "[" + ", ".join(names) + suffix + "]"


def _format_metrics(metrics: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted(metrics.items()))


def _read_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "auto", "none"}:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _read_int(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _read_float(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _read_bool(value: Any, fallback: bool) -> bool:
    if value is None:
        return bool(fallback)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(fallback)


def _read_bounded_float(value: Any, fallback: float, minimum: float, maximum: float) -> float:
    parsed = _read_float(value, fallback)
    return max(minimum, min(maximum, parsed))


def _read_optional_float(value: Any, fallback: float | None) -> float | None:
    if value is None:
        return fallback
    normalized = str(value).strip().lower()
    if normalized in {"", "auto"}:
        return fallback
    if normalized in {"none", "off", "false"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
