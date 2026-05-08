from __future__ import annotations

# 本文件为 LiteVGGT 极速预览推理入口，按 GarlicBa/LiteVGGT-repo 的 run_demo.py
# 关键流程改写：图像裁剪 -> VGGT camera/depth 推理 -> depth unproject -> 彩色点云 PLY。
# 上游仓库: https://github.com/GarlicBa/LiteVGGT-repo
# 固定提交: 4767c17f8b6f176bb751566e92f60eb885040033
# 许可证: MIT

from pathlib import Path
from typing import Callable

import numpy as np

from app.preview.io.ply import write_point_cloud_ply
from app.preview.types import PreviewFailure
from app.preview.utils import VENDOR_ROOT, image_files, prepend_sys_path


Progress = Callable[[str, int, str], None]


def run_litevggt_pointcloud(
    *,
    input_dir: Path,
    checkpoint_path: Path,
    output_ply: Path,
    keep_ratio: float,
    max_points: int,
    spatial_keep_quantile: float = 0.995,
    progress: Progress,
) -> dict[str, int | float | str]:
    """执行 LiteVGGT 直接点云预览。

    这里没有调用原仓库脚本，而是把 run_demo.py 的关键步骤变成系统函数。
    LiteVGGT 原实现要求图像数量按 8 对齐，所以少于 8 张会直接失败。
    """

    import torch

    require_transformer_engine()
    with prepend_sys_path(VENDOR_ROOT / "litevggt"):
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format
        from vggt.models.vggt import VGGT
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_image_file_crop
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        files = image_files(input_dir)
        if len(files) < 8:
            raise PreviewFailure("LITEVGGT_NOT_ENOUGH_IMAGES", "LiteVGGT preview requires at least 8 images")
        aligned_count = (len(files) // 8) * 8
        files = files[:aligned_count]

        if not torch.cuda.is_available():
            raise PreviewFailure("GPU_RESOURCE_UNAVAILABLE", "LiteVGGT requires CUDA")
        device = "cuda:0"
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        progress("litevggt_loading_model", 24, f"loading LiteVGGT checkpoint: {checkpoint_path.name}")
        model = VGGT().to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        model.to(torch.bfloat16)
        model.eval()

        images = []
        for index, image_path in enumerate(files):
            image = load_image_file_crop(str(image_path))
            image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
            images.append(image_tensor)
            if index % 4 == 0:
                progress("litevggt_preprocess", 28 + int(index / max(1, len(files)) * 8), f"preprocessed {index + 1}/{len(files)} images")

        image_batch = torch.stack(images, dim=0).to(device)
        patch_width = image_batch.shape[-1] // 14
        patch_height = image_batch.shape[-2] // 14
        model.update_patch_dimensions(patch_width, patch_height)
        image_batch = image_batch[None]

        progress("litevggt_inference", 40, f"running LiteVGGT on {aligned_count} aligned images")
        with torch.no_grad():
            fp8_recipe = DelayedScaling(fp8_format=Format.E4M3, amax_history_len=80, amax_compute_algo="max")
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                aggregated_tokens_list, patch_start_idx = model.aggregator(image_batch)

            with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                w2c_pre, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_batch.shape[-2:])
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, image_batch, patch_start_idx)

            progress("litevggt_unproject", 58, "unprojecting depth maps to world point cloud")
            points_3d = unproject_depth_map_to_point_map(depth_map.squeeze(0), w2c_pre.squeeze(0), intrinsic.squeeze(0))
            points = points_3d.reshape(-1, 3)
            color_image = image_batch[0].permute(0, 2, 3, 1).reshape(-1, 3).detach().cpu().numpy()
            colors = np.clip(color_image * 255.0, 0, 255).astype(np.uint8)

            confidence = depth_conf.reshape(-1).detach().cpu().numpy()
            keep = max(1, int(len(confidence) * keep_ratio))
            keep_indices = np.argsort(confidence)[::-1][:keep]
            selected_points = points[keep_indices]
            selected_colors = colors[keep_indices]
            selected_confidence = confidence[keep_indices]
            trimmed_points, trimmed_colors, trimmed_confidence = trim_spatial_outliers(
                selected_points,
                selected_colors,
                selected_confidence,
                keep_quantile=spatial_keep_quantile,
            )
            point_count = write_point_cloud_ply(
                trimmed_points,
                trimmed_colors,
                output_ply,
                confidence=trimmed_confidence,
                max_points=max_points,
            )

        peak_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0
        return {
            "input_frame_count": aligned_count,
            "point_count": point_count,
            "keep_ratio": keep_ratio,
            "spatial_keep_quantile": spatial_keep_quantile,
            "point_count_before_spatial_trim": int(selected_points.shape[0]),
            "point_count_after_spatial_trim": int(trimmed_points.shape[0]),
            "cuda_memory_peak_mb": peak_mb,
        }


def require_transformer_engine() -> None:
    try:
        __import__("transformer_engine")
    except Exception as exc:  # pragma: no cover - depends on CUDA worker image
        raise PreviewFailure("TRANSFORMER_ENGINE_UNAVAILABLE", f"LiteVGGT requires transformer_engine: {exc}") from exc


def trim_spatial_outliers(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    *,
    keep_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep_quantile = float(np.clip(keep_quantile, 0.5, 1.0))
    if keep_quantile >= 1.0 or points.shape[0] <= 1:
        return points, colors, confidence

    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    if not np.any(finite):
        return points, colors, confidence
    points = points[finite]
    colors = colors[finite]
    confidence = confidence[finite]
    if points.shape[0] <= 1:
        return points, colors, confidence

    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    cutoff = float(np.quantile(distances, keep_quantile))
    keep = distances <= cutoff
    if not np.any(keep):
        return points, colors, confidence
    return points[keep], colors[keep], confidence[keep]
