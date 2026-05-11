from __future__ import annotations

# 本文件为 LiteVGGT 极速预览推理入口，按 GarlicBa/LiteVGGT-repo 的 run_demo.py
# 关键流程改写：图像裁剪 -> VGGT camera/depth 推理 -> depth unproject -> 彩色点云 PLY。
# 上游仓库: https://github.com/GarlicBa/LiteVGGT-repo
# 固定提交: 4767c17f8b6f176bb751566e92f60eb885040033
# 许可证: MIT

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from app.preview.io.ply import write_gaussian_splat_ply
from app.preview.types import PreviewFailure
from app.preview.utils import VENDOR_ROOT, image_files, prepend_sys_path


Progress = Callable[[str, int, str], None]


@dataclass(slots=True)
class LiteVGGTReconstruction:
    files: list[Path]
    images: np.ndarray
    valid_masks: np.ndarray
    w2c: np.ndarray
    intrinsics: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    metrics: dict[str, int | float | str | bool]


def _load_small_gray(image_path: Path, size: int = 160) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    img.thumbnail((size, size), Image.Resampling.BILINEAR)

    canvas = Image.new("L", (size, size), 0)
    left = (size - img.width) // 2
    top = (size - img.height) // 2
    canvas.paste(img, (left, top))

    return np.asarray(canvas).astype(np.float32) / 255.0


def _sharpness_score(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray)
    return float(np.mean(gx * gx + gy * gy))


def _scene_change_score(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)

    if values.size == 0:
        return values

    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))

    if hi <= lo + 1e-8:
        return np.zeros_like(values)

    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def select_aligned_frames(
    files: list[Path],
    *,
    multiple: int = 8,
    max_frames: int | None = None,
    mode: str = "uniform",
) -> list[Path]:
    if len(files) <= multiple:
        return files

    usable = (len(files) // multiple) * multiple

    if max_frames is not None and max_frames > 0:
        usable = min(usable, (max_frames // multiple) * multiple)

    usable = max(multiple, usable)

    if mode == "head":
        return files[:usable]

    if mode == "tail":
        return files[-usable:]

    if usable >= len(files):
        return files[:usable]

    indices = np.linspace(0, len(files) - 1, usable)
    indices = np.round(indices).astype(int)
    indices = np.clip(indices, 0, len(files) - 1)

    selected: list[Path] = []
    seen: set[int] = set()

    for idx in indices:
        idx = int(idx)
        if idx not in seen:
            selected.append(files[idx])
            seen.add(idx)

    if len(selected) < usable:
        selected_set = set(selected)
        for file in files:
            if file not in selected_set:
                selected.append(file)
                selected_set.add(file)
                if len(selected) == usable:
                    break

    return selected[:usable]


def select_scene_aware_frames(
    files: list[Path],
    *,
    multiple: int = 8,
    max_frames: int | None = None,
    min_scene_change: float = 0.045,
    sharpness_drop_quantile: float = 0.15,
    sharpness_weight: float = 0.35,
    diversity_weight: float = 0.65,
) -> list[Path]:
    if len(files) <= multiple:
        return files

    target_count = (len(files) // multiple) * multiple

    if max_frames is not None and max_frames > 0:
        target_count = min(target_count, (max_frames // multiple) * multiple)

    target_count = max(multiple, target_count)

    if len(files) <= target_count:
        return files[:target_count]

    small_images: list[np.ndarray] = []
    sharpness_values: list[float] = []

    for file in files:
        gray = _load_small_gray(file)
        small_images.append(gray)
        sharpness_values.append(_sharpness_score(gray))

    sharpness = np.asarray(sharpness_values, dtype=np.float32)
    sharpness_norm = _normalize_scores(sharpness)

    scene_change = np.zeros(len(files), dtype=np.float32)
    scene_change[0] = 1.0

    for i in range(1, len(files)):
        scene_change[i] = _scene_change_score(small_images[i - 1], small_images[i])

    scene_change_norm = _normalize_scores(scene_change)

    sharpness_threshold = float(np.quantile(sharpness, sharpness_drop_quantile))
    candidate_indices = [
        i for i in range(len(files))
        if sharpness[i] >= sharpness_threshold
    ]

    if len(candidate_indices) < target_count:
        candidate_indices = list(range(len(files)))

    change_candidates = [
        i for i in candidate_indices
        if scene_change[i] >= min_scene_change
    ]

    if len(change_candidates) < target_count:
        change_candidates = candidate_indices

    scores = (
        sharpness_weight * sharpness_norm
        + diversity_weight * scene_change_norm
    )

    selected: list[int] = [0, len(files) - 1]

    bucket_count = target_count
    bucket_edges = np.linspace(0, len(files), bucket_count + 1).astype(int)

    for b in range(bucket_count):
        start = int(bucket_edges[b])
        end = int(bucket_edges[b + 1])

        bucket_indices = [
            i for i in change_candidates
            if start <= i < end and i not in selected
        ]

        if not bucket_indices:
            continue

        best = max(bucket_indices, key=lambda i: float(scores[i]))
        selected.append(best)

        if len(selected) >= target_count:
            break

    if len(selected) < target_count:
        remaining = [
            i for i in change_candidates
            if i not in selected
        ]
        remaining.sort(key=lambda i: float(scores[i]), reverse=True)

        for i in remaining:
            selected.append(i)
            if len(selected) >= target_count:
                break

    if len(selected) < target_count:
        uniform_indices = np.linspace(0, len(files) - 1, target_count)
        uniform_indices = np.round(uniform_indices).astype(int)

        for i in uniform_indices:
            i = int(i)
            if i not in selected:
                selected.append(i)
            if len(selected) >= target_count:
                break

    selected = sorted(set(selected))

    if len(selected) > target_count:
        selected_scores = [(i, float(scores[i])) for i in selected]
        selected_scores.sort(key=lambda item: item[1], reverse=True)
        selected = sorted(i for i, _ in selected_scores[:target_count])

    aligned_count = (len(selected) // multiple) * multiple
    aligned_count = max(multiple, aligned_count)

    selected = selected[:aligned_count]

    return [files[i] for i in selected]


def make_edge_mask(
    height: int,
    width: int,
    border_ratio: float = 0.18,
) -> np.ndarray:
    border_y = max(1, int(height * border_ratio))
    border_x = max(1, int(width * border_ratio))

    mask = np.zeros((height, width), dtype=bool)
    mask[:border_y, :] = True
    mask[-border_y:, :] = True
    mask[:, :border_x] = True
    mask[:, -border_x:] = True

    return mask


def select_points_per_frame(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    valid_pixels: np.ndarray,
    *,
    frame_count: int,
    keep_ratio: float,
    edge_keep_ratio: float,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = confidence.shape[0]
    pixels_per_frame = height * width

    selected_indices: list[np.ndarray] = []

    edge_mask = make_edge_mask(height, width, border_ratio=0.18).reshape(-1)

    for frame_index in range(frame_count):
        start = frame_index * pixels_per_frame
        end = min(start + pixels_per_frame, total)

        if start >= total:
            break

        frame_valid = valid_pixels[start:end]
        frame_conf = confidence[start:end]

        finite_local = np.isfinite(frame_conf)
        valid_local = np.where(frame_valid & finite_local)[0]

        if valid_local.size == 0:
            continue

        frame_keep_ratio = float(np.clip(keep_ratio, 0.01, 1.0))
        keep = max(1, int(valid_local.size * frame_keep_ratio))

        local_scores = frame_conf[valid_local]
        local_keep = valid_local[np.argsort(local_scores)[::-1][:keep]]

        if edge_keep_ratio > 0:
            frame_edge = edge_mask[: end - start]
            edge_valid = np.where(frame_valid & finite_local & frame_edge)[0]

            if edge_valid.size > 0:
                edge_keep = max(1, int(edge_valid.size * float(np.clip(edge_keep_ratio, 0.0, 1.0))))
                edge_scores = frame_conf[edge_valid]
                edge_selected = edge_valid[np.argsort(edge_scores)[::-1][:edge_keep]]
                local_keep = np.unique(np.concatenate([local_keep, edge_selected]))

        selected_indices.append(local_keep + start)

    if not selected_indices:
        raise PreviewFailure(
            "LITEVGGT_EMPTY_POINT_CLOUD",
            "LiteVGGT produced no valid points after per-frame filtering",
        )

    indices = np.concatenate(selected_indices, axis=0)

    return points[indices], colors[indices], confidence[indices]


def select_points_global(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    valid_pixels: np.ndarray,
    *,
    keep_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    valid = finite & valid_pixels

    if not np.any(valid):
        raise PreviewFailure(
            "LITEVGGT_EMPTY_POINT_CLOUD",
            "LiteVGGT produced no valid points after global filtering",
        )

    points = points[valid]
    colors = colors[valid]
    confidence = confidence[valid]

    keep_ratio = float(np.clip(keep_ratio, 0.01, 1.0))
    keep = max(1, int(len(confidence) * keep_ratio))

    keep_indices = np.argsort(confidence)[::-1][:keep]

    return points[keep_indices], colors[keep_indices], confidence[keep_indices]


def trim_axis_quantile_outliers(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    *,
    low_quantile: float = 0.0005,
    high_quantile: float = 0.9995,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)

    points = points[finite]
    colors = colors[finite]
    confidence = confidence[finite]

    if points.shape[0] <= 10:
        return points, colors, confidence

    mins = np.quantile(points, low_quantile, axis=0)
    maxs = np.quantile(points, high_quantile, axis=0)

    keep = np.all((points >= mins) & (points <= maxs), axis=1)

    if not np.any(keep):
        return points, colors, confidence

    return points[keep], colors[keep], confidence[keep]


def voxel_downsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    *,
    max_points: int,
    voxel_size: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors, confidence

    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)

    points = points[finite]
    colors = colors[finite]
    confidence = confidence[finite]

    if points.shape[0] <= max_points:
        return points, colors, confidence

    if voxel_size is None:
        lo = np.quantile(points, 0.01, axis=0)
        hi = np.quantile(points, 0.99, axis=0)
        extent = hi - lo
        scene_scale = float(np.linalg.norm(extent))
        voxel_size = max(scene_scale / 512.0, 1e-6)

    keys = np.floor(points / voxel_size).astype(np.int64)

    best_by_voxel: dict[tuple[int, int, int], int] = {}

    for idx, key_arr in enumerate(keys):
        key = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))
        previous = best_by_voxel.get(key)

        if previous is None or confidence[idx] > confidence[previous]:
            best_by_voxel[key] = idx

    selected = np.fromiter(best_by_voxel.values(), dtype=np.int64)

    if selected.size > max_points:
        order = np.argsort(confidence[selected])[::-1]
        selected = selected[order[:max_points]]

    return points[selected], colors[selected], confidence[selected]


def _run_litevggt_reconstruction(
    *,
    input_dir: Path,
    checkpoint_path: Path,
    keep_ratio: float,
    max_points: int,
    spatial_keep_quantile: float,
    preserve_full_image: bool,
    letterbox_size: int,
    max_input_frames: int | None,
    frame_selection: str,
    min_scene_change: float,
    edge_keep_ratio: float,
    axis_trim_low_quantile: float,
    axis_trim_high_quantile: float,
    selection_strategy: str,
    progress: Progress,
) -> LiteVGGTReconstruction:
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
        original_frame_count = len(files)

        if frame_selection == "scene":
            files = select_scene_aware_frames(
                files,
                multiple=8,
                max_frames=max_input_frames,
                min_scene_change=min_scene_change,
            )
        elif frame_selection == "uniform":
            files = select_aligned_frames(
                files,
                multiple=8,
                max_frames=max_input_frames,
                mode="uniform",
            )
        elif frame_selection == "tail":
            files = select_aligned_frames(
                files,
                multiple=8,
                max_frames=max_input_frames,
                mode="tail",
            )
        else:
            files = select_aligned_frames(
                files,
                multiple=8,
                max_frames=max_input_frames,
                mode="head",
            )

        aligned_count = len(files)

        if aligned_count < 8:
            raise PreviewFailure(
                "LITEVGGT_NOT_ENOUGH_FRAMES",
                f"LiteVGGT requires at least 8 images, got {aligned_count}",
            )

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
        valid_masks = []
        for index, image_path in enumerate(files):
            if preserve_full_image:
                image, valid_mask = load_image_file_letterbox(
                    str(image_path),
                    target_size=letterbox_size,
                )
            else:
                image = load_image_file_crop(str(image_path))
                valid_mask = np.ones(image.shape[:2], dtype=bool)

            image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
            mask_tensor = torch.from_numpy(valid_mask)
            images.append(image_tensor)
            valid_masks.append(mask_tensor)
            if index % 4 == 0:
                progress("litevggt_preprocess", 28 + int(index / max(1, len(files)) * 8), f"preprocessed {index + 1}/{len(files)} images")

        image_batch = torch.stack(images, dim=0).to(device)
        valid_mask_batch = torch.stack(valid_masks, dim=0).to(device)
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
            image_array = image_batch[0].permute(0, 2, 3, 1).detach().cpu().numpy()
            color_image = image_array.reshape(-1, 3)
            colors = np.clip(color_image * 255.0, 0, 255).astype(np.uint8)

            confidence = depth_conf.reshape(-1).detach().cpu().numpy()
            valid_mask_array = valid_mask_batch.detach().cpu().numpy().astype(bool)
            valid_pixels = valid_mask_array.reshape(-1)
            finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
            valid_pixels = valid_pixels & finite

            height = int(image_batch.shape[-2])
            width = int(image_batch.shape[-1])

            if selection_strategy == "global":
                selected_points, selected_colors, selected_confidence = select_points_global(
                    points,
                    colors,
                    confidence,
                    valid_pixels,
                    keep_ratio=keep_ratio,
                )
            else:
                selected_points, selected_colors, selected_confidence = select_points_per_frame(
                    points,
                    colors,
                    confidence,
                    valid_pixels,
                    frame_count=aligned_count,
                    keep_ratio=keep_ratio,
                    edge_keep_ratio=edge_keep_ratio,
                    height=height,
                    width=width,
                )

            if spatial_keep_quantile >= 1.0:
                trimmed_points, trimmed_colors, trimmed_confidence = trim_axis_quantile_outliers(
                    selected_points,
                    selected_colors,
                    selected_confidence,
                    low_quantile=axis_trim_low_quantile,
                    high_quantile=axis_trim_high_quantile,
                )
            else:
                trimmed_points, trimmed_colors, trimmed_confidence = trim_spatial_outliers(
                    selected_points,
                    selected_colors,
                    selected_confidence,
                    keep_quantile=spatial_keep_quantile,
                )

            point_count_after_spatial_trim = int(trimmed_points.shape[0])
            point_count_before_downsample = int(trimmed_points.shape[0])

            trimmed_points, trimmed_colors, trimmed_confidence = voxel_downsample_points(
                trimmed_points,
                trimmed_colors,
                trimmed_confidence,
                max_points=max_points,
            )

            point_count_after_downsample = int(trimmed_points.shape[0])

        peak_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0
        return LiteVGGTReconstruction(
            files=files,
            images=image_array,
            valid_masks=valid_mask_array,
            w2c=w2c_pre.squeeze(0).detach().float().cpu().numpy(),
            intrinsics=intrinsic.squeeze(0).detach().float().cpu().numpy(),
            points=trimmed_points,
            colors=trimmed_colors,
            confidence=trimmed_confidence,
            metrics={
                "original_frame_count": int(original_frame_count),
                "input_frame_count": aligned_count,
                "frame_selection": frame_selection,
                "min_scene_change": float(min_scene_change),
                "point_selection_strategy": selection_strategy,
                "keep_ratio": float(keep_ratio),
                "edge_keep_ratio": float(edge_keep_ratio),
                "spatial_keep_quantile": float(spatial_keep_quantile),
                "axis_trim_low_quantile": float(axis_trim_low_quantile),
                "axis_trim_high_quantile": float(axis_trim_high_quantile),
                "preserve_full_image": bool(preserve_full_image),
                "letterbox_size": int(letterbox_size),
                "valid_pixel_count": int(valid_pixels.sum()),
                "point_count_before_spatial_trim": int(selected_points.shape[0]),
                "point_count_after_spatial_trim": point_count_after_spatial_trim,
                "point_count_before_downsample": point_count_before_downsample,
                "point_count_after_downsample": point_count_after_downsample,
                "cuda_memory_peak_mb": float(peak_mb),
            },
        )


def run_litevggt_pointcloud(
    *,
    input_dir: Path,
    checkpoint_path: Path,
    output_ply: Path,
    keep_ratio: float,
    max_points: int,
    spatial_keep_quantile: float = 0.995,
    preserve_full_image: bool = True,
    letterbox_size: int = 518,
    max_input_frames: int | None = None,
    frame_selection: str = "scene",
    min_scene_change: float = 0.045,
    edge_keep_ratio: float = 0.15,
    axis_trim_low_quantile: float = 0.0005,
    axis_trim_high_quantile: float = 0.9995,
    selection_strategy: str = "per_frame",
    progress: Progress,
) -> dict[str, int | float | str | bool]:
    reconstruction = _run_litevggt_reconstruction(
        input_dir=input_dir,
        checkpoint_path=checkpoint_path,
        keep_ratio=keep_ratio,
        max_points=max_points,
        spatial_keep_quantile=spatial_keep_quantile,
        preserve_full_image=preserve_full_image,
        letterbox_size=letterbox_size,
        max_input_frames=max_input_frames,
        frame_selection=frame_selection,
        min_scene_change=min_scene_change,
        edge_keep_ratio=edge_keep_ratio,
        axis_trim_low_quantile=axis_trim_low_quantile,
        axis_trim_high_quantile=axis_trim_high_quantile,
        selection_strategy=selection_strategy,
        progress=progress,
    )
    point_count = write_gaussian_splat_ply(
        reconstruction.points,
        reconstruction.colors,
        output_ply,
        confidence=reconstruction.confidence,
        max_points=max_points,
    )
    return {**reconstruction.metrics, "point_count": point_count}


def require_transformer_engine() -> None:
    try:
        __import__("transformer_engine")
    except Exception as exc:  # pragma: no cover - depends on CUDA worker image
        raise PreviewFailure("TRANSFORMER_ENGINE_UNAVAILABLE", f"LiteVGGT requires transformer_engine: {exc}") from exc


def load_image_file_letterbox(
    img_path: str,
    *,
    target_size: int = 518,
    divisor: int = 14,
    pad_value: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(img_path)

    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    img = img.convert("RGB")
    width, height = img.size

    target_size = int(round(target_size / divisor) * divisor)
    target_size = max(divisor, target_size)

    scale = min(target_size / width, target_size / height)

    new_width = max(divisor, int(round(width * scale / divisor) * divisor))
    new_height = max(divisor, int(round(height * scale / divisor) * divisor))

    new_width = min(new_width, target_size)
    new_height = min(new_height, target_size)

    resized = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

    canvas = np.full(
        (target_size, target_size, 3),
        fill_value=pad_value,
        dtype=np.float32,
    )
    valid_mask = np.zeros(
        (target_size, target_size),
        dtype=bool,
    )

    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2

    arr = np.asarray(resized).astype(np.float32) / 255.0

    canvas[top : top + new_height, left : left + new_width, :] = arr
    valid_mask[top : top + new_height, left : left + new_width] = True

    return canvas, valid_mask


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
