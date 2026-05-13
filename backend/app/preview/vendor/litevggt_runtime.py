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
    frame_indices: np.ndarray
    images: np.ndarray
    valid_masks: np.ndarray
    w2c: np.ndarray
    intrinsics: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    point_frame_indices: np.ndarray
    metrics: dict[str, int | float | str | bool]


@dataclass(slots=True)
class LiteVGGTBatchResult:
    frame_indices: np.ndarray
    images: np.ndarray
    valid_masks: np.ndarray
    w2c: np.ndarray
    intrinsics: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    point_frame_indices: np.ndarray
    valid_pixel_count: int
    point_count_before_filter: int
    point_count_after_filter: int


@dataclass(slots=True)
class LiteVGGTWindowResult:
    start: int
    end: int
    images: np.ndarray
    valid_masks: np.ndarray
    w2c: np.ndarray
    intrinsics: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    valid_pixel_count: int
    point_count_before_filter: int
    point_count_after_filter: int
    frame_indices: np.ndarray | None = None
    point_frame_indices: np.ndarray | None = None


class LiteVGGTWindowFrameMismatch(RuntimeError):
    pass


def build_litevggt_windows(frame_count: int, window_size: int, overlap: int) -> list[tuple[int, int]]:
    frame_count = int(frame_count)
    window_size = max(8, int(window_size))
    overlap = max(0, min(int(overlap), window_size - 1))

    if frame_count <= 0:
        return []
    if frame_count <= window_size:
        return [(0, frame_count)]

    step = max(1, window_size - overlap)
    windows: list[tuple[int, int]] = []
    start = 0

    while start < frame_count:
        end = min(frame_count, start + window_size)
        if end - start < window_size:
            start = max(0, end - window_size)
        window = (start, end)
        if not windows or windows[-1] != window:
            windows.append(window)
        if end >= frame_count:
            break
        start += step

    return windows


def effective_litevggt_overlap(window_size: int, configured_overlap: int) -> int:
    window_size = max(8, int(window_size))
    configured_overlap = max(0, int(configured_overlap))
    return min(configured_overlap, max(3, window_size // 2))


def resolve_litevggt_window_attempts(initial_window_size: int, oom_window_sizes: list[int]) -> list[int]:
    attempts: list[int] = []
    for value in [initial_window_size, *oom_window_sizes]:
        window_size = max(8, int(value))
        window_size = max(8, (window_size // 8) * 8)
        if window_size not in attempts:
            attempts.append(window_size)
    return attempts


def _first_dim(array: np.ndarray) -> int | None:
    shape = getattr(array, "shape", ())
    if len(shape) == 0:
        return None
    return int(shape[0])


def _validate_litevggt_window_frame_counts(
    *,
    start: int,
    end: int,
    fields: dict[str, np.ndarray],
) -> None:
    expected = int(end) - int(start)
    mismatches = []
    for name, value in fields.items():
        actual = _first_dim(value)
        if actual != expected:
            mismatches.append(f"{name}={actual}")

    if mismatches:
        detail = ", ".join(mismatches)
        raise LiteVGGTWindowFrameMismatch(
            f"LiteVGGT window frames {start + 1}-{end} expected {expected} frames, got {detail}"
        )


def _validate_litevggt_window_result(result: LiteVGGTWindowResult) -> None:
    _validate_litevggt_window_frame_counts(
        start=result.start,
        end=result.end,
        fields={
            "images": result.images,
            "valid_masks": result.valid_masks,
            "w2c": result.w2c,
            "intrinsics": result.intrinsics,
        },
    )


def estimate_sim3_umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[valid]
    dst = dst[valid]

    if src.shape[0] < 3:
        raise PreviewFailure(
            "LITEVGGT_WINDOW_ALIGNMENT_FAILED",
            "LiteVGGT window alignment requires at least 3 overlapping camera centers",
        )

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    src_var = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if not np.isfinite(src_var) or src_var <= 1e-12:
        raise PreviewFailure(
            "LITEVGGT_WINDOW_ALIGNMENT_FAILED",
            "LiteVGGT window alignment source cameras are degenerate",
        )

    covariance = (src_centered.T @ dst_centered) / src.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(vt.T @ u.T) < 0:
        sign[-1] = -1.0
    rotation = vt.T @ np.diag(sign) @ u.T
    scale = float(np.sum(singular_values * sign) / src_var)
    translation = dst_mean - scale * (src_mean @ rotation.T)

    if not np.isfinite(scale) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise PreviewFailure("LITEVGGT_WINDOW_ALIGNMENT_FAILED", "LiteVGGT window alignment produced non-finite Sim(3)")

    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def camera_centers_from_w2c(w2c: np.ndarray) -> np.ndarray:
    matrices = w2c_to_homogeneous(w2c)
    rotations = matrices[:, :3, :3]
    translations = matrices[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), translations).astype(np.float32)


def w2c_to_homogeneous(w2c: np.ndarray) -> np.ndarray:
    matrices = np.asarray(w2c, dtype=np.float64)
    if matrices.ndim < 3:
        raise PreviewFailure("LITEVGGT_CAMERA_SHAPE_INVALID", f"LiteVGGT camera matrices must be 3D, got shape {matrices.shape}")
    if matrices.shape[-2:] == (4, 4):
        return matrices.reshape(-1, 4, 4)
    if matrices.shape[-2:] == (3, 4):
        matrices_3x4 = matrices.reshape(-1, 3, 4)
        homogeneous = np.tile(np.eye(4, dtype=np.float64), (matrices_3x4.shape[0], 1, 1))
        homogeneous[:, :3, :] = matrices_3x4
        return homogeneous
    raise PreviewFailure("LITEVGGT_CAMERA_SHAPE_INVALID", f"LiteVGGT camera matrices must be Nx3x4 or Nx4x4, got shape {matrices.shape}")


def transform_points_sim3(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return (float(scale) * (pts @ rotation.T) + translation).astype(np.float32, copy=False)


def compute_sim3_alignment_metrics(
    source: np.ndarray,
    target: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> dict[str, float]:
    src = np.asarray(source, dtype=np.float32).reshape(-1, 3)
    dst = np.asarray(target, dtype=np.float32).reshape(-1, 3)
    aligned = float(scale) * (src @ np.asarray(rotation, dtype=np.float32).reshape(3, 3).T) + np.asarray(translation, dtype=np.float32).reshape(3)
    residual = np.linalg.norm(aligned - dst, axis=1)
    target_extent = float(np.linalg.norm(np.max(dst, axis=0) - np.min(dst, axis=0))) if dst.size else 0.0
    denom = max(target_extent, 1e-6)
    return {
        "median": float(np.median(residual)) if residual.size else float("inf"),
        "p90": float(np.percentile(residual, 90)) if residual.size else float("inf"),
        "rel_median": float(np.median(residual) / denom) if residual.size else float("inf"),
        "rel_p90": float(np.percentile(residual, 90) / denom) if residual.size else float("inf"),
        "target_extent": target_extent,
    }


def validate_sim3_alignment(
    *,
    scale: float,
    metrics: dict[str, float],
    alignment_min_scale: float,
    alignment_max_scale: float,
    alignment_max_rel_median: float,
    alignment_max_rel_p90: float,
    code: str,
    message: str,
) -> None:
    if scale < alignment_min_scale or scale > alignment_max_scale:
        raise PreviewFailure(code, f"{message}: scale={scale:.4f}")
    if metrics["rel_median"] > alignment_max_rel_median or metrics["rel_p90"] > alignment_max_rel_p90:
        raise PreviewFailure(
            code,
            f"{message}: scale={scale:.4f}, rel_median={metrics['rel_median']:.4f}, rel_p90={metrics['rel_p90']:.4f}",
        )


def transform_w2c_sim3(w2c: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrices = w2c_to_homogeneous(w2c)
    align_rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    align_translation = np.asarray(translation, dtype=np.float64).reshape(3)
    transformed = np.empty_like(matrices)

    for index, matrix in enumerate(matrices):
        c2w = np.linalg.inv(matrix)
        center = c2w[:3, 3]
        c2w[:3, :3] = align_rotation @ c2w[:3, :3]
        c2w[:3, 3] = float(scale) * (align_rotation @ center) + align_translation
        transformed[index] = np.linalg.inv(c2w)

    return transformed.astype(np.float32)


def is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "cudnn" in message or "gpu" in message)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    return points[indices], colors[indices], confidence[indices], indices.astype(np.int64, copy=False)


def select_points_global(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    valid_pixels: np.ndarray,
    *,
    keep_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    valid = finite & valid_pixels

    if not np.any(valid):
        raise PreviewFailure(
            "LITEVGGT_EMPTY_POINT_CLOUD",
            "LiteVGGT produced no valid points after global filtering",
        )

    valid_indices = np.where(valid)[0]
    points = points[valid]
    colors = colors[valid]
    confidence = confidence[valid]

    keep_ratio = float(np.clip(keep_ratio, 0.01, 1.0))
    keep = max(1, int(len(confidence) * keep_ratio))

    keep_indices = np.argsort(confidence)[::-1][:keep]

    selected_indices = valid_indices[keep_indices]

    return points[keep_indices], colors[keep_indices], confidence[keep_indices], selected_indices.astype(np.int64, copy=False)


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


def reset_litevggt_aggregator_cache(model) -> None:
    aggregator = getattr(model, "aggregator", None)
    if aggregator is not None and hasattr(aggregator, "m_u"):
        aggregator.m_u = None


def point_indices_to_frame_indices(
    selected_pixel_indices: np.ndarray,
    *,
    frame_indices: list[int],
    height: int,
    width: int,
) -> np.ndarray:
    pixels_per_frame = int(height) * int(width)
    if pixels_per_frame <= 0:
        raise PreviewFailure("LITEVGGT_INVALID_IMAGE_SHAPE", "LiteVGGT point-frame mapping requires positive image dimensions")
    local_frame_ids = np.asarray(selected_pixel_indices, dtype=np.int64) // pixels_per_frame
    original = np.asarray(frame_indices, dtype=np.int32)
    if np.any(local_frame_ids < 0) or np.any(local_frame_ids >= original.shape[0]):
        raise PreviewFailure("LITEVGGT_POINT_FRAME_MAPPING_FAILED", "LiteVGGT selected point indices exceed batch frame count")
    return original[local_frame_ids]


def _select_litevggt_points(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    valid_pixels: np.ndarray,
    frame_count: int,
    keep_ratio: float,
    edge_keep_ratio: float,
    height: int,
    width: int,
    selection_strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if selection_strategy == "global":
        return select_points_global(
            points,
            colors,
            confidence,
            valid_pixels,
            keep_ratio=keep_ratio,
        )
    return select_points_per_frame(
        points,
        colors,
        confidence,
        valid_pixels,
        frame_count=frame_count,
        keep_ratio=keep_ratio,
        edge_keep_ratio=edge_keep_ratio,
        height=height,
        width=width,
    )


def _run_litevggt_batch(
    *,
    model,
    image_tensors: list,
    valid_mask_tensors: list,
    frame_indices: list[int],
    device: str,
    dtype,
    keep_ratio: float,
    edge_keep_ratio: float,
    selection_strategy: str,
    te,
    DelayedScaling,
    Format,
    unproject_depth_map_to_point_map,
    pose_encoding_to_extri_intri,
) -> LiteVGGTBatchResult:
    import torch

    reset_litevggt_aggregator_cache(model)

    image_batch = None
    valid_mask_batch = None
    aggregated_tokens_list = None
    pose_enc = None
    w2c_pre = None
    intrinsic = None
    depth_map = None
    depth_conf = None
    points_3d = None

    try:
        if len(image_tensors) != len(valid_mask_tensors) or len(image_tensors) != len(frame_indices):
            raise LiteVGGTWindowFrameMismatch(
                f"LiteVGGT batch expected aligned tensors and frame indices, got "
                f"images={len(image_tensors)}, masks={len(valid_mask_tensors)}, frame_indices={len(frame_indices)}"
            )
        image_batch = torch.stack(image_tensors, dim=0).to(device)
        valid_mask_batch = torch.stack(valid_mask_tensors, dim=0).to(device)
        patch_width = image_batch.shape[-1] // 14
        patch_height = image_batch.shape[-2] // 14
        model.update_patch_dimensions(patch_width, patch_height)
        image_batch = image_batch[None]

        with torch.no_grad():
            fp8_recipe = DelayedScaling(fp8_format=Format.E4M3, amax_history_len=80, amax_compute_algo="max")
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                aggregated_tokens_list, patch_start_idx = model.aggregator(image_batch)

            with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                w2c_pre, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_batch.shape[-2:])
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, image_batch, patch_start_idx)

            points_3d = unproject_depth_map_to_point_map(depth_map.squeeze(0), w2c_pre.squeeze(0), intrinsic.squeeze(0))
            points = points_3d.reshape(-1, 3)
            image_array = image_batch[0].permute(0, 2, 3, 1).detach().cpu().numpy()
            color_image = image_array.reshape(-1, 3)
            colors = np.clip(color_image * 255.0, 0, 255).astype(np.uint8)
            valid_mask_array = valid_mask_batch.detach().cpu().numpy().astype(bool)
            w2c_array = w2c_pre.squeeze(0).detach().float().cpu().numpy()
            intrinsic_array = intrinsic.squeeze(0).detach().float().cpu().numpy()

            _validate_litevggt_window_frame_counts(
                start=0,
                end=len(frame_indices),
                fields={
                    "images": image_array,
                    "valid_masks": valid_mask_array,
                    "w2c": w2c_array,
                    "intrinsics": intrinsic_array,
                    "points_3d": points_3d,
                    "depth_conf": depth_conf.squeeze(0),
                },
            )

            confidence = depth_conf.reshape(-1).detach().cpu().numpy()
            valid_pixels = valid_mask_array.reshape(-1)
            finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
            valid_pixels = valid_pixels & finite

            height = int(image_batch.shape[-2])
            width = int(image_batch.shape[-1])
            selected_points, selected_colors, selected_confidence, selected_indices = _select_litevggt_points(
                points=points,
                colors=colors,
                confidence=confidence,
                valid_pixels=valid_pixels,
                frame_count=end - start,
                keep_ratio=keep_ratio,
                edge_keep_ratio=edge_keep_ratio,
                height=height,
                width=width,
                selection_strategy=selection_strategy,
            )
            point_frame_indices = point_indices_to_frame_indices(
                selected_indices,
                frame_indices=frame_indices,
                height=height,
                width=width,
            )

            return LiteVGGTBatchResult(
                frame_indices=np.asarray(frame_indices, dtype=np.int32),
                images=image_array,
                valid_masks=valid_mask_array,
                w2c=w2c_array,
                intrinsics=intrinsic_array,
                points=selected_points,
                colors=selected_colors,
                confidence=selected_confidence,
                point_frame_indices=point_frame_indices,
                valid_pixel_count=int(valid_pixels.sum()),
                point_count_before_filter=int(points.shape[0]),
                point_count_after_filter=int(selected_points.shape[0]),
            )
    finally:
        if image_batch is not None:
            del image_batch
        if valid_mask_batch is not None:
            del valid_mask_batch
        aggregated_tokens_list = None
        pose_enc = None
        w2c_pre = None
        intrinsic = None
        depth_map = None
        depth_conf = None
        points_3d = None
        reset_litevggt_aggregator_cache(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_litevggt_window(
    *,
    model,
    image_tensors: list,
    valid_mask_tensors: list,
    start: int,
    end: int,
    frame_indices: list[int] | None = None,
    device: str,
    dtype,
    keep_ratio: float,
    edge_keep_ratio: float,
    selection_strategy: str,
    te,
    DelayedScaling,
    Format,
    unproject_depth_map_to_point_map,
    pose_encoding_to_extri_intri,
) -> LiteVGGTWindowResult:
    batch = _run_litevggt_batch(
        model=model,
        image_tensors=image_tensors[start:end],
        valid_mask_tensors=valid_mask_tensors[start:end],
        frame_indices=(frame_indices[start:end] if frame_indices is not None else list(range(start, end))),
        device=device,
        dtype=dtype,
        keep_ratio=keep_ratio,
        edge_keep_ratio=edge_keep_ratio,
        selection_strategy=selection_strategy,
        te=te,
        DelayedScaling=DelayedScaling,
        Format=Format,
        unproject_depth_map_to_point_map=unproject_depth_map_to_point_map,
        pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
    )
    return LiteVGGTWindowResult(
        start=start,
        end=end,
        frame_indices=batch.frame_indices,
        images=batch.images,
        valid_masks=batch.valid_masks,
        w2c=batch.w2c,
        intrinsics=batch.intrinsics,
        points=batch.points,
        colors=batch.colors,
        confidence=batch.confidence,
        point_frame_indices=batch.point_frame_indices,
        valid_pixel_count=batch.valid_pixel_count,
        point_count_before_filter=batch.point_count_before_filter,
        point_count_after_filter=batch.point_count_after_filter,
    )


def _merge_litevggt_windows(
    window_results: list[LiteVGGTWindowResult],
    frame_count: int,
    *,
    alignment_max_rel_median: float = 0.05,
    alignment_max_rel_p90: float = 0.12,
    alignment_min_scale: float = 0.25,
    alignment_max_scale: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not window_results:
        raise PreviewFailure("LITEVGGT_EMPTY_WINDOWS", "LiteVGGT windowed inference produced no windows")

    images_by_frame: list[np.ndarray | None] = [None] * frame_count
    masks_by_frame: list[np.ndarray | None] = [None] * frame_count
    w2c_by_frame: list[np.ndarray | None] = [None] * frame_count
    intrinsics_by_frame: list[np.ndarray | None] = [None] * frame_count
    centers_by_frame: list[np.ndarray | None] = [None] * frame_count
    merged_points: list[np.ndarray] = []
    merged_colors: list[np.ndarray] = []
    merged_confidence: list[np.ndarray] = []

    for window_index, result in enumerate(window_results):
        _validate_litevggt_window_result(result)
        local_w2c = result.w2c
        local_points = result.points

        if window_index == 0:
            scale = 1.0
            rotation = np.eye(3, dtype=np.float32)
            translation = np.zeros(3, dtype=np.float32)
        else:
            local_centers = camera_centers_from_w2c(local_w2c)
            source_centers = []
            target_centers = []
            for local_index, frame_index in enumerate(range(result.start, result.end)):
                global_center = centers_by_frame[frame_index]
                if global_center is not None:
                    source_centers.append(local_centers[local_index])
                    target_centers.append(global_center)
            scale, rotation, translation = estimate_sim3_umeyama(
                np.asarray(source_centers, dtype=np.float32),
                np.asarray(target_centers, dtype=np.float32),
            )
            metrics = compute_sim3_alignment_metrics(
                source=np.asarray(source_centers, dtype=np.float32),
                target=np.asarray(target_centers, dtype=np.float32),
                scale=scale,
                rotation=rotation,
                translation=translation,
            )
            validate_sim3_alignment(
                scale=scale,
                metrics=metrics,
                alignment_min_scale=alignment_min_scale,
                alignment_max_scale=alignment_max_scale,
                alignment_max_rel_median=alignment_max_rel_median,
                alignment_max_rel_p90=alignment_max_rel_p90,
                code="LITEVGGT_WINDOW_ALIGNMENT_UNSTABLE",
                message=f"unstable window alignment: window={window_index}",
            )

        transformed_w2c = transform_w2c_sim3(local_w2c, scale, rotation, translation)
        transformed_centers = camera_centers_from_w2c(transformed_w2c)
        transformed_points = transform_points_sim3(local_points, scale, rotation, translation)
        _validate_litevggt_window_frame_counts(
            start=result.start,
            end=result.end,
            fields={
                "transformed_w2c": transformed_w2c,
                "transformed_centers": transformed_centers,
            },
        )

        for local_index, frame_index in enumerate(range(result.start, result.end)):
            if images_by_frame[frame_index] is None:
                images_by_frame[frame_index] = result.images[local_index]
                masks_by_frame[frame_index] = result.valid_masks[local_index]
                w2c_by_frame[frame_index] = transformed_w2c[local_index]
                intrinsics_by_frame[frame_index] = result.intrinsics[local_index]
                centers_by_frame[frame_index] = transformed_centers[local_index]

        merged_points.append(transformed_points)
        merged_colors.append(result.colors)
        merged_confidence.append(result.confidence)

    missing = [index for index, image in enumerate(images_by_frame) if image is None]
    if missing:
        raise PreviewFailure("LITEVGGT_WINDOW_COVERAGE_FAILED", f"LiteVGGT windows did not cover frames: {missing[:8]}")

    return (
        np.stack([image for image in images_by_frame if image is not None], axis=0),
        np.stack([mask for mask in masks_by_frame if mask is not None], axis=0),
        np.stack([matrix for matrix in w2c_by_frame if matrix is not None], axis=0),
        np.stack([intrinsic for intrinsic in intrinsics_by_frame if intrinsic is not None], axis=0),
        np.concatenate(merged_points, axis=0),
        np.concatenate(merged_colors, axis=0),
        np.concatenate(merged_confidence, axis=0),
    )


def resolve_litevggt_effective_mode(
    *,
    inference_mode: str,
    aligned_count: int,
    single_frame_limit: int,
    hierarchical_enable: bool,
) -> str:
    requested = str(inference_mode or "auto").strip().lower()
    if requested in {"single", "global_keyframe", "hierarchical", "windowed"}:
        return requested
    if requested != "auto":
        raise PreviewFailure("LITEVGGT_INFERENCE_MODE_INVALID", f"Unsupported LiteVGGT inference mode: {inference_mode}")
    if int(aligned_count) <= int(single_frame_limit):
        return "single"
    if hierarchical_enable:
        return "hierarchical"
    return "global_keyframe"


def select_global_keyframe_indices(
    files: list[Path],
    *,
    multiple: int,
    max_frames: int,
    min_scene_change: float,
) -> list[int]:
    selected_files = select_scene_aware_frames(
        files,
        multiple=multiple,
        max_frames=max_frames,
        min_scene_change=min_scene_change,
    )
    file_to_index = {file: index for index, file in enumerate(files)}
    return [file_to_index[file] for file in selected_files]


def build_litevggt_chunks(
    frame_count: int,
    *,
    chunk_size: int,
    overlap: int,
    exclude_indices: set[int] | None = None,
) -> list[list[int]]:
    frame_count = max(0, int(frame_count))
    chunk_size = max(8, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))
    excluded = exclude_indices or set()
    if frame_count <= 0:
        return []
    step = max(1, chunk_size - overlap)
    chunks: list[list[int]] = []
    start = 0
    while start < frame_count:
        end = min(frame_count, start + chunk_size)
        if end - start < chunk_size and frame_count >= chunk_size:
            start = max(0, frame_count - chunk_size)
            end = frame_count
        indices = [index for index in range(start, end) if index not in excluded]
        if indices and (not chunks or chunks[-1] != indices):
            chunks.append(indices)
        if end >= frame_count:
            break
        start += step
    return chunks


def nearest_anchor_indices(
    chunk_indices: list[int],
    global_indices: list[int],
    *,
    anchor_count: int,
) -> list[int]:
    if not chunk_indices or not global_indices:
        return []
    center = 0.5 * (chunk_indices[0] + chunk_indices[-1])
    anchors = sorted(global_indices, key=lambda i: abs(i - center))[: max(0, int(anchor_count))]
    return sorted(anchors)


def surrounding_anchor_indices(
    chunk_indices: list[int],
    global_indices: list[int],
    *,
    anchor_count: int,
) -> list[int]:
    if not chunk_indices or not global_indices:
        return []
    anchor_count = max(0, int(anchor_count))
    half = max(1, anchor_count // 2)
    before = [i for i in global_indices if i <= chunk_indices[0]]
    after = [i for i in global_indices if i >= chunk_indices[-1]]
    selected = before[-half:] + after[:half]
    if len(set(selected)) < anchor_count:
        selected.extend(nearest_anchor_indices(chunk_indices, global_indices, anchor_count=anchor_count))
    return sorted(set(selected))[:anchor_count]


def align_indices_to_multiple_of_8(batch_indices: list[int], available_indices: list[int]) -> list[int]:
    selected = sorted(set(int(index) for index in batch_indices))
    available = sorted(set(int(index) for index in available_indices))
    if len(selected) < 8:
        for index in available:
            if index not in selected:
                selected.append(index)
            if len(selected) >= 8:
                break
    remainder = len(selected) % 8
    if remainder:
        needed = 8 - remainder
        center = 0.5 * (selected[0] + selected[-1]) if selected else 0.0
        fill = [index for index in available if index not in selected]
        fill.sort(key=lambda index: abs(index - center))
        selected.extend(fill[:needed])
    return sorted(selected[: (len(selected) // 8) * 8])


def _subset_tensors_by_frame(
    frame_indices: list[int],
    *,
    image_tensors_by_frame: dict[int, object],
    valid_mask_tensors_by_frame: dict[int, object],
) -> tuple[list, list]:
    return (
        [image_tensors_by_frame[int(index)] for index in frame_indices],
        [valid_mask_tensors_by_frame[int(index)] for index in frame_indices],
    )


def _run_litevggt_single_mode(
    *,
    model,
    frame_indices: list[int],
    image_tensors_by_frame: dict[int, object],
    valid_mask_tensors_by_frame: dict[int, object],
    progress: Progress,
    **batch_kwargs,
) -> LiteVGGTBatchResult:
    image_tensors, valid_mask_tensors = _subset_tensors_by_frame(
        frame_indices,
        image_tensors_by_frame=image_tensors_by_frame,
        valid_mask_tensors_by_frame=valid_mask_tensors_by_frame,
    )
    progress("litevggt_inference", 40, f"running LiteVGGT single on {len(frame_indices)} aligned images")
    return _run_litevggt_batch(
        model=model,
        image_tensors=image_tensors,
        valid_mask_tensors=valid_mask_tensors,
        frame_indices=frame_indices,
        **batch_kwargs,
    )


def _run_litevggt_global_keyframe_mode(
    *,
    model,
    files: list[Path],
    frame_indices: list[int],
    global_keyframe_count: int,
    min_scene_change: float,
    image_tensors_by_frame: dict[int, object],
    valid_mask_tensors_by_frame: dict[int, object],
    progress: Progress,
    **batch_kwargs,
) -> LiteVGGTBatchResult:
    positions = select_global_keyframe_indices(
        files,
        multiple=8,
        max_frames=global_keyframe_count,
        min_scene_change=min_scene_change,
    )
    selected_indices = [frame_indices[position] for position in positions]
    image_tensors, valid_mask_tensors = _subset_tensors_by_frame(
        selected_indices,
        image_tensors_by_frame=image_tensors_by_frame,
        valid_mask_tensors_by_frame=valid_mask_tensors_by_frame,
    )
    progress(
        "litevggt_inference",
        40,
        f"running LiteVGGT global_keyframe on {len(selected_indices)}/{len(frame_indices)} images",
    )
    return _run_litevggt_batch(
        model=model,
        image_tensors=image_tensors,
        valid_mask_tensors=valid_mask_tensors,
        frame_indices=selected_indices,
        **batch_kwargs,
    )


def align_local_result_to_global(
    local_result: LiteVGGTBatchResult,
    *,
    global_pose_by_frame: dict[int, np.ndarray],
    anchor_indices: list[int],
    alignment_min_scale: float,
    alignment_max_scale: float,
    alignment_max_rel_median: float,
    alignment_max_rel_p90: float,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    local_centers_all = camera_centers_from_w2c(local_result.w2c)
    anchor_set = set(int(index) for index in anchor_indices)
    source_centers = []
    target_centers = []
    for local_i, frame_idx in enumerate(local_result.frame_indices):
        frame_idx = int(frame_idx)
        if frame_idx not in anchor_set or frame_idx not in global_pose_by_frame:
            continue
        source_centers.append(local_centers_all[local_i])
        target_centers.append(camera_centers_from_w2c(global_pose_by_frame[frame_idx][None])[0])
    if len(source_centers) < 3:
        raise PreviewFailure("LITEVGGT_HIERARCHICAL_ALIGNMENT_FAILED", "LiteVGGT hierarchical alignment requires at least 3 anchors")
    source = np.asarray(source_centers, dtype=np.float32)
    target = np.asarray(target_centers, dtype=np.float32)
    scale, rotation, translation = estimate_sim3_umeyama(source, target)
    metrics = compute_sim3_alignment_metrics(source, target, scale, rotation, translation)
    validate_sim3_alignment(
        scale=scale,
        metrics=metrics,
        alignment_min_scale=alignment_min_scale,
        alignment_max_scale=alignment_max_scale,
        alignment_max_rel_median=alignment_max_rel_median,
        alignment_max_rel_p90=alignment_max_rel_p90,
        code="LITEVGGT_HIERARCHICAL_ALIGNMENT_UNSTABLE",
        message="unstable hierarchical chunk alignment",
    )
    return scale, rotation, translation, metrics


def filter_local_non_anchor_points(
    local_result: LiteVGGTBatchResult,
    *,
    anchor_indices: set[int],
    chunk_indices: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep = np.asarray(
        [int(frame_idx) in chunk_indices and int(frame_idx) not in anchor_indices for frame_idx in local_result.point_frame_indices],
        dtype=bool,
    )
    return (
        local_result.points[keep],
        local_result.colors[keep],
        local_result.confidence[keep],
        local_result.point_frame_indices[keep],
    )


def _run_litevggt_hierarchical_mode(
    *,
    model,
    files: list[Path],
    frame_indices: list[int],
    global_keyframe_count: int,
    min_scene_change: float,
    chunk_size: int,
    chunk_overlap: int,
    anchor_count: int,
    alignment_min_scale: float,
    alignment_max_scale: float,
    alignment_max_rel_median: float,
    alignment_max_rel_p90: float,
    image_tensors_by_frame: dict[int, object],
    valid_mask_tensors_by_frame: dict[int, object],
    progress: Progress,
    **batch_kwargs,
) -> tuple[LiteVGGTBatchResult, dict[str, int | float]]:
    global_result = _run_litevggt_global_keyframe_mode(
        model=model,
        files=files,
        frame_indices=frame_indices,
        global_keyframe_count=global_keyframe_count,
        min_scene_change=min_scene_change,
        image_tensors_by_frame=image_tensors_by_frame,
        valid_mask_tensors_by_frame=valid_mask_tensors_by_frame,
        progress=progress,
        **batch_kwargs,
    )
    global_indices = [int(index) for index in global_result.frame_indices]
    chunks_by_position = build_litevggt_chunks(len(frame_indices), chunk_size=chunk_size, overlap=chunk_overlap)
    chunks = [[frame_indices[position] for position in chunk] for chunk in chunks_by_position]
    progress(
        "litevggt_inference",
        42,
        f"running LiteVGGT hierarchical: global={len(global_indices)}, chunks={len(chunks)}, chunk_size={chunk_size}, anchors={anchor_count}",
    )
    global_pose_by_frame = {int(frame_idx): global_result.w2c[local_i] for local_i, frame_idx in enumerate(global_result.frame_indices)}
    all_points = [global_result.points]
    all_colors = [global_result.colors]
    all_confidence = [global_result.confidence]
    all_point_frame_indices = [global_result.point_frame_indices]
    accepted_chunk_count = 0
    rejected_chunk_count = 0
    rel_median_values: list[float] = []
    rel_p90_values: list[float] = []
    scale_values: list[float] = []
    available_indices = list(frame_indices)
    for chunk_index, chunk_indices in enumerate(chunks):
        current_anchor_count = anchor_count
        local_result = None
        alignment = None
        for attempt in range(2):
            anchor_indices = surrounding_anchor_indices(chunk_indices, global_indices, anchor_count=current_anchor_count)
            batch_indices = align_indices_to_multiple_of_8(anchor_indices + chunk_indices, available_indices)
            image_tensors, valid_mask_tensors = _subset_tensors_by_frame(
                batch_indices,
                image_tensors_by_frame=image_tensors_by_frame,
                valid_mask_tensors_by_frame=valid_mask_tensors_by_frame,
            )
            try:
                local_result = _run_litevggt_batch(
                    model=model,
                    image_tensors=image_tensors,
                    valid_mask_tensors=valid_mask_tensors,
                    frame_indices=batch_indices,
                    **batch_kwargs,
                )
                alignment = align_local_result_to_global(
                    local_result,
                    global_pose_by_frame=global_pose_by_frame,
                    anchor_indices=anchor_indices,
                    alignment_min_scale=alignment_min_scale,
                    alignment_max_scale=alignment_max_scale,
                    alignment_max_rel_median=alignment_max_rel_median,
                    alignment_max_rel_p90=alignment_max_rel_p90,
                )
                break
            except PreviewFailure as exc:
                if attempt == 0:
                    current_anchor_count *= 2
                    continue
                rejected_chunk_count += 1
                progress("litevggt_inference", 43, f"chunk {chunk_index + 1}/{len(chunks)} rejected: {exc.message}")
        if local_result is None or alignment is None:
            continue
        scale, rotation, translation, metrics = alignment
        local_points, local_colors, local_confidence, local_point_frame_indices = filter_local_non_anchor_points(
            local_result,
            anchor_indices=set(anchor_indices) | set(global_indices),
            chunk_indices=set(chunk_indices),
        )
        if local_points.size:
            all_points.append(transform_points_sim3(local_points, scale, rotation, translation))
            all_colors.append(local_colors)
            all_confidence.append(local_confidence)
            all_point_frame_indices.append(local_point_frame_indices)
        accepted_chunk_count += 1
        rel_median_values.append(float(metrics["rel_median"]))
        rel_p90_values.append(float(metrics["rel_p90"]))
        scale_values.append(float(scale))
        progress(
            "litevggt_inference",
            43 + int((chunk_index + 1) / max(1, len(chunks)) * 12),
            f"chunk {chunk_index + 1}/{len(chunks)} accepted: scale={scale:.4f}, rel_median={metrics['rel_median']:.4f}, rel_p90={metrics['rel_p90']:.4f}",
        )
    merged = LiteVGGTBatchResult(
        frame_indices=global_result.frame_indices,
        images=global_result.images,
        valid_masks=global_result.valid_masks,
        w2c=global_result.w2c,
        intrinsics=global_result.intrinsics,
        points=np.concatenate(all_points, axis=0),
        colors=np.concatenate(all_colors, axis=0),
        confidence=np.concatenate(all_confidence, axis=0),
        point_frame_indices=np.concatenate(all_point_frame_indices, axis=0),
        valid_pixel_count=global_result.valid_pixel_count,
        point_count_before_filter=global_result.point_count_before_filter,
        point_count_after_filter=int(sum(points.shape[0] for points in all_points)),
    )
    metrics: dict[str, int | float] = {
        "chunk_count": int(len(chunks)),
        "accepted_chunk_count": int(accepted_chunk_count),
        "rejected_chunk_count": int(rejected_chunk_count),
        "alignment_rel_median_max": float(max(rel_median_values) if rel_median_values else 0.0),
        "alignment_rel_p90_max": float(max(rel_p90_values) if rel_p90_values else 0.0),
        "alignment_scale_min": float(min(scale_values) if scale_values else 1.0),
        "alignment_scale_max": float(max(scale_values) if scale_values else 1.0),
    }
    return merged, metrics


def _run_litevggt_windowed_mode_with_quality_checks(
    *,
    model,
    image_tensors: list,
    valid_mask_tensors: list,
    frame_indices: list[int],
    aligned_count: int,
    window_size: int,
    window_overlap: int,
    oom_window_sizes: list[int],
    alignment_max_rel_median: float,
    alignment_max_rel_p90: float,
    alignment_min_scale: float,
    alignment_max_scale: float,
    progress: Progress,
    **batch_kwargs,
) -> tuple[LiteVGGTBatchResult, dict[str, int | float]]:
    import torch

    attempts = resolve_litevggt_window_attempts(window_size, oom_window_sizes)
    last_oom: RuntimeError | None = None
    oom_retry_count = 0
    window_frame_retry_count = 0
    for attempt_index, attempt_window_size in enumerate(attempts):
        selected_attempt = min(int(attempt_window_size), aligned_count)
        selected_overlap = effective_litevggt_overlap(selected_attempt, window_overlap)
        windows = build_litevggt_windows(aligned_count, selected_attempt, selected_overlap)
        progress(
            "litevggt_inference",
            40,
            f"running LiteVGGT windowed on {aligned_count} aligned images "
            f"(window_size={selected_attempt}, overlap={selected_overlap}, windows={len(windows)})",
        )
        try:
            window_results = []
            for window_index, (start, end) in enumerate(windows):
                progress(
                    "litevggt_inference",
                    40 + int(window_index / max(1, len(windows)) * 14),
                    f"running LiteVGGT window {window_index + 1}/{len(windows)}: frames {start + 1}-{end}",
                )
                window_results.append(
                    _run_litevggt_window(
                        model=model,
                        image_tensors=image_tensors,
                        valid_mask_tensors=valid_mask_tensors,
                        start=start,
                        end=end,
                        frame_indices=frame_indices,
                        **batch_kwargs,
                    )
                )
            progress("litevggt_unproject", 58, "aligning LiteVGGT windows and merging point clouds")
            image_array, valid_mask_array, w2c_array, intrinsic_array, points, colors, confidence = _merge_litevggt_windows(
                window_results,
                aligned_count,
                alignment_max_rel_median=alignment_max_rel_median,
                alignment_max_rel_p90=alignment_max_rel_p90,
                alignment_min_scale=alignment_min_scale,
                alignment_max_scale=alignment_max_scale,
            )
            point_frame_indices = np.concatenate(
                [
                    result.point_frame_indices
                    if result.point_frame_indices is not None
                    else np.full(result.points.shape[0], result.start, dtype=np.int32)
                    for result in window_results
                ],
                axis=0,
            )
            result = LiteVGGTBatchResult(
                frame_indices=np.asarray(frame_indices, dtype=np.int32),
                images=image_array,
                valid_masks=valid_mask_array,
                w2c=w2c_array,
                intrinsics=intrinsic_array,
                points=points,
                colors=colors,
                confidence=confidence,
                point_frame_indices=point_frame_indices,
                valid_pixel_count=sum(result.valid_pixel_count for result in window_results),
                point_count_before_filter=sum(result.point_count_before_filter for result in window_results),
                point_count_after_filter=sum(result.point_count_after_filter for result in window_results),
            )
            return result, {
                "litevggt_window_size_effective": int(selected_attempt),
                "litevggt_window_overlap": int(selected_overlap),
                "litevggt_window_count": int(len(windows)),
                "litevggt_oom_retry_count": int(oom_retry_count),
                "litevggt_window_frame_retry_count": int(window_frame_retry_count),
            }
        except LiteVGGTWindowFrameMismatch as exc:
            if attempt_index < len(attempts) - 1:
                window_frame_retry_count += 1
                reset_litevggt_aggregator_cache(model)
                torch.cuda.empty_cache()
                progress("litevggt_inference", 40, f"{exc}; retrying with window_size={attempts[attempt_index + 1]}")
                continue
            raise PreviewFailure("LITEVGGT_WINDOW_FRAME_MISMATCH", str(exc)) from exc
        except RuntimeError as exc:
            if is_cuda_oom_error(exc) and attempt_index < len(attempts) - 1:
                last_oom = exc
                oom_retry_count += 1
                reset_litevggt_aggregator_cache(model)
                torch.cuda.empty_cache()
                progress("litevggt_inference", 40, f"LiteVGGT CUDA OOM at window_size={selected_attempt}; retrying with window_size={attempts[attempt_index + 1]}")
                continue
            raise
    if last_oom is not None:
        raise last_oom
    raise PreviewFailure("LITEVGGT_EMPTY_WINDOWS", "LiteVGGT windowed inference produced no merged result")


def trim_axis_quantile_outliers_with_frames(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    point_frame_indices: np.ndarray,
    *,
    low_quantile: float,
    high_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    points = points[finite]
    colors = colors[finite]
    confidence = confidence[finite]
    point_frame_indices = point_frame_indices[finite]
    if points.shape[0] <= 10:
        return points, colors, confidence, point_frame_indices
    mins = np.quantile(points, low_quantile, axis=0)
    maxs = np.quantile(points, high_quantile, axis=0)
    keep = np.all((points >= mins) & (points <= maxs), axis=1)
    if not np.any(keep):
        return points, colors, confidence, point_frame_indices
    return points[keep], colors[keep], confidence[keep], point_frame_indices[keep]


def trim_spatial_outliers_with_frames(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    point_frame_indices: np.ndarray,
    *,
    keep_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep_quantile = float(np.clip(keep_quantile, 0.5, 1.0))
    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    points = points[finite]
    colors = colors[finite]
    confidence = confidence[finite]
    point_frame_indices = point_frame_indices[finite]
    if keep_quantile >= 1.0 or points.shape[0] <= 1:
        return points, colors, confidence, point_frame_indices
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    cutoff = float(np.quantile(distances, keep_quantile))
    keep = distances <= cutoff
    if not np.any(keep):
        return points, colors, confidence, point_frame_indices
    return points[keep], colors[keep], confidence[keep], point_frame_indices[keep]


def voxel_downsample_points_with_frames(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    point_frame_indices: np.ndarray,
    *,
    max_points: int,
    voxel_size: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors, confidence, point_frame_indices
    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    points = points[finite]
    colors = colors[finite]
    confidence = confidence[finite]
    point_frame_indices = point_frame_indices[finite]
    if points.shape[0] <= max_points:
        return points, colors, confidence, point_frame_indices
    if voxel_size is None:
        lo = np.quantile(points, 0.01, axis=0)
        hi = np.quantile(points, 0.99, axis=0)
        scene_scale = float(np.linalg.norm(hi - lo))
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
    return points[selected], colors[selected], confidence[selected], point_frame_indices[selected]


def run_litevggt_reconstruction(
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
    inference_mode: str = "auto",
    single_frame_limit: int = 192,
    global_keyframe_count: int = 192,
    hierarchical_enable: bool = False,
    chunk_size: int = 64,
    chunk_overlap: int = 16,
    anchor_count: int = 8,
    alignment_max_rel_median: float = 0.05,
    alignment_max_rel_p90: float = 0.12,
    alignment_min_scale: float = 0.25,
    alignment_max_scale: float = 4.0,
    window_size: int = 48,
    window_overlap: int = 16,
    oom_window_sizes: list[int] | None = None,
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

        original_files = image_files(input_dir)
        if len(original_files) < 8:
            raise PreviewFailure("LITEVGGT_NOT_ENOUGH_IMAGES", "LiteVGGT preview requires at least 8 images")
        original_frame_count = len(original_files)
        files = original_files

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
        elif frame_selection == "all":
            files = select_aligned_frames(
                files,
                multiple=8,
                max_frames=max_input_frames,
                mode="head",
            )
        else:
            files = select_aligned_frames(
                files,
                multiple=8,
                max_frames=max_input_frames,
                mode="head",
            )

        aligned_count = len(files)
        original_index_by_file = {file: index for index, file in enumerate(original_files)}
        frame_indices = [original_index_by_file[file] for file in files]

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

        image_tensors = []
        valid_mask_tensors = []
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
            image_tensors.append(image_tensor)
            valid_mask_tensors.append(mask_tensor)
            if index % 4 == 0:
                progress("litevggt_preprocess", 28 + int(index / max(1, len(files)) * 8), f"preprocessed {index + 1}/{len(files)} images")

        image_tensors_by_frame = {frame_index: image_tensors[position] for position, frame_index in enumerate(frame_indices)}
        valid_mask_tensors_by_frame = {frame_index: valid_mask_tensors[position] for position, frame_index in enumerate(frame_indices)}
        batch_kwargs = {
            "device": device,
            "dtype": dtype,
            "keep_ratio": keep_ratio,
            "edge_keep_ratio": edge_keep_ratio,
            "selection_strategy": selection_strategy,
            "te": te,
            "DelayedScaling": DelayedScaling,
            "Format": Format,
            "unproject_depth_map_to_point_map": unproject_depth_map_to_point_map,
            "pose_encoding_to_extri_intri": pose_encoding_to_extri_intri,
        }
        requested_mode = str(inference_mode or "auto").strip().lower()
        resolved_mode = resolve_litevggt_effective_mode(
            inference_mode=requested_mode,
            aligned_count=aligned_count,
            single_frame_limit=single_frame_limit,
            hierarchical_enable=hierarchical_enable,
        )
        mode_metrics: dict[str, int | float] = {
            "chunk_count": 0,
            "accepted_chunk_count": 0,
            "rejected_chunk_count": 0,
            "alignment_rel_median_max": 0.0,
            "alignment_rel_p90_max": 0.0,
            "alignment_scale_min": 1.0,
            "alignment_scale_max": 1.0,
            "litevggt_window_size_effective": 0,
            "litevggt_window_overlap": 0,
            "litevggt_window_count": 0,
            "litevggt_oom_retry_count": 0,
            "litevggt_window_frame_retry_count": 0,
        }

        if requested_mode == "auto" and resolved_mode == "single":
            try:
                batch_result = _run_litevggt_single_mode(
                    model=model,
                    frame_indices=frame_indices,
                    image_tensors_by_frame=image_tensors_by_frame,
                    valid_mask_tensors_by_frame=valid_mask_tensors_by_frame,
                    progress=progress,
                    **batch_kwargs,
                )
            except RuntimeError as exc:
                if not is_cuda_oom_error(exc):
                    raise
                torch.cuda.empty_cache()
                progress("litevggt_oom_fallback", 42, "LiteVGGT single OOM, falling back to global keyframes")
                resolved_mode = "global_keyframe"
                batch_result = _run_litevggt_global_keyframe_mode(
                    model=model,
                    files=files,
                    frame_indices=frame_indices,
                    global_keyframe_count=global_keyframe_count,
                    min_scene_change=min_scene_change,
                    image_tensors_by_frame=image_tensors_by_frame,
                    valid_mask_tensors_by_frame=valid_mask_tensors_by_frame,
                    progress=progress,
                    **batch_kwargs,
                )
        elif resolved_mode == "single":
            batch_result = _run_litevggt_single_mode(
                model=model,
                frame_indices=frame_indices,
                image_tensors_by_frame=image_tensors_by_frame,
                valid_mask_tensors_by_frame=valid_mask_tensors_by_frame,
                progress=progress,
                **batch_kwargs,
            )
        elif resolved_mode == "global_keyframe":
            batch_result = _run_litevggt_global_keyframe_mode(
                model=model,
                files=files,
                frame_indices=frame_indices,
                global_keyframe_count=global_keyframe_count,
                min_scene_change=min_scene_change,
                image_tensors_by_frame=image_tensors_by_frame,
                valid_mask_tensors_by_frame=valid_mask_tensors_by_frame,
                progress=progress,
                **batch_kwargs,
            )
        elif resolved_mode == "hierarchical":
            batch_result, hierarchical_metrics = _run_litevggt_hierarchical_mode(
                model=model,
                files=files,
                frame_indices=frame_indices,
                global_keyframe_count=global_keyframe_count,
                min_scene_change=min_scene_change,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                anchor_count=anchor_count,
                alignment_min_scale=alignment_min_scale,
                alignment_max_scale=alignment_max_scale,
                alignment_max_rel_median=alignment_max_rel_median,
                alignment_max_rel_p90=alignment_max_rel_p90,
                image_tensors_by_frame=image_tensors_by_frame,
                valid_mask_tensors_by_frame=valid_mask_tensors_by_frame,
                progress=progress,
                **batch_kwargs,
            )
            mode_metrics.update(hierarchical_metrics)
        elif resolved_mode == "windowed":
            batch_result, window_metrics = _run_litevggt_windowed_mode_with_quality_checks(
                model=model,
                image_tensors=image_tensors,
                valid_mask_tensors=valid_mask_tensors,
                frame_indices=frame_indices,
                aligned_count=aligned_count,
                window_size=window_size,
                window_overlap=window_overlap,
                oom_window_sizes=oom_window_sizes or [32, 16, 8],
                alignment_max_rel_median=alignment_max_rel_median,
                alignment_max_rel_p90=alignment_max_rel_p90,
                alignment_min_scale=alignment_min_scale,
                alignment_max_scale=alignment_max_scale,
                progress=progress,
                **batch_kwargs,
            )
            mode_metrics.update(window_metrics)
        else:
            raise PreviewFailure("LITEVGGT_INFERENCE_MODE_INVALID", f"Unsupported LiteVGGT inference mode: {resolved_mode}")

        selected_points = batch_result.points
        selected_colors = batch_result.colors
        selected_confidence = batch_result.confidence
        selected_point_frame_indices = batch_result.point_frame_indices

        if spatial_keep_quantile >= 1.0:
            trimmed_points, trimmed_colors, trimmed_confidence, trimmed_point_frame_indices = trim_axis_quantile_outliers_with_frames(
                selected_points,
                selected_colors,
                selected_confidence,
                selected_point_frame_indices,
                low_quantile=axis_trim_low_quantile,
                high_quantile=axis_trim_high_quantile,
            )
        else:
            trimmed_points, trimmed_colors, trimmed_confidence, trimmed_point_frame_indices = trim_spatial_outliers_with_frames(
                selected_points,
                selected_colors,
                selected_confidence,
                selected_point_frame_indices,
                keep_quantile=spatial_keep_quantile,
            )

        point_count_after_spatial_trim = int(trimmed_points.shape[0])
        point_count_before_downsample = int(trimmed_points.shape[0])

        trimmed_points, trimmed_colors, trimmed_confidence, trimmed_point_frame_indices = voxel_downsample_points_with_frames(
            trimmed_points,
            trimmed_colors,
            trimmed_confidence,
            trimmed_point_frame_indices,
            max_points=max_points,
        )

        point_count_after_downsample = int(trimmed_points.shape[0])
        valid_pixel_count = batch_result.valid_pixel_count
        point_count_before_filter = batch_result.point_count_before_filter
        point_count_after_filter = batch_result.point_count_after_filter

        peak_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0
        used_frame_count = int(aligned_count if resolved_mode == "hierarchical" else batch_result.frame_indices.shape[0])
        effective_global_keyframe_count = int(batch_result.frame_indices.shape[0] if resolved_mode in {"global_keyframe", "hierarchical"} else 0)
        return LiteVGGTReconstruction(
            files=files,
            frame_indices=batch_result.frame_indices,
            images=batch_result.images,
            valid_masks=batch_result.valid_masks,
            w2c=batch_result.w2c,
            intrinsics=batch_result.intrinsics,
            points=trimmed_points,
            colors=trimmed_colors,
            confidence=trimmed_confidence,
            point_frame_indices=trimmed_point_frame_indices,
            metrics={
                "original_frame_count": int(original_frame_count),
                "input_frame_count": used_frame_count,
                "aligned_frame_count": int(aligned_count),
                "global_keyframe_count": effective_global_keyframe_count,
                "skipped_frame_count": int(original_frame_count - used_frame_count),
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
                "litevggt_inference_mode": resolved_mode,
                "litevggt_inference_mode_requested": requested_mode,
                "litevggt_inference_mode_effective": resolved_mode,
                **mode_metrics,
                "valid_pixel_count": int(valid_pixel_count),
                "point_count_before_filter": int(point_count_before_filter),
                "point_count_after_filter": int(point_count_after_filter),
                "point_count_before_spatial_trim": int(selected_points.shape[0]),
                "point_count_after_spatial_trim": point_count_after_spatial_trim,
                "point_count_before_downsample": point_count_before_downsample,
                "point_count_after_downsample": point_count_after_downsample,
                "point_count_after_voxel_downsample": point_count_after_downsample,
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
    edge_keep_ratio: float = 0.0,
    axis_trim_low_quantile: float = 0.0005,
    axis_trim_high_quantile: float = 0.9995,
    selection_strategy: str = "global",
    inference_mode: str = "auto",
    single_frame_limit: int = 192,
    global_keyframe_count: int = 192,
    hierarchical_enable: bool = False,
    chunk_size: int = 64,
    chunk_overlap: int = 16,
    anchor_count: int = 8,
    alignment_max_rel_median: float = 0.05,
    alignment_max_rel_p90: float = 0.12,
    alignment_min_scale: float = 0.25,
    alignment_max_scale: float = 4.0,
    window_size: int = 48,
    window_overlap: int = 16,
    oom_window_sizes: list[int] | None = None,
    progress: Progress,
) -> dict[str, int | float | str | bool]:
    reconstruction = run_litevggt_reconstruction(
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
        inference_mode=inference_mode,
        single_frame_limit=single_frame_limit,
        global_keyframe_count=global_keyframe_count,
        hierarchical_enable=hierarchical_enable,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        anchor_count=anchor_count,
        alignment_max_rel_median=alignment_max_rel_median,
        alignment_max_rel_p90=alignment_max_rel_p90,
        alignment_min_scale=alignment_min_scale,
        alignment_max_scale=alignment_max_scale,
        window_size=window_size,
        window_overlap=window_overlap,
        oom_window_sizes=oom_window_sizes,
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
