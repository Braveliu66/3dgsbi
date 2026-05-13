from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

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
class LiteVGGTQualitySettings:
    target_size: int
    keep_ratio: float
    quality_profile: str
    keep_ratio_source: str
    target_size_source: str


def resolve_litevggt_quality_settings(frame_count: int, options: dict[str, Any] | None = None) -> LiteVGGTQualitySettings:
    options = options or {}
    frame_count = int(frame_count)

    if frame_count <= 16:
        profile = "small"
        target_size = 448
        keep_ratio = 0.90
    elif frame_count <= 48:
        profile = "medium"
        target_size = 392
        keep_ratio = 0.85
    elif frame_count <= 128:
        profile = "large"
        target_size = 336
        keep_ratio = 0.75
    elif frame_count <= 256:
        profile = "xlarge"
        target_size = 308
        keep_ratio = 0.60
    else:
        profile = "huge"
        target_size = 280
        keep_ratio = 0.42

    target_size_source = "auto"
    if options.get("target_size") is not None:
        target_size = _align_litevggt_target_size(options["target_size"])
        target_size_source = "user"

    keep_ratio_source = "auto"
    if options.get("keep_ratio") is not None:
        keep_ratio = float(np.clip(float(options["keep_ratio"]), 0.01, 1.0))
        keep_ratio_source = "user"

    return LiteVGGTQualitySettings(
        target_size=_align_litevggt_target_size(target_size),
        keep_ratio=keep_ratio,
        quality_profile=profile,
        keep_ratio_source=keep_ratio_source,
        target_size_source=target_size_source,
    )


def _align_litevggt_target_size(value: Any) -> int:
    target_size = int(round(float(value) / 14.0) * 14)
    return max(14, target_size)


def select_aligned_frames(
    files: list[Path],
    *,
    multiple: int = 8,
    max_frames: int | None = None,
) -> list[Path]:
    if multiple <= 0:
        raise PreviewFailure("LITEVGGT_INVALID_FRAME_MULTIPLE", "LiteVGGT frame multiple must be positive")

    usable = len(files)
    if max_frames is not None and max_frames > 0:
        usable = min(usable, int(max_frames))
    usable = (usable // multiple) * multiple
    if usable <= 0:
        return []
    return files[:usable]


def point_indices_to_frame_indices(
    selected_pixel_indices: np.ndarray,
    *,
    frame_indices: list[int] | np.ndarray,
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


def select_points_by_confidence(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    *,
    frame_indices: list[int] | np.ndarray,
    height: int,
    width: int,
    keep_ratio: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = np.clip(np.asarray(colors).reshape(-1, 3), 0, 255).astype(np.uint8)
    conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if pts.shape[0] != rgb.shape[0] or pts.shape[0] != conf.shape[0]:
        raise PreviewFailure("LITEVGGT_POINT_SHAPE_MISMATCH", "LiteVGGT points, colors and confidence must have equal length")

    valid = np.isfinite(pts).all(axis=1) & np.isfinite(conf)
    valid_indices = np.where(valid)[0]
    if valid_indices.size == 0:
        raise PreviewFailure("LITEVGGT_EMPTY_POINT_CLOUD", "LiteVGGT produced no valid points")

    keep_ratio = float(np.clip(keep_ratio, 0.01, 1.0))
    keep_count = max(1, int(valid_indices.size * keep_ratio))
    if max_points > 0:
        keep_count = min(keep_count, int(max_points))

    ranked = valid_indices[np.argsort(conf[valid_indices])[::-1]]
    selected = ranked[:keep_count]
    point_frame_indices = point_indices_to_frame_indices(
        selected,
        frame_indices=frame_indices,
        height=height,
        width=width,
    )
    return pts[selected], rgb[selected], conf[selected], point_frame_indices


def reset_litevggt_aggregator_cache(model) -> None:
    aggregator = getattr(model, "aggregator", None)
    if aggregator is not None and hasattr(aggregator, "m_u"):
        aggregator.m_u = None


def load_litevggt_image_tensors(files: list[Path], load_image_file_crop, target_size: int) -> list:
    import torch

    return [
        torch.from_numpy(np.transpose(load_image_file_crop(str(file), target_size=target_size), (2, 0, 1))).float()
        for file in files
    ]


def run_litevggt_reconstruction(
    *,
    input_dir: Path,
    checkpoint_path: Path,
    keep_ratio: float | None,
    max_points: int,
    progress: Progress,
    max_input_frames: int | None = None,
    target_size: int | None = None,
    spatial_keep_quantile: float = 1.0,
    preserve_full_image: bool = False,
    letterbox_size: int = 518,
    frame_selection: str = "all",
    min_scene_change: float = 0.0,
    edge_keep_ratio: float = 0.0,
    axis_trim_low_quantile: float = 0.0,
    axis_trim_high_quantile: float = 1.0,
    selection_strategy: str = "global",
    **_unused_options,
) -> LiteVGGTReconstruction:
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
        original_frame_count = len(original_files)
        if original_frame_count < 8:
            raise PreviewFailure("LITEVGGT_NOT_ENOUGH_IMAGES", "LiteVGGT preview requires at least 8 images")

        files = select_aligned_frames(original_files, multiple=8, max_frames=max_input_frames)
        if len(files) < 8:
            raise PreviewFailure("LITEVGGT_NOT_ENOUGH_FRAMES", f"LiteVGGT requires at least 8 images, got {len(files)}")

        quality = resolve_litevggt_quality_settings(
            len(files),
            {
                "target_size": target_size,
                "keep_ratio": keep_ratio,
            },
        )
        original_index_by_file = {file: index for index, file in enumerate(original_files)}
        frame_indices = [original_index_by_file[file] for file in files]

        if not torch.cuda.is_available():
            raise PreviewFailure("GPU_RESOURCE_UNAVAILABLE", "LiteVGGT requires CUDA")
        device = "cuda:0"
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        progress(
            "litevggt_preprocess",
            28,
            f"loading {len(files)} LiteVGGT images at target_size={quality.target_size}",
        )
        image_tensors = load_litevggt_image_tensors(files, load_image_file_crop, quality.target_size)
        image_batch = torch.stack(image_tensors, dim=0).to(device)
        height = int(image_batch.shape[-2])
        width = int(image_batch.shape[-1])

        progress("litevggt_loading_model", 34, f"loading LiteVGGT checkpoint: {checkpoint_path.name}")
        model = VGGT().to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        model.to(torch.bfloat16)
        model.eval()

        aggregated_tokens_list = None
        pose_enc = None
        w2c_pre = None
        intrinsic = None
        depth_map = None
        depth_conf = None
        points_3d = None

        try:
            patch_width = width // 14
            patch_height = height // 14
            model.update_patch_dimensions(patch_width, patch_height)
            image_batch = image_batch[None]

            progress("litevggt_inference", 42, f"running LiteVGGT on {len(files)} aligned images")
            with torch.no_grad():
                fp8_recipe = DelayedScaling(fp8_format=Format.E4M3, amax_history_len=80, amax_compute_algo="max")
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    aggregated_tokens_list, patch_start_idx = model.aggregator(image_batch)

                with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                    w2c_pre, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_batch.shape[-2:])
                    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, image_batch, patch_start_idx)

                points_3d = unproject_depth_map_to_point_map(
                    depth_map.squeeze(0),
                    w2c_pre.squeeze(0),
                    intrinsic.squeeze(0),
                )

            image_array = image_batch[0].permute(0, 2, 3, 1).detach().float().cpu().numpy()
            valid_masks = np.ones((len(files), height, width), dtype=bool)
            points = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)
            colors = np.clip(image_array.reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)
            confidence = depth_conf.reshape(-1).detach().float().cpu().numpy()

            selected_points, selected_colors, selected_confidence, point_frame_indices = select_points_by_confidence(
                points,
                colors,
                confidence,
                frame_indices=frame_indices,
                height=height,
                width=width,
                keep_ratio=quality.keep_ratio,
                max_points=max_points,
            )

            w2c_array = w2c_pre.squeeze(0).detach().float().cpu().numpy()
            intrinsic_array = intrinsic.squeeze(0).detach().float().cpu().numpy()
            selected_count = int(selected_points.shape[0])
            peak_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0
            return LiteVGGTReconstruction(
                files=files,
                frame_indices=np.asarray(frame_indices, dtype=np.int32),
                images=image_array,
                valid_masks=valid_masks,
                w2c=w2c_array,
                intrinsics=intrinsic_array,
                points=selected_points,
                colors=selected_colors,
                confidence=selected_confidence,
                point_frame_indices=point_frame_indices,
                metrics={
                    "original_frame_count": int(original_frame_count),
                    "input_frame_count": int(len(files)),
                    "aligned_frame_count": int(len(files)),
                    "skipped_frame_count": int(original_frame_count - len(files)),
                    "frame_selection": "all",
                    "point_selection_strategy": "global_confidence",
                    "keep_ratio": float(quality.keep_ratio),
                    "keep_ratio_source": quality.keep_ratio_source,
                    "max_points": int(max_points),
                    "litevggt_target_size": int(quality.target_size),
                    "litevggt_target_size_source": quality.target_size_source,
                    "litevggt_quality_profile": quality.quality_profile,
                    "litevggt_inference_mode": "single",
                    "litevggt_inference_mode_requested": "single",
                    "litevggt_inference_mode_effective": "single",
                    "valid_pixel_count": int(np.isfinite(points).all(axis=1).sum()),
                    "point_count_before_filter": int(points.shape[0]),
                    "point_count_after_filter": selected_count,
                    "point_count_before_downsample": selected_count,
                    "point_count_after_downsample": selected_count,
                    "point_count_after_voxel_downsample": selected_count,
                    "cuda_memory_peak_mb": float(peak_mb),
                    "official_single_path": True,
                },
            )
        finally:
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


def run_litevggt_pointcloud(
    *,
    input_dir: Path,
    checkpoint_path: Path,
    output_ply: Path,
    keep_ratio: float | None,
    max_points: int,
    max_input_frames: int | None = None,
    target_size: int | None = None,
    progress: Progress,
    **unused_options,
) -> dict[str, int | float | str | bool]:
    reconstruction = run_litevggt_reconstruction(
        input_dir=input_dir,
        checkpoint_path=checkpoint_path,
        keep_ratio=keep_ratio,
        max_points=max_points,
        max_input_frames=max_input_frames,
        target_size=target_size,
        progress=progress,
        **unused_options,
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
