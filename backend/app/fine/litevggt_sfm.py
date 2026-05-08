from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

from app.fine.preprocess import SceneBuildResult
from app.fine.types import FineFailure
from app.preview.utils import VENDOR_ROOT, image_files, prepend_sys_path


Progress = Callable[[str, int, str], None]
GS_ROOT = VENDOR_ROOT / "edgs" / "gaussian_splatting"


@dataclass(slots=True)
class ProcessedLiteImage:
    path: Path
    width: int
    height: int
    processed_width: int
    processed_height: int
    image: np.ndarray


def build_litevggt_colmap_scene(
    input_dir: Path,
    scene_dir: Path,
    *,
    checkpoint_path: Path,
    keep_ratio: float,
    max_points: int,
    progress: Progress,
) -> SceneBuildResult:
    import torch

    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FineFailure("LITEVGGT_WEIGHT_MISSING", f"LiteVGGT checkpoint not found: {checkpoint_path}")
    if not torch.cuda.is_available():
        raise FineFailure("GPU_RESOURCE_UNAVAILABLE", "LiteVGGT SfM initialization requires CUDA")

    files = image_files(input_dir)
    if len(files) < 3:
        raise FineFailure("LITEVGGT_NOT_ENOUGH_IMAGES", "LiteVGGT SfM initialization requires at least 3 images")

    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    images_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    require_transformer_engine()
    with prepend_sys_path(VENDOR_ROOT / "litevggt"):
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format
        from vggt.models.vggt import VGGT
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_image_file_crop
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        real_count = len(files)
        aligned_files = align_to_litevggt_batch(files)
        padding_count = len(aligned_files) - real_count
        processed = [load_litevggt_image(path, load_image_file_crop) for path in aligned_files]
        shapes = {(item.processed_height, item.processed_width) for item in processed}
        if len(shapes) != 1:
            raise FineFailure("LITEVGGT_INPUT_SHAPE_MISMATCH", "LiteVGGT fine SfM requires images with the same aspect ratio")

        copy_training_images(processed[:real_count], images_dir)
        progress("fine_litevggt_loading_model", 24, f"loading LiteVGGT checkpoint: {checkpoint_path.name}")
        device = "cuda:0"
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        model = VGGT().to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        model.to(torch.bfloat16)
        model.eval()

        image_tensor = torch.stack([torch.from_numpy(np.transpose(item.image, (2, 0, 1))) for item in processed], dim=0).to(device)
        patch_width = image_tensor.shape[-1] // 14
        patch_height = image_tensor.shape[-2] // 14
        model.update_patch_dimensions(patch_width, patch_height)
        image_batch = image_tensor[None]

        progress("fine_litevggt_sfm", 32, f"running LiteVGGT SfM on {real_count} images")
        with torch.no_grad():
            fp8_recipe = DelayedScaling(fp8_format=Format.E4M3, amax_history_len=80, amax_compute_algo="max")
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                aggregated_tokens_list, patch_start_idx = model.aggregator(image_batch)
            with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                w2c_pre, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_batch.shape[-2:])
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, image_batch, patch_start_idx)

        w2c_np = w2c_pre.squeeze(0)[:real_count].detach().cpu().numpy()
        intrinsic_np = intrinsic.squeeze(0)[:real_count].detach().cpu().numpy()
        points_3d = unproject_depth_map_to_point_map(
            depth_map.squeeze(0)[:real_count],
            w2c_pre.squeeze(0)[:real_count],
            intrinsic.squeeze(0)[:real_count],
        )
        colors = image_tensor[:real_count].permute(0, 2, 3, 1).reshape(-1, 3).detach().cpu().numpy()
        points = points_3d.reshape(-1, 3)
        confidence = depth_conf.squeeze(0)[:real_count].reshape(-1).detach().cpu().numpy()

    keep_indices = select_confident_points(points, confidence, keep_ratio=keep_ratio, max_points=max_points)
    sampled_points = points[keep_indices]
    sampled_colors = np.clip(colors[keep_indices] * 255.0, 0, 255).astype(np.uint8)
    write_gaussian_splatting_ply(sparse_dir / "points3D.ply", sampled_points, sampled_colors)
    write_colmap_model(sparse_dir, processed[:real_count], w2c_np, intrinsic_np, sampled_points, sampled_colors)

    point_count = int(sampled_points.shape[0])
    elapsed = round(time.monotonic() - started, 3)
    return SceneBuildResult(
        scene_dir=scene_dir,
        backend="litevggt_colmap_no_exif",
        image_count=real_count,
        registered_images=real_count,
        point_count=point_count,
        metrics={
            "sfm_backend": "litevggt_colmap_no_exif",
            "sfm_elapsed_seconds": elapsed,
            "sfm_registered_images": real_count,
            "sfm_sparse_points": point_count,
            "litevggt_real_images": real_count,
            "litevggt_padding_images": padding_count,
            "litevggt_aligned_batch_images": len(aligned_files),
            "litevggt_keep_ratio": keep_ratio,
        },
    )


def require_transformer_engine() -> None:
    try:
        __import__("transformer_engine")
    except Exception as exc:
        raise FineFailure("TRANSFORMER_ENGINE_UNAVAILABLE", f"LiteVGGT requires transformer_engine: {exc}") from exc


def align_to_litevggt_batch(files: list[Path]) -> list[Path]:
    remainder = len(files) % 8
    if remainder == 0:
        return files
    return [*files, *([files[-1]] * (8 - remainder))]


def load_litevggt_image(path: Path, loader) -> ProcessedLiteImage:
    image = loader(str(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = np.asarray(image[..., :3], dtype=np.float32)
    with Image.open(path) as original:
        width, height = ImageOps.exif_transpose(original).size
    return ProcessedLiteImage(
        path=path,
        width=int(width),
        height=int(height),
        processed_width=int(image.shape[1]),
        processed_height=int(image.shape[0]),
        image=image,
    )


def copy_training_images(images: list[ProcessedLiteImage], output_dir: Path) -> None:
    for index, item in enumerate(images):
        with Image.open(item.path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.save(output_dir / f"{index:06d}.jpg", format="JPEG", quality=94)


def select_confident_points(points: np.ndarray, confidence: np.ndarray, *, keep_ratio: float, max_points: int) -> np.ndarray:
    valid = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        raise FineFailure("LITEVGGT_EMPTY_POINT_CLOUD", "LiteVGGT produced no valid 3D points")
    keep = max(1, int(valid_indices.size * max(0.01, min(1.0, keep_ratio))))
    ranked = valid_indices[np.argsort(confidence[valid_indices])[::-1][:keep]]
    if ranked.size > max_points:
        rng = np.random.default_rng(20260506)
        ranked = rng.choice(ranked, size=max_points, replace=False)
    return ranked


def write_colmap_model(
    sparse_dir: Path,
    images: list[ProcessedLiteImage],
    w2c: np.ndarray,
    intrinsic: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    with prepend_sys_path(GS_ROOT / "utils"):
        from read_write_model import Camera, Image as ColmapImage, Point3D, rotmat2qvec, write_model

    cameras = {}
    colmap_images = {}
    for index, item in enumerate(images, start=1):
        k = intrinsic[index - 1].copy()
        scale_x = item.width / float(item.processed_width)
        scale_y = item.height / float(item.processed_height)
        params = np.array([k[0, 0] * scale_x, k[1, 1] * scale_y, k[0, 2] * scale_x, k[1, 2] * scale_y], dtype=np.float64)
        cameras[index] = Camera(id=index, model="PINHOLE", width=item.width, height=item.height, params=params)
        colmap_images[index] = ColmapImage(
            id=index,
            qvec=rotmat2qvec(w2c[index - 1, :3, :3]),
            tvec=w2c[index - 1, :3, 3].astype(np.float64),
            camera_id=index,
            name=f"{index - 1:06d}.jpg",
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )

    points3d = {}
    for index, (xyz, rgb) in enumerate(zip(points, colors), start=1):
        points3d[index] = Point3D(
            id=index,
            xyz=np.asarray(xyz, dtype=np.float64),
            rgb=np.asarray(rgb, dtype=np.uint8),
            error=0.0,
            image_ids=np.empty((0,), dtype=np.int32),
            point2D_idxs=np.empty((0,), dtype=np.int32),
        )
    write_model(cameras, colmap_images, points3d, str(sparse_dir), ext=".bin")


def write_gaussian_splatting_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    normals = np.zeros_like(xyz, dtype=np.float32)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    records = np.empty(xyz.shape[0], dtype=dtype)
    records["x"], records["y"], records["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    records["nx"], records["ny"], records["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    records["red"], records["green"], records["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(records, "vertex")]).write(path)
