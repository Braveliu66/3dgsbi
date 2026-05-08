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
AMB3R_COMMIT = "7aae7fbb77a750651ffa236bb9c3212290c6fc78"
AMB3R_BACKEND = "amb3r_sfm_colmap_no_exif"
AMB3R_WEIGHT_RELATIVE_PATH = Path("amb3r") / "amb3r.pt"
AMB3R_WIDTH = 518
AMB3R_HEIGHT = 392


@dataclass(slots=True)
class ProcessedAmb3rImage:
    path: Path
    width: int
    height: int
    processed_width: int
    processed_height: int
    crop_left: int
    crop_top: int
    crop_width: int
    crop_height: int
    image: np.ndarray


def amb3r_weight_path(model_cache_dir: Path) -> Path:
    path = Path(model_cache_dir) / AMB3R_WEIGHT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_amb3r_weight(model_cache_dir: Path) -> Path:
    path = amb3r_weight_path(model_cache_dir)
    if not path.exists() or not path.is_file():
        raise FineFailure("AMB3R_WEIGHT_MISSING", f"AMB3R checkpoint not found: {path}")
    return path


def build_amb3r_colmap_scene(
    input_dir: Path,
    scene_dir: Path,
    *,
    checkpoint_path: Path,
    keep_ratio: float,
    max_points: int,
    progress: Progress,
) -> SceneBuildResult:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FineFailure("AMB3R_WEIGHT_MISSING", f"AMB3R checkpoint not found: {checkpoint_path}")
    import torch

    if not torch.cuda.is_available():
        raise FineFailure("GPU_RESOURCE_UNAVAILABLE", "AMB3R SfM initialization requires CUDA")

    files = image_files(input_dir)
    if len(files) < 3:
        raise FineFailure("AMB3R_NOT_ENOUGH_IMAGES", "AMB3R SfM initialization requires at least 3 images")

    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    images_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    processed = prepare_amb3r_images(files, images_dir)
    real_count = len(processed)

    progress("fine_amb3r_loading_model", 24, f"loading AMB3R checkpoint: {checkpoint_path.name}")
    from app.fine.amb3r_runtime.amb3r.model import AMB3R
    from app.fine.amb3r_runtime.sfm.pipeline import AMB3R_SfM

    device = "cuda:0"
    model = AMB3R(device=device)
    model.load_weights(str(checkpoint_path), strict=False)
    model.to(device)
    model.eval()

    image_tensor = torch.stack(
        [torch.from_numpy(np.transpose(item.image, (2, 0, 1))) for item in processed],
        dim=0,
    ).unsqueeze(0).to(device)
    pipeline = AMB3R_SfM(model)

    progress("fine_amb3r_sfm", 32, f"running AMB3R-SfM on {real_count} images")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            memory = pipeline.run(image_tensor)

    poses_c2w = to_numpy(memory.poses)
    intrinsics = to_numpy(memory.intrinsics) if hasattr(memory, "intrinsics") else default_intrinsics(real_count)
    points_all = to_numpy(memory.pts)
    confidence_all = to_numpy(memory.conf)
    unmapped_frames = {int(index) for index in getattr(memory, "unmapped_frames", set())}
    registered_indices = valid_registered_indices(poses_c2w, real_count, unmapped_frames)
    if len(registered_indices) < 3:
        raise FineFailure("AMB3R_RECONSTRUCTION_INCOMPLETE", f"AMB3R registered {len(registered_indices)}/{real_count} images")

    points = points_all[registered_indices].reshape(-1, 3)
    confidence = confidence_all[registered_indices].reshape(-1)
    colors = np.stack([((processed[index].image + 1.0) * 0.5) for index in registered_indices], axis=0).reshape(-1, 3)
    keep_indices = select_confident_points(points, confidence, keep_ratio=keep_ratio, max_points=max_points)
    sampled_points = points[keep_indices]
    sampled_colors = np.clip(colors[keep_indices] * 255.0, 0, 255).astype(np.uint8)

    write_gaussian_splatting_ply(sparse_dir / "points3D.ply", sampled_points, sampled_colors)
    write_colmap_model(
        sparse_dir,
        processed,
        registered_indices,
        poses_c2w,
        intrinsics,
        sampled_points,
        sampled_colors,
    )

    point_count = int(sampled_points.shape[0])
    elapsed = round(time.monotonic() - started, 3)
    return SceneBuildResult(
        scene_dir=scene_dir,
        backend=AMB3R_BACKEND,
        image_count=real_count,
        registered_images=len(registered_indices),
        point_count=point_count,
        metrics={
            "sfm_backend": AMB3R_BACKEND,
            "sfm_elapsed_seconds": elapsed,
            "sfm_registered_images": len(registered_indices),
            "sfm_sparse_points": point_count,
            "amb3r_registered_images": len(registered_indices),
            "amb3r_unmapped_images": len(unmapped_frames),
            "amb3r_sparse_points": point_count,
            "amb3r_resolution": f"{AMB3R_WIDTH}x{AMB3R_HEIGHT}",
            "amb3r_keep_ratio": keep_ratio,
            "amb3r_max_points": max_points,
            "amb3r_source_commit": AMB3R_COMMIT,
        },
    )


def prepare_amb3r_images(files: list[Path], images_dir: Path) -> list[ProcessedAmb3rImage]:
    processed = []
    target_aspect = AMB3R_WIDTH / float(AMB3R_HEIGHT)
    for index, path in enumerate(files):
        with Image.open(path) as original:
            full = ImageOps.exif_transpose(original).convert("RGB")
        width, height = full.size
        full.save(images_dir / f"{index:06d}.jpg", format="JPEG", quality=94)

        crop_width = width
        crop_height = height
        if width / float(height) > target_aspect:
            crop_width = int(round(height * target_aspect))
        else:
            crop_height = int(round(width / target_aspect))
        crop_left = max(0, (width - crop_width) // 2)
        crop_top = max(0, (height - crop_height) // 2)
        cropped = full.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
        resized = cropped.resize((AMB3R_WIDTH, AMB3R_HEIGHT), Image.Resampling.BICUBIC)
        image = np.asarray(resized, dtype=np.float32) / 255.0
        processed.append(
            ProcessedAmb3rImage(
                path=path,
                width=int(width),
                height=int(height),
                processed_width=AMB3R_WIDTH,
                processed_height=AMB3R_HEIGHT,
                crop_left=int(crop_left),
                crop_top=int(crop_top),
                crop_width=int(crop_width),
                crop_height=int(crop_height),
                image=image * 2.0 - 1.0,
            )
        )
    return processed


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value, dtype=np.float32)


def default_intrinsics(count: int) -> np.ndarray:
    intrinsics = np.tile(np.eye(3, dtype=np.float32), (count, 1, 1))
    focal = 0.9 * max(AMB3R_WIDTH, AMB3R_HEIGHT)
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = AMB3R_WIDTH / 2.0
    intrinsics[:, 1, 2] = AMB3R_HEIGHT / 2.0
    return intrinsics


def valid_registered_indices(poses_c2w: np.ndarray, image_count: int, unmapped_frames: set[int]) -> list[int]:
    registered = []
    for index in range(image_count):
        if index in unmapped_frames:
            continue
        pose = poses_c2w[index]
        if np.isfinite(pose).all() and float(np.abs(pose).sum()) > 1e-6:
            registered.append(index)
    return registered


def select_confident_points(points: np.ndarray, confidence: np.ndarray, *, keep_ratio: float, max_points: int) -> np.ndarray:
    valid = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        raise FineFailure("AMB3R_EMPTY_POINT_CLOUD", "AMB3R produced no valid 3D points")
    keep = max(1, int(valid_indices.size * max(0.01, min(1.0, keep_ratio))))
    ranked = valid_indices[np.argsort(confidence[valid_indices])[::-1][:keep]]
    if ranked.size > max_points:
        rng = np.random.default_rng(20260508)
        ranked = rng.choice(ranked, size=max_points, replace=False)
    return ranked


def write_colmap_model(
    sparse_dir: Path,
    images: list[ProcessedAmb3rImage],
    registered_indices: list[int],
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    with prepend_sys_path(GS_ROOT / "utils"):
        from read_write_model import Camera, Image as ColmapImage, Point3D, rotmat2qvec, write_model

    cameras = {}
    colmap_images = {}
    for image_index in registered_indices:
        item = images[image_index]
        camera_id = image_index + 1
        image_id = image_index + 1
        k = intrinsics[image_index] if valid_intrinsic(intrinsics[image_index]) else default_intrinsics(1)[0]
        params = map_intrinsic_to_full_image(item, k)
        cameras[camera_id] = Camera(id=camera_id, model="PINHOLE", width=item.width, height=item.height, params=params)
        w2c = np.linalg.inv(as_homogeneous_pose(poses_c2w[image_index]))[:3, :]
        colmap_images[image_id] = ColmapImage(
            id=image_id,
            qvec=rotmat2qvec(w2c[:3, :3]),
            tvec=w2c[:3, 3].astype(np.float64),
            camera_id=camera_id,
            name=f"{image_index:06d}.jpg",
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


def valid_intrinsic(k: np.ndarray) -> bool:
    return np.isfinite(k).all() and float(k[0, 0]) > 0 and float(k[1, 1]) > 0


def map_intrinsic_to_full_image(item: ProcessedAmb3rImage, k: np.ndarray) -> np.ndarray:
    scale_x = item.crop_width / float(item.processed_width)
    scale_y = item.crop_height / float(item.processed_height)
    return np.array(
        [
            float(k[0, 0]) * scale_x,
            float(k[1, 1]) * scale_y,
            float(k[0, 2]) * scale_x + item.crop_left,
            float(k[1, 2]) * scale_y + item.crop_top,
        ],
        dtype=np.float64,
    )


def as_homogeneous_pose(pose: np.ndarray) -> np.ndarray:
    if pose.shape == (4, 4):
        return np.asarray(pose, dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = np.asarray(pose[:3, :4], dtype=np.float64)
    return matrix


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
