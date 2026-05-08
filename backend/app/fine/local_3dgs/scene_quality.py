from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.preview.utils import VENDOR_ROOT, image_files, prepend_sys_path


GS_ROOT = VENDOR_ROOT / "edgs" / "gaussian_splatting"


@dataclass(slots=True)
class SceneQuality:
    pass_quality: bool
    reasons: list[str]
    metrics: dict[str, Any]


def assess_sfm_scene_quality(scene_dir: Path, *, prefix: str, min_points: int = 50_000) -> SceneQuality:
    sparse_dir = scene_dir / "sparse" / "0"
    images_dir = scene_dir / "images"
    reasons: list[str] = []
    metrics: dict[str, Any] = {
        f"{prefix}_quality_pass": False,
        f"{prefix}_quality_reasons": [],
    }

    try:
        with prepend_sys_path(GS_ROOT):
            from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary

            extrinsics = read_extrinsics_binary(str(sparse_dir / "images.bin"))
            intrinsics = read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
            points, _, _ = read_points3D_binary(str(sparse_dir / "points3D.bin"))
    except Exception as exc:
        reasons.append(f"quality_read_failed:{exc}")
        metrics[f"{prefix}_quality_reasons"] = reasons
        return SceneQuality(False, reasons, metrics)

    image_count = len(image_files(images_dir)) if images_dir.exists() else 0
    names = [item.name for item in extrinsics.values()]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    centers = []
    focal_values = []
    for item in extrinsics.values():
        rot = qvec2rotmat(item.qvec)
        center = -rot.T @ np.asarray(item.tvec, dtype=np.float64)
        centers.append(center)
        camera = intrinsics[item.camera_id]
        if len(camera.params) >= 2 and camera.model == "PINHOLE":
            focal_values.extend([float(camera.params[0]), float(camera.params[1])])
        elif len(camera.params) >= 1:
            focal_values.append(float(camera.params[0]))

    centers_np = np.asarray(centers, dtype=np.float64) if centers else np.zeros((0, 3), dtype=np.float64)
    span = float(np.linalg.norm(centers_np.max(axis=0) - centers_np.min(axis=0))) if len(centers_np) else 0.0
    point_count = int(points.shape[0]) if hasattr(points, "shape") else 0
    focal_min = min(focal_values) if focal_values else 0.0
    focal_max = max(focal_values) if focal_values else 0.0

    if len(extrinsics) < max(3, min(image_count, 8)):
        reasons.append("too_few_registered_cameras")
    if duplicate_names:
        reasons.append("duplicate_camera_names")
    if point_count < min_points:
        reasons.append("too_few_sparse_points")
    if span <= 1e-4:
        reasons.append("camera_centers_collapsed")
    if focal_min <= 0 or (focal_max > 0 and focal_max / max(focal_min, 1e-6) > 4.0):
        reasons.append("abnormal_focal_range")

    metrics.update(
        {
            f"{prefix}_quality_pass": not reasons,
            f"{prefix}_quality_reasons": reasons,
            f"{prefix}_quality_camera_count": len(extrinsics),
            f"{prefix}_quality_image_count": image_count,
            f"{prefix}_quality_duplicate_camera_names": len(duplicate_names),
            f"{prefix}_quality_camera_span": round(span, 6),
            f"{prefix}_quality_focal_min": round(focal_min, 6),
            f"{prefix}_quality_focal_max": round(focal_max, 6),
            f"{prefix}_quality_sparse_points": point_count,
        }
    )
    return SceneQuality(not reasons, reasons, metrics)


def assess_litevggt_scene_quality(scene_dir: Path, *, min_points: int = 50_000) -> SceneQuality:
    return assess_sfm_scene_quality(scene_dir, prefix="litevggt", min_points=min_points)
