from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement

from app.fine.option_utils import read_float, read_int
from app.fine.types import FineFailure


@dataclass(slots=True)
class SparseCompensationResult:
    metrics: dict[str, Any]


def compensate_sparse_point_cloud(scene_dir: Path, options: dict[str, Any]) -> SparseCompensationResult:
    enabled = str(options.get("fine_sparse_compensation_enabled", "true")).lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return SparseCompensationResult({"sparse_compensation_enabled": False})

    sparse_dir = scene_dir / "sparse" / "0"
    ply_path = sparse_dir / "points3D.ply"
    if not ply_path.exists():
        return SparseCompensationResult({"sparse_compensation_enabled": True, "sparse_compensation_points": 0, "sparse_compensation_reason": "points3D.ply_missing"})

    target_ratio = read_float(options.get("fine_sparse_compensation_ratio"), 0.20, minimum=0.0, maximum=1.0)
    max_added = read_int(options.get("fine_sparse_compensation_max_points"), 200_000, minimum=0, maximum=2_000_000)
    if target_ratio <= 0.0 or max_added <= 0:
        return SparseCompensationResult({"sparse_compensation_enabled": True, "sparse_compensation_points": 0})

    ply = PlyData.read(ply_path)
    vertices = ply["vertex"]
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
    rgb = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.uint8)
    if xyz.shape[0] < 8:
        raise FineFailure("SPARSE_COMPENSATION_EMPTY", "Sparse compensation needs at least 8 initial points")

    center = np.median(xyz, axis=0)
    radius = np.linalg.norm(xyz - center, axis=1)
    cutoff = float(np.quantile(radius, 0.995))
    keep = radius <= cutoff
    removed = int((~keep).sum())
    xyz = xyz[keep]
    rgb = rgb[keep]

    add_count = min(max_added, int(xyz.shape[0] * target_ratio))
    if add_count <= 0:
        _write_points(ply_path, xyz, rgb)
        return SparseCompensationResult(
            {
                "sparse_compensation_enabled": True,
                "sparse_compensation_points": 0,
                "sparse_compensation_removed_outliers": removed,
            }
        )

    rng = np.random.default_rng(20260508)
    sample_idx = rng.integers(0, xyz.shape[0], size=add_count)
    neighbor_idx = np.clip(sample_idx + rng.integers(-4, 5, size=add_count), 0, xyz.shape[0] - 1)
    base = xyz[sample_idx]
    neighbor = xyz[neighbor_idx]
    local_vec = neighbor - base
    scene_diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    fallback_sigma = max(scene_diag * 0.0005, 1e-5)
    noise = rng.normal(0.0, fallback_sigma, size=(add_count, 3)).astype(np.float32)
    alpha = rng.uniform(0.15, 0.85, size=(add_count, 1)).astype(np.float32)
    new_xyz = base + alpha * local_vec + noise
    new_rgb = rgb[sample_idx]

    merged_xyz = np.concatenate([xyz, new_xyz.astype(np.float32)], axis=0)
    merged_rgb = np.concatenate([rgb, new_rgb], axis=0)
    _write_points(ply_path, merged_xyz, merged_rgb)
    return SparseCompensationResult(
        {
            "sparse_compensation_enabled": True,
            "sparse_compensation_points": int(add_count),
            "sparse_compensation_removed_outliers": removed,
            "sparse_compensation_final_points": int(merged_xyz.shape[0]),
        }
    )


def _write_points(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
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

