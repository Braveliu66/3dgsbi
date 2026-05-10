from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.preview.types import PreviewFailure


def write_point_cloud_ply(
    points: Any,
    colors: Any,
    output_path: Path,
    *,
    confidence: Any | None = None,
    max_points: int = 15_000_000,
    default_alpha: int = 255,
) -> int:
    """Write a plain colored point-cloud PLY."""

    pts, rgb, _conf = prepare_point_data(points, colors, confidence=confidence, max_points=max_points)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
        ]
    )
    records = np.empty(pts.shape[0], dtype=dtype)
    records["x"] = pts[:, 0]
    records["y"] = pts[:, 1]
    records["z"] = pts[:, 2]
    records["red"] = rgb[:, 0]
    records["green"] = rgb[:, 1]
    records["blue"] = rgb[:, 2]
    records["alpha"] = np.uint8(default_alpha)

    with output_path.open("wb") as handle:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {pts.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "property uchar alpha\n"
            "end_header\n"
        )
        handle.write(header.encode("ascii"))
        handle.write(records.tobytes())
    return int(pts.shape[0])


def write_gaussian_splat_ply(
    points: Any,
    colors: Any,
    output_path: Path,
    *,
    confidence: Any | None = None,
    max_points: int = 15_000_000,
    scale: float = 0.002,
    opacity_logit: float = -2.0,
) -> int:
    """Write a SuperSplat / 3DGS-compatible Gaussian Splat PLY."""

    pts, rgb, _conf = prepare_point_data(points, colors, confidence=confidence, max_points=max_points)
    count = int(pts.shape[0])
    if scale <= 0:
        raise PreviewFailure("PLY_INVALID_SCALE", "Gaussian splat scale must be positive")

    sh_c0 = np.float32(0.28209479177387814)
    sh_dc = (rgb.astype(np.float32) / 255.0 - 0.5) / sh_c0
    normals = np.zeros((count, 3), dtype=np.float32)
    f_rest = np.zeros((count, 45), dtype=np.float32)
    opacity = np.full((count, 1), float(opacity_logit), dtype=np.float32)
    log_scale = np.full((count, 3), float(np.log(scale)), dtype=np.float32)
    rotation = np.zeros((count, 4), dtype=np.float32)
    rotation[:, 0] = 1.0

    data = np.concatenate(
        [
            pts.astype(np.float32, copy=False),
            normals,
            sh_dc,
            f_rest,
            opacity,
            log_scale,
            rotation,
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
    )
    header += "".join(f"property float f_rest_{index}\n" for index in range(45))
    header += (
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(data.tobytes())
    return count


def prepare_point_data(
    points: Any,
    colors: Any,
    *,
    confidence: Any | None,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(colors).reshape(-1, 3)
    if pts.shape[0] != rgb.shape[0]:
        raise PreviewFailure("PLY_SHAPE_MISMATCH", "points and colors must have the same length")

    conf = None
    valid = np.isfinite(pts).all(axis=1)
    if confidence is not None:
        candidate = np.asarray(confidence, dtype=np.float32).reshape(-1)
        if candidate.shape[0] == pts.shape[0]:
            conf = candidate
            valid &= np.isfinite(conf)
    pts = pts[valid]
    rgb = np.clip(rgb[valid], 0, 255).astype(np.uint8)
    if conf is not None:
        conf = conf[valid]

    if pts.shape[0] == 0:
        raise PreviewFailure("EMPTY_POINT_CLOUD", "algorithm produced no valid 3D points")

    if max_points > 0 and pts.shape[0] > max_points:
        rng = np.random.default_rng(20260505)
        keep = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[keep]
        rgb = rgb[keep]
        if conf is not None:
            conf = conf[keep]
    return pts, rgb, conf
