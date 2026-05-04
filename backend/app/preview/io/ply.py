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
    """写 Spark 可转码的点云 PLY。

    Spark 的 PlyReader 对纯点云 PLY 会自动补默认 Gaussian scale/rotation，
    所以 LiteVGGT/LingBot 的稠密点云可以先落成 PLY，再统一转 SPZ。
    """

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(colors).reshape(-1, 3)
    if pts.shape[0] != rgb.shape[0]:
        raise PreviewFailure("PLY_SHAPE_MISMATCH", "points and colors must have the same length")

    valid = np.isfinite(pts).all(axis=1)
    if confidence is not None:
        conf = np.asarray(confidence).reshape(-1)
        if conf.shape[0] == pts.shape[0]:
            valid &= np.isfinite(conf)
    pts = pts[valid]
    rgb = np.clip(rgb[valid], 0, 255).astype(np.uint8)

    if pts.shape[0] == 0:
        raise PreviewFailure("EMPTY_POINT_CLOUD", "algorithm produced no valid 3D points")

    if pts.shape[0] > max_points:
        # 固定随机种子保证同一输入的预览点采样稳定，方便缓存和排错。
        rng = np.random.default_rng(20260505)
        keep = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[keep]
        rgb = rgb[keep]

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
