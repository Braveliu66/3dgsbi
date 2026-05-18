from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - worker image dependency
    np = None


def write_filtered_sparse_points_ply(reconstruction: Any, output_path: Path) -> int | None:
    if np is None:
        return None
    points = list(getattr(reconstruction, "points3D", {}).values())
    if not points:
        return None

    records = []
    for point in points:
        xyz = np.asarray(getattr(point, "xyz", []), dtype=np.float64)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            continue
        color = np.asarray(getattr(point, "color", [255, 255, 255]), dtype=np.float64)
        if color.shape != (3,):
            color = np.asarray([255, 255, 255], dtype=np.float64)
        records.append((xyz, np.clip(color, 0, 255).astype(np.uint8), float(getattr(point, "error", 0.0) or 0.0), _track_length(point)))
    if not records:
        return None

    xyzs = np.stack([item[0] for item in records], axis=0)
    errors = np.asarray([item[2] for item in records], dtype=np.float64)
    tracks = np.asarray([item[3] for item in records], dtype=np.int32)

    error_limit = max(4.0, float(np.quantile(errors, 0.90)))
    keep = (errors <= error_limit) & (tracks >= 3)
    if int(keep.sum()) < max(100, int(len(records) * 0.20)):
        keep = (errors <= max(8.0, float(np.quantile(errors, 0.95)))) & (tracks >= 2)

    if int(keep.sum()) >= max(100, int(len(records) * 0.20)):
        kept_xyz = xyzs[keep]
        low = np.quantile(kept_xyz, 0.01, axis=0)
        high = np.quantile(kept_xyz, 0.99, axis=0)
        pad = np.maximum((high - low) * 0.20, 1e-6)
        inlier_box = np.all((xyzs >= (low - pad)) & (xyzs <= (high + pad)), axis=1)
        keep = keep & inlier_box

    kept = [record for record, keep_item in zip(records, keep) if bool(keep_item)]
    if len(kept) < max(100, int(len(records) * 0.10)):
        kept = records

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(kept)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for xyz, color, _error, _track in kept:
            handle.write(f"{xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g} {int(color[0])} {int(color[1])} {int(color[2])}\n")
    return len(kept)


def _track_length(point: Any) -> int:
    track = getattr(point, "track", None)
    length = getattr(track, "length", None)
    if callable(length):
        try:
            return int(length())
        except Exception:
            return 0
    elements = getattr(track, "elements", None)
    if elements is not None:
        try:
            return len(elements)
        except TypeError:
            return 0
    return 0
