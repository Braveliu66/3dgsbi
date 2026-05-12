from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.preview.types import PreviewFailure

SH_C0 = np.float32(0.28209479177387814)
POINT_CLOUD_PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("confidence", "<f4"),
    ]
)
FIXED_SPLAT_PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("f_dc_0", "<f4"),
        ("f_dc_1", "<f4"),
        ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"),
        ("scale_1", "<f4"),
        ("scale_2", "<f4"),
        ("rot_0", "<f4"),
        ("rot_1", "<f4"),
        ("rot_2", "<f4"),
        ("rot_3", "<f4"),
    ]
)


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

    pts, rgb, conf = prepare_point_data(points, colors, confidence=confidence, max_points=max_points)
    if conf is None:
        conf = np.ones(pts.shape[0], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = np.empty(pts.shape[0], dtype=POINT_CLOUD_PLY_DTYPE)
    records["x"] = pts[:, 0]
    records["y"] = pts[:, 1]
    records["z"] = pts[:, 2]
    records["red"] = rgb[:, 0]
    records["green"] = rgb[:, 1]
    records["blue"] = rgb[:, 2]
    records["confidence"] = conf

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
            "property float confidence\n"
            "end_header\n"
        )
        handle.write(header.encode("ascii"))
        handle.write(records.tobytes())
    return int(pts.shape[0])


def convert_pointcloud_ply_to_fixed_splat_ply(
    input_points_ply: Path,
    output_splats_ply: Path,
    *,
    point_radius: float,
    opacity: float = 0.75,
) -> int:
    count, records = read_fixed_pointcloud_ply(input_points_ply)
    points = np.column_stack((records["x"], records["y"], records["z"])).astype(np.float32, copy=False)
    colors = np.column_stack((records["red"], records["green"], records["blue"])).astype(np.uint8, copy=False)

    valid = np.isfinite(points).all(axis=1) & np.isfinite(records["confidence"])
    points = points[valid]
    colors = colors[valid]
    count = int(points.shape[0])
    if count <= 0:
        raise PreviewFailure("EMPTY_POINT_CLOUD", "point-cloud PLY contains no valid 3D points")

    splats = np.zeros(count, dtype=FIXED_SPLAT_PLY_DTYPE)
    splats["x"] = points[:, 0]
    splats["y"] = points[:, 1]
    splats["z"] = points[:, 2]
    sh = rgb_to_sh(colors)
    splats["f_dc_0"] = sh[:, 0]
    splats["f_dc_1"] = sh[:, 1]
    splats["f_dc_2"] = sh[:, 2]
    log_scale = np.float32(np.log(max(float(point_radius), 1e-8)))
    splats["opacity"] = np.float32(logit(opacity))
    splats["scale_0"] = log_scale
    splats["scale_1"] = log_scale
    splats["scale_2"] = log_scale
    splats["rot_0"] = np.float32(1.0)

    output_splats_ply.parent.mkdir(parents=True, exist_ok=True)
    with output_splats_ply.open("wb") as handle:
        handle.write(fixed_splat_ply_header(count).encode("ascii"))
        splats.tofile(handle)
    return count


def read_fixed_pointcloud_ply(input_points_ply: Path) -> tuple[int, np.ndarray]:
    if not input_points_ply.exists() or input_points_ply.stat().st_size <= 0:
        raise PreviewFailure("PLY_NOT_FOUND", f"non-empty point-cloud PLY not found: {input_points_ply}")
    payload = input_points_ply.read_bytes()
    header_end = payload.find(b"end_header\n")
    if header_end < 0:
        raise PreviewFailure("PLY_INVALID", f"PLY header terminator missing: {input_points_ply}")
    header = payload[: header_end + len(b"end_header\n")].decode("ascii", errors="strict")
    lines = header.splitlines()
    if len(lines) < 11 or lines[0] != "ply" or lines[1] != "format binary_little_endian 1.0":
        raise PreviewFailure("PLY_INVALID", f"unsupported point-cloud PLY header: {input_points_ply}")
    count = parse_vertex_count(lines, input_points_ply)
    expected = [
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float confidence",
    ]
    if lines[3:10] != expected:
        raise PreviewFailure(
            "PLY_INVALID",
            f"point-cloud PLY schema does not match LingBot preview schema: {input_points_ply}",
        )
    body = payload[header_end + len(b"end_header\n") :]
    required_size = count * POINT_CLOUD_PLY_DTYPE.itemsize
    if len(body) < required_size:
        raise PreviewFailure("PLY_INVALID", f"point-cloud PLY body is truncated: {input_points_ply}")
    return count, np.frombuffer(body[:required_size], dtype=POINT_CLOUD_PLY_DTYPE, count=count)


def parse_vertex_count(lines: list[str], path: Path) -> int:
    if len(lines) < 3 or not lines[2].startswith("element vertex "):
        raise PreviewFailure("PLY_INVALID", f"PLY vertex count missing: {path}")
    try:
        count = int(lines[2].split()[-1])
    except ValueError as exc:
        raise PreviewFailure("PLY_INVALID", f"PLY vertex count is invalid: {path}") from exc
    if count <= 0:
        raise PreviewFailure("EMPTY_POINT_CLOUD", "point-cloud PLY contains no vertices")
    return count


def fixed_splat_ply_header(count: int) -> str:
    return (
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


def rgb_to_sh(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32) / 255.0
    return (rgb - 0.5) / SH_C0


def logit(x: float) -> float:
    x = float(np.clip(x, 1e-4, 1.0 - 1e-4))
    return float(np.log(x / (1.0 - x)))


def fixed_preview_radius(points: np.ndarray) -> float:
    bbox_min = np.nanmin(points, axis=0)
    bbox_max = np.nanmax(points, axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if not np.isfinite(diag) or diag <= 0:
        return 0.002
    return float(np.clip(diag / 350.0, 0.0008, 0.006))


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
