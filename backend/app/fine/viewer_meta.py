from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

PLY_TYPE_MAP: dict[str, str] = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def write_final_viewer_meta_json(
    output_path: Path,
    *,
    final_ply: Path,
    scene_dir: Path,
    preferred_image_names: list[str] | None = None,
    asset_type: str = "fine_colmap_sparse_pointcloud",
    point_source: str = "colmap_sparse_points",
    default_view_mode: str = "points",
) -> dict[str, Any]:
    bounds = read_ply_xyz_bounds(final_ply)
    recommended_view = recommended_training_camera_view(scene_dir, bounds, preferred_image_names=preferred_image_names)
    payload: dict[str, Any] = {
        "asset_type": asset_type,
        "point_source": point_source,
        "num_points": bounds["vertex_count"],
        "point_count_exported": bounds["vertex_count"],
        "bbox_min": bounds["bbox_min"],
        "bbox_max": bounds["bbox_max"],
        "bbox_center": bounds["center"],
        "bbox_radius": bounds["radius"],
        "center": bounds["center"],
        "radius": bounds["radius"],
        "scale_applied": 1.0,
        "coordinate_system": "colmap_world",
        "recommended_frontend": {
            "default_view_mode": default_view_mode,
        },
    }
    if recommended_view is not None:
        payload["recommended_view"] = recommended_view
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def write_scaled_viewer_ply(
    input_ply: Path,
    output_ply: Path,
    *,
    scale_multiplier: float = 0.65,
    max_scale: float | None = None,
) -> dict[str, Any]:
    """Write a viewer-only PLY with smaller Gaussian scales.

    3DGS PLY scale fields are log-space. Multiplying physical scale by k means
    adding log(k) to each scale component.
    """

    scale_multiplier = max(float(scale_multiplier), 1e-6)
    vertex_count, vertex_dtype, data_offset = read_binary_little_endian_ply_layout(input_ply)
    names = set(vertex_dtype.names or ())
    scale_names = [name for name in ("scale_0", "scale_1", "scale_2") if name in names]
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    if not scale_names:
        shutil.copy2(input_ply, output_ply)
        return {
            "viewer_scale_multiplier": scale_multiplier,
            "viewer_scale_max": max_scale,
            "viewer_scale_clamped": 0,
            "viewer_scale_vertices": vertex_count,
            "viewer_scale_fields": [],
        }

    payload = input_ply.read_bytes()
    required_size = vertex_count * vertex_dtype.itemsize
    body_end = data_offset + required_size
    if len(payload) < body_end:
        raise ValueError(f"PLY body is truncated: {input_ply}")

    records = np.frombuffer(payload[data_offset:body_end], dtype=vertex_dtype, count=vertex_count).copy()
    delta = np.float32(math.log(scale_multiplier))
    clamp_log = np.float32(math.log(max_scale)) if max_scale and max_scale > 0 else None
    clamped = 0
    for name in scale_names:
        records[name] = records[name].astype(np.float32, copy=False) + delta
        if clamp_log is not None:
            over = records[name] > clamp_log
            clamped += int(np.count_nonzero(over))
            records[name] = np.minimum(records[name], clamp_log)

    with output_ply.open("wb") as handle:
        handle.write(payload[:data_offset])
        handle.write(records.tobytes())
        handle.write(payload[body_end:])

    return {
        "viewer_scale_multiplier": scale_multiplier,
        "viewer_scale_max": max_scale,
        "viewer_scale_clamped": clamped,
        "viewer_scale_vertices": vertex_count,
        "viewer_scale_fields": scale_names,
    }


def write_far_noise_filtered_ply(input_ply: Path, output_ply: Path, *, profile: str) -> dict[str, Any]:
    profile = profile if profile in FAR_NOISE_PROFILE_FACTORS else "mixed_balanced"
    vertex_count, vertex_dtype, data_offset = read_binary_little_endian_ply_layout(input_ply)
    payload = input_ply.read_bytes()
    required_size = vertex_count * vertex_dtype.itemsize
    body_end = data_offset + required_size
    if len(payload) < body_end:
        raise ValueError(f"PLY body is truncated: {input_ply}")

    records = np.frombuffer(payload[data_offset:body_end], dtype=vertex_dtype, count=vertex_count).copy()
    points = np.column_stack((records["x"], records["y"], records["z"])).astype(np.float32, copy=False)
    finite = np.isfinite(points).all(axis=1)
    finite_points = points[finite]
    if finite_points.shape[0] <= 0:
        raise ValueError(f"PLY contains no finite xyz vertices: {input_ply}")

    bbox_min = np.percentile(finite_points, 1, axis=0).astype(np.float32)
    bbox_max = np.percentile(finite_points, 99, axis=0).astype(np.float32)
    center = ((bbox_min + bbox_max) * 0.5).astype(np.float32)
    distances = np.linalg.norm(points - center, axis=1)
    finite_distances = distances[finite & np.isfinite(distances)]
    robust_radius = max(float(np.percentile(finite_distances, 98)), 0.05)
    threshold = robust_radius * FAR_NOISE_PROFILE_FACTORS[profile]

    keep = finite & np.isfinite(distances) & (distances <= threshold)
    if "opacity" in (vertex_dtype.names or ()):
        opacities = _ply_opacity_to_probability(records["opacity"].astype(np.float32, copy=False))
        low_opacity = opacities < FAR_NOISE_LOW_OPACITY
        keep &= ~((distances > robust_radius) & low_opacity)
    if int(np.count_nonzero(keep)) <= 0:
        keep = finite & np.isfinite(distances)

    removed = int(vertex_count - np.count_nonzero(keep))
    if removed <= 0:
        shutil.copy2(input_ply, output_ply)
    else:
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        filtered_records = records[keep]
        with output_ply.open("wb") as handle:
            handle.write(_rewrite_vertex_count(payload[:data_offset], int(filtered_records.shape[0])))
            handle.write(filtered_records.tobytes())
            handle.write(payload[body_end:])

    return {
        "far_noise_filter_enabled": True,
        "far_noise_profile": profile,
        "far_noise_removed_points": removed,
        "far_noise_kept_points": int(vertex_count - removed),
        "far_noise_input_points": int(vertex_count),
        "far_noise_distance_threshold": threshold,
    }


def read_ply_xyz_bounds(path: Path) -> dict[str, Any]:
    vertex_count, vertex_dtype, data_offset = read_binary_little_endian_ply_layout(path)
    records = np.memmap(path, dtype=vertex_dtype, mode="r", offset=data_offset, shape=(vertex_count,))
    try:
        points = np.column_stack((records["x"], records["y"], records["z"])).astype(np.float32, copy=False)
        finite = points[np.isfinite(points).all(axis=1)]
        if finite.shape[0] <= 0:
            raise ValueError(f"PLY contains no finite xyz vertices: {path}")
        bbox_min = np.percentile(finite, 1, axis=0).astype(np.float32)
        bbox_max = np.percentile(finite, 99, axis=0).astype(np.float32)
        center = ((bbox_min + bbox_max) * 0.5).astype(np.float32)
        radius = max(float(np.linalg.norm(bbox_max - bbox_min) * 0.5), 0.05)
        return {
            "vertex_count": int(vertex_count),
            "bbox_min": [float(value) for value in bbox_min],
            "bbox_max": [float(value) for value in bbox_max],
            "center": [float(value) for value in center],
            "radius": radius,
        }
    finally:
        del records


def read_binary_little_endian_ply_layout(path: Path) -> tuple[int, np.dtype, int]:
    if not path.exists() or path.stat().st_size <= 0:
        raise ValueError(f"Missing non-empty PLY: {path}")

    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    current_element: str | None = None
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii", errors="strict").strip()
        fmt = handle.readline().decode("ascii", errors="strict").strip()
        if first != "ply" or fmt != "format binary_little_endian 1.0":
            raise ValueError(f"Unsupported PLY format: {path}")

        while True:
            line_bytes = handle.readline()
            if not line_bytes:
                raise ValueError(f"PLY header terminator missing: {path}")
            line = line_bytes.decode("ascii", errors="strict").strip()
            if line == "end_header":
                data_offset = handle.tell()
                break
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
                continue
            if current_element == "vertex" and len(parts) == 3 and parts[0] == "property":
                property_type, property_name = parts[1], parts[2]
                dtype = PLY_TYPE_MAP.get(property_type)
                if dtype is None:
                    raise ValueError(f"Unsupported PLY vertex property type {property_type!r}: {path}")
                vertex_properties.append((property_name, dtype))

    if vertex_count is None or vertex_count <= 0:
        raise ValueError(f"PLY vertex count missing or empty: {path}")
    names = {name for name, _ in vertex_properties}
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY vertex xyz properties missing: {path}")
    return vertex_count, np.dtype(vertex_properties), data_offset


def recommended_training_camera_view(
    scene_dir: Path,
    bounds: dict[str, Any],
    *,
    preferred_image_names: list[str] | None = None,
) -> dict[str, Any] | None:
    sparse_dir = scene_dir / "sparse" / "0"
    images_bin = sparse_dir / "images.bin"
    cameras_bin = sparse_dir / "cameras.bin"
    if not images_bin.exists() or not cameras_bin.exists():
        return None

    try:
        import pycolmap
    except Exception:
        return None

    reconstruction = pycolmap.Reconstruction(sparse_dir)
    extrinsics = getattr(reconstruction, "images", {}) or {}
    intrinsics = getattr(reconstruction, "cameras", {}) or {}

    if not extrinsics:
        return None

    sorted_extrinsics = sorted(extrinsics.values(), key=lambda item: str(item.name))
    selected = sorted_extrinsics[0]
    try:
        rotation = np.asarray(selected.cam_from_world.rotation.matrix(), dtype=np.float32)
        translation = np.asarray(selected.cam_from_world.translation, dtype=np.float32)
    except Exception:
        qvec = np.asarray(getattr(selected, "qvec", []), dtype=np.float32)
        tvec = np.asarray(getattr(selected, "tvec", []), dtype=np.float32)
        if qvec.shape[0] != 4 or tvec.shape[0] != 3:
            return None
        rotation = qvec_to_rotmat(qvec)
        translation = tvec
    camera_to_world = rotation.T
    position = (-camera_to_world @ translation).astype(np.float32)
    forward = normalize(camera_to_world @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    up = normalize(camera_to_world @ np.asarray([0.0, -1.0, 0.0], dtype=np.float32))
    if position is None or forward is None or up is None:
        return None

    radius = float(bounds["radius"])
    target = position + forward * max(radius, 0.35)

    view: dict[str, Any] = {
        "source": "first_training_camera",
        "image_name": str(selected.name),
        "position": [float(value) for value in position],
        "target": [float(value) for value in target],
        "up": [float(value) for value in up],
    }
    fov_y_degrees = camera_fov_y_degrees(intrinsics.get(selected.camera_id))
    if fov_y_degrees is not None:
        view["fov_y_degrees"] = fov_y_degrees
    return view


def normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-6:
        return None
    return (vector / norm).astype(np.float32)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(value) for value in qvec]
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float32,
    )


def camera_fov_y_degrees(camera: Any) -> float | None:
    if camera is None:
        return None
    params = [float(value) for value in getattr(camera, "params", [])]
    height = float(getattr(camera, "height", 0) or 0)
    model = str(getattr(camera, "model", "") or "")
    if height <= 0 or not params:
        return None
    focal_y = params[1] if model == "PINHOLE" and len(params) >= 2 else params[0]
    if focal_y <= 0:
        return None
    return float(math.degrees(2.0 * math.atan(height / (2.0 * focal_y))))


FAR_NOISE_PROFILE_FACTORS = {
    "indoor_full": 1.15,
    "mixed_balanced": 1.25,
    "outdoor_fast_clean": 1.60,
}
FAR_NOISE_LOW_OPACITY = 0.04


def _ply_opacity_to_probability(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _rewrite_vertex_count(header: bytes, vertex_count: int) -> bytes:
    lines = header.decode("ascii", errors="strict").splitlines(keepends=True)
    rewritten = []
    replaced = False
    for line in lines:
        if line.startswith("element vertex "):
            suffix = "\r\n" if line.endswith("\r\n") else "\n"
            rewritten.append(f"element vertex {vertex_count}{suffix}")
            replaced = True
        else:
            rewritten.append(line)
    if not replaced:
        raise ValueError("PLY vertex count missing")
    return "".join(rewritten).encode("ascii")
