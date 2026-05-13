from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from app.fine.preprocess import SceneBuildResult
from app.fine.types import FineFailure
from app.preview.types import PreviewFailure
from app.preview.vendor.litevggt_runtime import run_litevggt_reconstruction


Progress = Callable[[str, int, str], None]


def build_litevggt_scene(
    input_dir: Path,
    scene_dir: Path,
    *,
    model_cache_dir: Path,
    options: dict[str, Any],
    progress: Progress,
) -> SceneBuildResult:
    weight = model_cache_dir / "litevggt" / "te_dict.pt"
    if not weight.exists() or weight.stat().st_size <= 0:
        raise FineFailure("LITEVGGT_WEIGHT_MISSING", f"LiteVGGT weight is missing: {weight}")

    params = {
        "keep_ratio": float(options.get("fine_litevggt_keep_ratio") or 1.0),
        "max_points": int(options.get("fine_litevggt_max_points") or 1_500_000),
        "spatial_keep_quantile": float(options.get("fine_litevggt_spatial_keep_quantile") or 0.999),
        "preserve_full_image": read_bool(options.get("fine_litevggt_preserve_full_image"), True),
        "letterbox_size": int(options.get("fine_litevggt_letterbox_size") or 518),
        "max_input_frames": read_optional_int(options.get("fine_litevggt_max_input_frames")),
        "frame_stride": read_optional_int(options.get("fine_litevggt_frame_stride")),
        "depth_conf_thresh": read_optional_float(options.get("fine_litevggt_depth_conf_thresh"), None),
        "preprocess_mode": str(options.get("fine_litevggt_preprocess_mode") or "pad"),
        "frame_selection": str(options.get("fine_litevggt_frame_selection") or "all"),
        "min_scene_change": float(options.get("fine_litevggt_min_scene_change") or 0.0),
        "edge_keep_ratio": float(options.get("fine_litevggt_edge_keep_ratio") or 0.15),
        "axis_trim_low_quantile": float(options.get("fine_litevggt_axis_trim_low_quantile") or 0.0005),
        "axis_trim_high_quantile": float(options.get("fine_litevggt_axis_trim_high_quantile") or 0.9995),
        "selection_strategy": str(options.get("fine_litevggt_point_selection_strategy") or "per_frame"),
        "window_size": int(options.get("fine_litevggt_window_size") or 48),
        "window_overlap": int(options.get("fine_litevggt_window_overlap") or 16),
        "oom_window_sizes": read_int_list(options.get("fine_litevggt_oom_window_sizes"), [32, 16, 8]),
    }
    print(
        "[fine-litevggt-scene] start "
        f"input_dir={input_dir} scene_dir={scene_dir} weight={weight} weight_bytes={weight.stat().st_size} "
        + " ".join(f"{key}={value}" for key, value in params.items()),
        flush=True,
    )

    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    images_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()

    def report(stage: str, value: int, message: str) -> None:
        progress(f"fine_{stage}", 22 + int(value * 0.18), message)

    try:
        reconstruction = run_litevggt_reconstruction(
            input_dir=input_dir,
            checkpoint_path=weight,
            keep_ratio=params["keep_ratio"],
            max_points=params["max_points"],
            spatial_keep_quantile=params["spatial_keep_quantile"],
            preserve_full_image=params["preserve_full_image"],
            letterbox_size=params["letterbox_size"],
            max_input_frames=params["max_input_frames"],
            frame_stride=params["frame_stride"],
            depth_conf_thresh=params["depth_conf_thresh"],
            preprocess_mode=params["preprocess_mode"],
            frame_selection=params["frame_selection"],
            min_scene_change=params["min_scene_change"],
            edge_keep_ratio=params["edge_keep_ratio"],
            axis_trim_low_quantile=params["axis_trim_low_quantile"],
            axis_trim_high_quantile=params["axis_trim_high_quantile"],
            selection_strategy=params["selection_strategy"],
            inference_mode="windowed",
            window_size=params["window_size"],
            window_overlap=params["window_overlap"],
            oom_window_sizes=params["oom_window_sizes"],
            progress=report,
        )
    except PreviewFailure as exc:
        raise FineFailure(exc.code, exc.message) from exc
    print(
        "[fine-litevggt-scene] reconstruction complete "
        f"images_shape={getattr(reconstruction.images, 'shape', None)} points_shape={getattr(reconstruction.points, 'shape', None)} "
        f"colors_shape={getattr(reconstruction.colors, 'shape', None)} metrics={reconstruction.metrics}",
        flush=True,
    )

    image_names = write_training_images(reconstruction.images, images_dir)
    write_cameras_txt(sparse_dir / "cameras.txt", reconstruction.intrinsics, reconstruction.images)
    write_images_txt(sparse_dir / "images.txt", reconstruction.w2c, image_names)
    write_points3d_ply(sparse_dir / "points3D.ply", reconstruction.points, reconstruction.colors)
    print(
        "[fine-litevggt-scene] COLMAP-compatible files written "
        f"images_dir={images_dir} image_count={len(image_names)} "
        f"cameras_txt={sparse_dir / 'cameras.txt'} images_txt={sparse_dir / 'images.txt'} "
        f"points3d_ply={sparse_dir / 'points3D.ply'} points3d_bytes={(sparse_dir / 'points3D.ply').stat().st_size}",
        flush=True,
    )

    elapsed = round(time.monotonic() - started, 3)
    selected_count = len(image_names)
    point_count = int(reconstruction.points.shape[0])
    return SceneBuildResult(
        scene_dir=scene_dir,
        backend="litevggt",
        image_count=int(reconstruction.metrics.get("original_frame_count", selected_count)),
        registered_images=selected_count,
        point_count=point_count,
        metrics={
            "sfm_backend": "litevggt",
            "litevggt_scene_elapsed_seconds": elapsed,
            "litevggt_registered_images": selected_count,
            "litevggt_initial_points": point_count,
            **{f"litevggt_{key}": value for key, value in reconstruction.metrics.items()},
        },
    )


def write_training_images(images: np.ndarray, images_dir: Path) -> list[str]:
    names = []
    for index, image in enumerate(images):
        name = f"{index:06d}.jpg"
        array = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(array).save(images_dir / name, format="JPEG", quality=94)
        names.append(name)
    return names


def write_cameras_txt(path: Path, intrinsics: np.ndarray, images: np.ndarray) -> None:
    height = int(images.shape[1])
    width = int(images.shape[2])
    lines = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(intrinsics)}",
    ]
    for index, intrinsic in enumerate(intrinsics, start=1):
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        lines.append(f"{index} PINHOLE {width} {height} {fx:.8f} {fy:.8f} {cx:.8f} {cy:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_images_txt(path: Path, w2c: np.ndarray, image_names: list[str]) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(image_names)}, mean observations per image: 0",
    ]
    for index, name in enumerate(image_names, start=1):
        matrix = w2c[index - 1]
        rotation = np.asarray(matrix[:3, :3], dtype=np.float64)
        translation = np.asarray(matrix[:3, 3], dtype=np.float64)
        qvec = rotmat2qvec(rotation)
        lines.append(
            f"{index} {qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} {qvec[3]:.12f} "
            f"{translation[0]:.12f} {translation[1]:.12f} {translation[2]:.12f} {index} {name}"
        )
        lines.append("0.0 0.0 -1")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_points3d_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = np.clip(np.asarray(colors).reshape(-1, 3), 0, 255).astype(np.uint8)
    valid = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid]
    rgb = rgb[valid]
    if xyz.shape[0] == 0:
        raise FineFailure("LITEVGGT_EMPTY_POINT_CLOUD", "LiteVGGT produced no valid points for fine initialization")

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    records = np.empty(xyz.shape[0], dtype=dtype)
    records["x"] = xyz[:, 0]
    records["y"] = xyz[:, 1]
    records["z"] = xyz[:, 2]
    records["nx"] = 0.0
    records["ny"] = 0.0
    records["nz"] = 0.0
    records["red"] = rgb[:, 0]
    records["green"] = rgb[:, 1]
    records["blue"] = rgb[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(records.tobytes())


def rotmat2qvec(rotation: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    matrix = np.array(
        [
            [rxx - ryy - rzz, 0, 0, 0],
            [ryx + rxy, ryy - rxx - rzz, 0, 0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(matrix)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def read_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "auto", "none"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_optional_float(value: Any, fallback: float | None) -> float | None:
    if value is None:
        return fallback
    normalized = str(value).strip().lower()
    if normalized in {"", "auto"}:
        return fallback
    if normalized in {"none", "off", "false"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def read_int_list(value: Any, fallback: list[int]) -> list[int]:
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = str(value).replace(";", ",").split(",")

    parsed: list[int] = []
    for item in values:
        try:
            parsed.append(int(str(item).strip()))
        except (TypeError, ValueError):
            continue
    return parsed or fallback


def read_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return fallback
