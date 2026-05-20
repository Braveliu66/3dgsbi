from __future__ import annotations

import json
import shutil
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from app.fine.colmap_cli import _run_colmap_command, _run_colmap_with_gpu_fallback, detect_colmap_capabilities
from app.fine.option_utils import read_int
from app.fine.preprocess import ensure_colmap_sparse_zero
from app.fine.types import FineFailure
from app.preview.utils import image_files

try:
    import numpy as np
except Exception:  # pragma: no cover - worker image dependency
    np = None


Progress = Callable[[str, int, str], None]


@dataclass(frozen=True, slots=True)
class EapAugmentationResult:
    original_points: int
    eap_points: int
    multiplier: float
    aug_images: int
    points_bin: Path
    points_ply: Path
    meta_json: Path
    elapsed_seconds: float

    def metrics(self) -> dict[str, Any]:
        return {
            "fine_eap_enabled": True,
            "fine_eap_original_points": self.original_points,
            "fine_eap_points": self.eap_points,
            "fine_eap_multiplier": self.multiplier,
            "fine_eap_aug_images": self.aug_images,
            "fine_eap_points_bin": str(self.points_bin),
            "fine_eap_points_ply": str(self.points_ply),
            "fine_eap_elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True, slots=True)
class EapCameraSpec:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


def run_eap_augmentation(
    scene_dir: Path,
    options: dict[str, Any],
    *,
    prefer_gpu: bool,
    gpu_index: str | None,
    progress: Progress,
) -> EapAugmentationResult:
    started = time.monotonic()
    capabilities = detect_colmap_capabilities()
    _require_colmap_command(capabilities.commands, capabilities.help_text, "point_triangulator")

    try:
        import pycolmap
    except Exception as exc:  # pragma: no cover - worker image dependency
        raise FineFailure("EAP_PYCOLMAP_UNAVAILABLE", f"EAP requires pycolmap to read COLMAP sparse models: {exc}") from exc

    sparse_dir = ensure_colmap_sparse_zero(scene_dir / "sparse")
    images_dir = scene_dir / "images"
    if not images_dir.exists():
        raise FineFailure("EAP_IMAGES_MISSING", f"EAP input images not found: {images_dir}")

    reconstruction = pycolmap.Reconstruction(sparse_dir)
    original_points = len(getattr(reconstruction, "points3D", {}) or {})
    if original_points <= 0:
        raise FineFailure("EAP_EMPTY_SPARSE_MODEL", "EAP requires a non-empty COLMAP sparse reconstruction")
    camera_spec = _single_undistorted_camera(reconstruction)

    eps = read_int(options.get("fine_eap_dbscan_eps"), 30, minimum=1, maximum=256)
    min_samples = read_int(options.get("fine_eap_min_samples"), 10, minimum=1, maximum=512)
    mask_radius = read_int(options.get("fine_eap_mask_radius"), 20, minimum=1, maximum=256)
    max_multiplier = read_int(options.get("fine_eap_max_point_multiplier"), 10, minimum=1, maximum=100)

    workspace = scene_dir / "eap_workspace"
    aug_dir = scene_dir / "augimages"
    if workspace.exists():
        shutil.rmtree(workspace)
    if aug_dir.exists():
        shutil.rmtree(aug_dir)
    work_images_dir = workspace / "images"
    seed_model_dir = workspace / "seed_model"
    output_sparse_dir = workspace / "sparse"
    database_path = workspace / "database.db"
    work_images_dir.mkdir(parents=True, exist_ok=True)
    aug_dir.mkdir(parents=True, exist_ok=True)
    output_sparse_dir.mkdir(parents=True, exist_ok=True)

    source_images = image_files(images_dir)
    if not source_images:
        raise FineFailure("EAP_IMAGES_MISSING", f"EAP found no images under {images_dir}")

    image_by_name = _reconstruction_images_by_name(reconstruction)
    projected_by_name = _observed_points_by_image(reconstruction)
    aug_count = 0
    for source in source_images:
        shutil.copy2(source, work_images_dir / source.name)
        if source.name not in image_by_name:
            continue
        projected = projected_by_name.get(source.name, [])
        if not projected:
            continue
        aug_path = aug_dir / _augmented_image_name(source.name)
        _write_attention_image(
            source,
            aug_path,
            projected,
            eps=eps,
            min_samples=min_samples,
            mask_radius=mask_radius,
        )
        shutil.copy2(aug_path, work_images_dir / aug_path.name)
        aug_count += 1

    if aug_count == 0:
        raise FineFailure("EAP_NO_AUG_IMAGES", "EAP could not generate any augmented images from the sparse observations")

    gpu_flag = "1" if prefer_gpu else "0"
    progress("fine_eap_features", 41, f"extracting EAP features from {len(source_images) + aug_count} images")
    feature_cmd = _build_eap_feature_command(
        capabilities.executable,
        database_path,
        work_images_dir,
        camera_spec,
        use_gpu=gpu_flag,
        gpu_index=gpu_index,
    )
    _run_colmap_with_gpu_fallback(feature_cmd, "--SiftExtraction.use_gpu", progress, "fine_eap_features", 41)

    db_images = _read_database_images(database_path)
    _sync_eap_database_single_camera(database_path, camera_spec)
    _write_seed_text_model(seed_model_dir, reconstruction, db_images)

    progress("fine_eap_matching", 41, "matching EAP original and augmented images")
    match_cmd = [
        capabilities.executable,
        "exhaustive_matcher",
        "--database_path",
        str(database_path),
        "--SiftMatching.use_gpu",
        gpu_flag,
    ]
    if gpu_index:
        match_cmd.extend(["--SiftMatching.gpu_index", gpu_index])
    _run_colmap_with_gpu_fallback(match_cmd, "--SiftMatching.use_gpu", progress, "fine_eap_matching", 41)

    progress("fine_eap_triangulating", 42, "triangulating EAP enhanced sparse pointcloud")
    _run_colmap_command(
        [
            capabilities.executable,
            "point_triangulator",
            "--database_path",
            str(database_path),
            "--image_path",
            str(work_images_dir),
            "--input_path",
            str(seed_model_dir),
            "--output_path",
            str(output_sparse_dir),
        ],
        stage="fine_eap_triangulating",
        progress=progress,
        progress_value=42,
    )

    enhanced = pycolmap.Reconstruction(output_sparse_dir)
    eap_points = len(getattr(enhanced, "points3D", {}) or {})
    multiplier = _check_point_multiplier(original_points, eap_points, max_multiplier)

    points_bin, points_ply = _install_eap_points(output_sparse_dir, sparse_dir, enhanced)
    meta_json = sparse_dir / "points3D_eap_meta.json"
    elapsed = round(time.monotonic() - started, 3)
    meta = {
        "original_points": original_points,
        "eap_points": eap_points,
        "multiplier": multiplier,
        "aug_images": aug_count,
        "dbscan_eps": eps,
        "min_samples": min_samples,
        "mask_radius": mask_radius,
        "max_point_multiplier": max_multiplier,
        "points_bin": str(points_bin),
        "points_ply": str(points_ply),
        "elapsed_seconds": elapsed,
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return EapAugmentationResult(
        original_points=original_points,
        eap_points=eap_points,
        multiplier=multiplier,
        aug_images=aug_count,
        points_bin=points_bin,
        points_ply=points_ply,
        meta_json=meta_json,
        elapsed_seconds=elapsed,
    )


def _require_colmap_command(commands: set[str], help_text: str, command: str) -> None:
    if command in commands or command in help_text:
        return
    raise FineFailure("EAP_COLMAP_UNSUPPORTED", f"EAP requires COLMAP command '{command}', but it is not available")


def _reconstruction_images_by_name(reconstruction: Any) -> dict[str, Any]:
    return {str(getattr(image, "name", "")): image for image in _mapping_values(getattr(reconstruction, "images", {})) if getattr(image, "name", None)}


def _observed_points_by_image(reconstruction: Any) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = {}
    for image in _mapping_values(getattr(reconstruction, "images", {})):
        name = str(getattr(image, "name", ""))
        if not name:
            continue
        points = []
        for point2d in getattr(image, "points2D", []) or []:
            point3d_id = int(getattr(point2d, "point3D_id", -1) or -1)
            if point3d_id < 0:
                continue
            xy = getattr(point2d, "xy", None)
            if xy is None:
                continue
            try:
                x, y = float(xy[0]), float(xy[1])
            except (TypeError, ValueError, IndexError):
                continue
            points.append((x, y))
        result[name] = points
    return result


def _write_attention_image(
    source: Path,
    output: Path,
    projected_points: list[tuple[float, float]],
    *,
    eps: int,
    min_samples: int,
    mask_radius: int,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - worker image dependency
        raise FineFailure("EAP_PIL_UNAVAILABLE", f"EAP requires Pillow to write augmented images: {exc}") from exc

    with Image.open(source) as image:
        rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    for x, y in _dense_points(projected_points, eps=eps, min_samples=min_samples):
        draw.ellipse((x - mask_radius, y - mask_radius, x + mask_radius, y + mask_radius), fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(output, quality=95)


def _dense_points(points: list[tuple[float, float]], *, eps: int, min_samples: int) -> list[tuple[float, float]]:
    if len(points) < min_samples or np is None:
        return points
    try:
        from sklearn.cluster import DBSCAN
    except Exception:
        return points
    labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(np.asarray(points, dtype=np.float32)).labels_
    dense = [point for point, label in zip(points, labels) if int(label) >= 0]
    return dense or points


def _augmented_image_name(name: str) -> str:
    path = Path(name)
    return f"{path.stem}_aug{path.suffix or '.jpg'}"


def _source_image_name_for_aug(name: str) -> str:
    path = Path(name)
    stem = path.stem
    if stem.endswith("_aug"):
        return f"{stem[:-4]}{path.suffix}"
    return name


def _read_database_images(database_path: Path) -> dict[str, int]:
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute("select image_id, name from images").fetchall()
        return {str(name): int(image_id) for image_id, name in rows}
    finally:
        connection.close()


def _single_undistorted_camera(reconstruction: Any) -> EapCameraSpec:
    cameras = _mapping_by_id(getattr(reconstruction, "cameras", {}), "camera_id")
    if len(cameras) != 1:
        raise FineFailure("EAP_REQUIRES_SINGLE_CAMERA", f"EAP supports one undistorted camera, got {len(cameras)}")
    camera_id, camera = next(iter(cameras.items()))
    model = _camera_model_name(camera)
    params = tuple(float(value) for value in _iter_values(getattr(camera, "params", [])))
    expected_params = {"SIMPLE_PINHOLE": 3, "PINHOLE": 4}.get(model)
    if expected_params is None or len(params) != expected_params:
        raise FineFailure(
            "EAP_CAMERA_MODEL_UNSUPPORTED",
            f"EAP requires undistorted SIMPLE_PINHOLE/PINHOLE single-camera input, got {model}",
        )
    width = int(getattr(camera, "width", 0) or 0)
    height = int(getattr(camera, "height", 0) or 0)
    focal_params = params[:1] if model == "SIMPLE_PINHOLE" else params[:2]
    if width <= 0 or height <= 0 or any(value <= 0 for value in focal_params):
        raise FineFailure("EAP_CAMERA_INVALID", f"EAP found invalid camera intrinsics for {model}")
    return EapCameraSpec(camera_id=int(camera_id), model=model, width=width, height=height, params=params)


def _build_eap_feature_command(
    executable: str,
    database_path: Path,
    image_path: Path,
    camera: EapCameraSpec,
    *,
    use_gpu: str,
    gpu_index: str | None,
) -> list[str]:
    command = [
        executable,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        camera.model,
        "--ImageReader.camera_params",
        _camera_params_text(camera.params),
        "--SiftExtraction.use_gpu",
        use_gpu,
    ]
    if gpu_index:
        command.extend(["--SiftExtraction.gpu_index", gpu_index])
    return command


def _camera_params_text(params: Iterable[float]) -> str:
    return ",".join(_format_float(value) for value in params)


def _sync_eap_database_single_camera(database_path: Path, camera: EapCameraSpec) -> None:
    model_ids = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1}
    model_id = model_ids.get(camera.model)
    if model_id is None:
        raise FineFailure("EAP_CAMERA_MODEL_UNSUPPORTED", f"EAP cannot sync unsupported camera model {camera.model}")
    params_blob = struct.pack("<" + "d" * len(camera.params), *camera.params)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("delete from cameras")
        connection.execute(
            "insert into cameras(camera_id, model, width, height, params, prior_focal_length) values (?, ?, ?, ?, ?, ?)",
            (camera.camera_id, model_id, camera.width, camera.height, params_blob, 1),
        )
        connection.execute("update images set camera_id = ?", (camera.camera_id,))
        connection.commit()
    finally:
        connection.close()


def _write_seed_text_model(seed_model_dir: Path, reconstruction: Any, db_images: dict[str, int]) -> None:
    seed_model_dir.mkdir(parents=True, exist_ok=True)
    images_by_name = _reconstruction_images_by_name(reconstruction)
    used_camera_ids: set[int] = set()

    image_records = []
    for db_name, db_id in sorted(db_images.items(), key=lambda item: item[1]):
        source_name = _source_image_name_for_aug(db_name)
        source_image = images_by_name.get(source_name)
        if source_image is None:
            continue
        qvec, tvec = _image_pose_qvec_tvec(source_image, source_name)
        camera_id = int(getattr(source_image, "camera_id"))
        used_camera_ids.add(camera_id)
        image_records.append((db_id, qvec, tvec, camera_id, db_name))

    if not image_records:
        raise FineFailure("EAP_SEED_MODEL_EMPTY", "EAP could not map database image names back to registered COLMAP images")

    cameras_by_id = _mapping_by_id(getattr(reconstruction, "cameras", {}), "camera_id")
    with (seed_model_dir / "cameras.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for camera_id in sorted(used_camera_ids):
            camera = cameras_by_id.get(camera_id)
            if camera is None:
                raise FineFailure("EAP_CAMERA_MISSING", f"EAP seed model missing camera_id={camera_id}")
            model = _camera_model_name(camera)
            params = " ".join(_format_float(value) for value in _iter_values(getattr(camera, "params", [])))
            handle.write(f"{camera_id} {model} {int(getattr(camera, 'width'))} {int(getattr(camera, 'height'))} {params}\n")

    with (seed_model_dir / "images.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id, qvec, tvec, camera_id, name in image_records:
            pose = " ".join(_format_float(value) for value in [*qvec, *tvec])
            handle.write(f"{image_id} {pose} {camera_id} {name}\n\n")

    with (seed_model_dir / "points3D.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")


def _camera_model_name(camera: Any) -> str:
    model = getattr(camera, "model_name", None) or getattr(camera, "model", None)
    if hasattr(model, "name"):
        model = model.name
    text = str(model)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    text = text.upper()
    return text or "PINHOLE"


def _image_pose_qvec_tvec(image: Any, name: str) -> tuple[list[float], list[float]]:
    qvec = list(_iter_values(getattr(image, "qvec", None)))
    tvec = list(_iter_values(getattr(image, "tvec", None)))
    if len(qvec) == 4 and len(tvec) == 3:
        return [float(item) for item in qvec], [float(item) for item in tvec]

    cam_from_world = getattr(image, "cam_from_world", None)
    pose = cam_from_world() if callable(cam_from_world) else cam_from_world
    if pose is None:
        raise FineFailure("EAP_MODEL_UNSUPPORTED", f"EAP cannot read pose for image {name}")
    xyzw = list(_iter_values(getattr(getattr(pose, "rotation", None), "quat", None)))
    translation = list(_iter_values(getattr(pose, "translation", None)))
    if len(xyzw) != 4 or len(translation) != 3:
        raise FineFailure("EAP_MODEL_UNSUPPORTED", f"EAP cannot read pose for image {name}")
    # pycolmap Rotation3d stores [x, y, z, w], while COLMAP text images use [qw, qx, qy, qz].
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])], [float(item) for item in translation]


def _check_point_multiplier(original_points: int, eap_points: int, max_multiplier: int) -> float:
    multiplier = round(float(eap_points) / max(float(original_points), 1.0), 6)
    if eap_points > original_points * max_multiplier:
        raise FineFailure(
            "EAP_POINT_MULTIPLIER_EXCEEDED",
            f"EAP produced {eap_points} points from {original_points} original points, exceeding multiplier limit {max_multiplier}",
        )
    return multiplier


def _install_eap_points(output_sparse_dir: Path, sparse_dir: Path, reconstruction: Any) -> tuple[Path, Path]:
    output_bin = output_sparse_dir / "points3D.bin"
    output_txt = output_sparse_dir / "points3D.txt"
    target_bin = sparse_dir / "points3D_eap.bin"
    target_txt = sparse_dir / "points3D_eap.txt"
    target_ply = sparse_dir / "points3D_eap.ply"
    if output_bin.exists():
        shutil.copy2(output_bin, target_bin)
    elif output_txt.exists():
        shutil.copy2(output_txt, target_txt)
    else:
        raise FineFailure("EAP_OUTPUT_MISSING", f"EAP point_triangulator did not produce points3D under {output_sparse_dir}")
    _write_reconstruction_points_ply(reconstruction, target_ply)
    return target_bin if target_bin.exists() else target_txt, target_ply


def _write_reconstruction_points_ply(reconstruction: Any, output_path: Path) -> None:
    points = []
    for point in _mapping_values(getattr(reconstruction, "points3D", {})):
        xyz = _array_values(getattr(point, "xyz", None), 3, "point xyz")
        color = list(_iter_values(getattr(point, "color", [255, 255, 255])))
        if len(color) != 3:
            color = [255, 255, 255]
        points.append((xyz, [max(0, min(255, int(round(value)))) for value in color]))
    if not points:
        raise FineFailure("EAP_OUTPUT_EMPTY", "EAP point_triangulator produced an empty point cloud")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for xyz, color in points:
            handle.write(
                f"{_format_float(xyz[0])} {_format_float(xyz[1])} {_format_float(xyz[2])} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def _mapping_values(mapping: Any) -> Iterable[Any]:
    if hasattr(mapping, "values"):
        return mapping.values()
    return mapping or []


def _mapping_by_id(mapping: Any, attr: str) -> dict[int, Any]:
    result = {}
    if hasattr(mapping, "items"):
        for key, value in mapping.items():
            result[int(getattr(value, attr, key))] = value
    else:
        for value in mapping or []:
            result[int(getattr(value, attr))] = value
    return result


def _array_values(value: Any, length: int, label: str) -> list[float]:
    values = list(_iter_values(value))
    if len(values) != length:
        raise FineFailure("EAP_MODEL_UNSUPPORTED", f"EAP cannot read {label}")
    return [float(item) for item in values]


def _iter_values(value: Any) -> Iterable[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"
