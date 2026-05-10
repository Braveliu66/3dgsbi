from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
AMB3R_PATCH_SIZE = 14
AMB3R_AUTO_LONG_SIDES = (896, 756, 672, 518)


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


@dataclass(slots=True)
class Amb3rResolutionPlan:
    requested: str
    selected_width: int
    selected_height: int
    fallbacks: list[tuple[int, int]]
    token_budget: int
    estimated_tokens: int
    target_aspect: float


@dataclass(slots=True)
class Amb3rWindow:
    start: int
    end: int

    @property
    def indices(self) -> list[int]:
        return list(range(self.start, self.end))


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
    options: dict[str, Any] | None = None,
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
    resolution_plan = resolve_amb3r_resolution(files, options or {})

    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    images_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    progress("fine_amb3r_loading_model", 24, f"loading AMB3R checkpoint: {checkpoint_path.name}")
    from app.fine.amb3r_runtime.amb3r.model import AMB3R
    from app.fine.amb3r_runtime.sfm.pipeline import AMB3R_SfM

    device = "cuda:0"
    model = AMB3R(device=device)
    model.load_weights(str(checkpoint_path), strict=False)
    model.to(device)
    model.eval()
    model_device = next(model.parameters()).device
    if model_device.type != "cuda":
        raise FineFailure("AMB3R_DEVICE_MISMATCH", f"AMB3R model is on {model_device}, expected CUDA")

    pipeline = AMB3R_SfM(model, progress=progress, options=options or {})

    processed: list[ProcessedAmb3rImage] | None = None
    scene_data: dict[str, Any] | None = None
    oom_retries = 0
    attempted: list[str] = []
    selected_width = resolution_plan.selected_width
    selected_height = resolution_plan.selected_height
    for width, height in [(resolution_plan.selected_width, resolution_plan.selected_height), *resolution_plan.fallbacks]:
        attempted.append(f"{width}x{height}")
        processed = prepare_amb3r_images(files, images_dir, width=width, height=height, target_aspect=resolution_plan.target_aspect)
        real_count = len(processed)
        image_tensor = torch.stack(
            [torch.from_numpy(np.transpose(item.image, (2, 0, 1))) for item in processed],
            dim=0,
        ).unsqueeze(0)

        window_plan = plan_amb3r_windows(real_count, options or {})
        is_windowed = len(window_plan) > 1
        progress("fine_amb3r_sfm", 32, f"running AMB3R-SfM on {real_count} images at {width}x{height}")
        try:
            if is_windowed:
                scene_data = run_amb3r_windowed_scene(
                    processed,
                    image_tensor,
                    pipeline,
                    progress=progress,
                    width=width,
                    height=height,
                    windows=window_plan,
                    options=options or {},
                )
            else:
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        memory = pipeline.run(image_tensor.to(device))
                scene_data = scene_data_from_memory(
                    memory,
                    processed,
                    list(range(real_count)),
                    width=width,
                    height=height,
                    metrics={
                        "amb3r_windowed": False,
                        "amb3r_window_count": 1,
                        "amb3r_window_size": real_count,
                        "amb3r_window_overlap": 0,
                        "amb3r_window_registered_images": [int(len(valid_registered_indices(to_numpy(memory.poses), real_count, {int(index) for index in getattr(memory, "unmapped_frames", set())})))],
                        "amb3r_window_failed_count": 0,
                        "amb3r_window_alignment_failed_count": 0,
                    },
                )
            torch.cuda.synchronize()
            selected_width = width
            selected_height = height
            break
        except RuntimeError as exc:
            is_oom = "out of memory" in str(exc).lower()
            is_last_attempt = (width, height) == ((resolution_plan.fallbacks[-1]) if resolution_plan.fallbacks else (resolution_plan.selected_width, resolution_plan.selected_height))
            if not is_oom or is_last_attempt:
                raise
            oom_retries += 1
            scene_data = None
            del image_tensor
            torch.cuda.empty_cache()

    if processed is None or scene_data is None:
        raise FineFailure("AMB3R_RECONSTRUCTION_FAILED", "AMB3R did not produce a reconstruction")
    real_count = len(processed)

    poses_c2w = scene_data["poses_c2w"]
    intrinsics = scene_data["intrinsics"]
    points = scene_data["points"]
    confidence = scene_data["confidence"]
    colors = scene_data["colors"]
    registered_indices = scene_data["registered_indices"]
    if len(registered_indices) < 3:
        raise FineFailure("AMB3R_RECONSTRUCTION_INCOMPLETE", f"AMB3R registered {len(registered_indices)}/{real_count} images")

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
            "amb3r_unmapped_images": real_count - len(registered_indices),
            "amb3r_sparse_points": point_count,
            "amb3r_resolution": f"{selected_width}x{selected_height}",
            "amb3r_resolution_requested": resolution_plan.requested,
            "amb3r_resolution_selected": f"{selected_width}x{selected_height}",
            "amb3r_resolution_attempts": attempted,
            "amb3r_oom_retries": oom_retries,
            "amb3r_token_budget": resolution_plan.token_budget,
            "amb3r_estimated_tokens": int(real_count * (selected_width // AMB3R_PATCH_SIZE) * (selected_height // AMB3R_PATCH_SIZE)),
            "amb3r_preserve_aspect": True,
            "amb3r_runtime_device": str(pipeline.device),
            "amb3r_model_device": str(model_device),
            "amb3r_sfm_memory_device": str(scene_data.get("memory_device", "unknown")),
            **scene_data["metrics"],
            **pipeline.metrics(),
            "amb3r_keep_ratio": keep_ratio,
            "amb3r_max_points": max_points,
            "amb3r_source_commit": AMB3R_COMMIT,
        },
    )


def resolve_amb3r_resolution(files: list[Path], options: dict[str, Any]) -> Amb3rResolutionPlan:
    aspect = median_image_aspect(files)
    requested = str(options.get("fine_amb3r_resolution") or "auto").strip().lower()
    if requested and requested != "auto":
        width, height = parse_resolution(requested)
        width = round_to_multiple(width, AMB3R_PATCH_SIZE)
        height = round_to_multiple(height, AMB3R_PATCH_SIZE)
        tokens = len(files) * (width // AMB3R_PATCH_SIZE) * (height // AMB3R_PATCH_SIZE)
        return Amb3rResolutionPlan(
            requested=requested,
            selected_width=width,
            selected_height=height,
            fallbacks=lower_resolution_fallbacks(width, height, aspect),
            token_budget=tokens,
            estimated_tokens=tokens,
            target_aspect=aspect,
        )

    budget = resolve_amb3r_token_budget(options)
    candidates = [resolution_for_long_side(long_side, aspect) for long_side in AMB3R_AUTO_LONG_SIDES]
    selected = candidates[-1]
    estimated_tokens = len(files) * (selected[0] // AMB3R_PATCH_SIZE) * (selected[1] // AMB3R_PATCH_SIZE)
    for candidate in candidates:
        tokens = len(files) * (candidate[0] // AMB3R_PATCH_SIZE) * (candidate[1] // AMB3R_PATCH_SIZE)
        if tokens <= budget:
            selected = candidate
            estimated_tokens = tokens
            break
    selected_index = candidates.index(selected)
    return Amb3rResolutionPlan(
        requested="auto",
        selected_width=selected[0],
        selected_height=selected[1],
        fallbacks=candidates[selected_index + 1 :],
        token_budget=budget,
        estimated_tokens=estimated_tokens,
        target_aspect=aspect,
    )


def median_image_aspect(files: list[Path]) -> float:
    aspects = []
    for path in files:
        with Image.open(path) as original:
            width, height = ImageOps.exif_transpose(original).size
        if height > 0:
            aspects.append(width / float(height))
    if not aspects:
        return AMB3R_WIDTH / float(AMB3R_HEIGHT)
    return float(np.median(np.asarray(aspects, dtype=np.float32)))


def parse_resolution(value: str) -> tuple[int, int]:
    normalized = value.replace("*", "x").replace(",", "x")
    parts = [part for part in normalized.split("x") if part]
    if len(parts) != 2:
        raise FineFailure("AMB3R_RESOLUTION_INVALID", f"invalid AMB3R resolution: {value}")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise FineFailure("AMB3R_RESOLUTION_INVALID", f"invalid AMB3R resolution: {value}") from exc
    if width < AMB3R_PATCH_SIZE or height < AMB3R_PATCH_SIZE:
        raise FineFailure("AMB3R_RESOLUTION_INVALID", f"AMB3R resolution is too small: {value}")
    return width, height


def resolve_amb3r_token_budget(options: dict[str, Any]) -> int:
    raw = options.get("fine_amb3r_token_budget")
    normalized = "" if raw is None else str(raw).strip().lower()
    if normalized not in {"", "auto"}:
        try:
            return max(10_000, int(raw))
        except (TypeError, ValueError) as exc:
            raise FineFailure("AMB3R_TOKEN_BUDGET_INVALID", f"invalid AMB3R token budget: {raw}") from exc
    try:
        import torch

        total_gb = torch.cuda.get_device_properties(0).total_memory / float(1024**3)
    except Exception:
        total_gb = 0.0
    if total_gb >= 24.0:
        return 220_000
    if total_gb >= 16.0:
        return 170_000
    return 110_000


def resolution_for_long_side(long_side: int, aspect: float) -> tuple[int, int]:
    if aspect >= 1.0:
        width = round_to_multiple(long_side, AMB3R_PATCH_SIZE)
        height = round_to_multiple(int(round(width / max(aspect, 1e-6))), AMB3R_PATCH_SIZE)
    else:
        height = round_to_multiple(long_side, AMB3R_PATCH_SIZE)
        width = round_to_multiple(int(round(height * aspect)), AMB3R_PATCH_SIZE)
    return max(AMB3R_PATCH_SIZE, width), max(AMB3R_PATCH_SIZE, height)


def lower_resolution_fallbacks(width: int, height: int, aspect: float) -> list[tuple[int, int]]:
    long_side = max(width, height)
    fallbacks = []
    for candidate_long_side in AMB3R_AUTO_LONG_SIDES:
        if candidate_long_side < long_side:
            fallback = resolution_for_long_side(candidate_long_side, aspect)
            if fallback not in fallbacks:
                fallbacks.append(fallback)
    return fallbacks


def round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / float(multiple))) * multiple)


def plan_amb3r_windows(image_count: int, options: dict[str, Any]) -> list[Amb3rWindow]:
    threshold = read_amb3r_int_option(options, "fine_amb3r_window_threshold", 120, minimum=3, maximum=10_000)
    windowed = read_amb3r_bool_option(options.get("fine_amb3r_windowed"), default=image_count > threshold)
    if not windowed:
        return [Amb3rWindow(0, image_count)]

    size = read_amb3r_int_option(options, "fine_amb3r_window_size", 64, minimum=3, maximum=max(3, image_count))
    overlap = read_amb3r_int_option(options, "fine_amb3r_window_overlap", 12, minimum=3, maximum=max(3, size - 1))
    if image_count <= size:
        return [Amb3rWindow(0, image_count)]

    windows: list[Amb3rWindow] = []
    start = 0
    while start < image_count:
        end = min(image_count, start + size)
        if end - start < 3 and windows:
            windows[-1] = Amb3rWindow(windows[-1].start, image_count)
            break
        windows.append(Amb3rWindow(start, end))
        if end == image_count:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return windows


def read_amb3r_bool_option(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def read_amb3r_int_option(options: dict[str, Any], key: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = options.get(key)
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, min(maximum, value))


def run_amb3r_windowed_scene(
    processed: list[ProcessedAmb3rImage],
    image_tensor,
    pipeline,
    *,
    progress: Progress,
    width: int,
    height: int,
    windows: list[Amb3rWindow],
    options: dict[str, Any],
) -> dict[str, Any]:
    import torch

    image_count = len(processed)
    global_poses = np.zeros((image_count, 4, 4), dtype=np.float32)
    global_intrinsics = np.zeros((image_count, 3, 3), dtype=np.float32)
    registered: set[int] = set()
    point_chunks: list[np.ndarray] = []
    confidence_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    window_registered_counts: list[int] = []
    failed_windows = 0
    alignment_failures = 0
    overlap = max(0, windows[0].end - windows[1].start) if len(windows) > 1 else 0

    for window_index, window in enumerate(windows):
        window_indices = window.indices
        progress(
            "fine_amb3r_window_sfm",
            min(39, 32 + window_index),
            f"running AMB3R window {window_index + 1}/{len(windows)} on frames {window.start}-{window.end - 1}",
        )
        try:
            local = run_amb3r_window_once(image_tensor, pipeline, window_indices)
        except RuntimeError:
            raise
        local_registered = local["registered_indices"]
        if len(local_registered) < 3:
            expanded = expand_amb3r_window(window, image_count, read_amb3r_int_option(options, "fine_amb3r_window_overlap", 12, minimum=3, maximum=max(3, image_count - 1)))
            if expanded.indices != window_indices:
                local = run_amb3r_window_once(image_tensor, pipeline, expanded.indices)
                window_indices = expanded.indices
                local_registered = local["registered_indices"]
        if len(local_registered) < 3:
            failed_windows += 1
            raise FineFailure("AMB3R_WINDOW_INCOMPLETE", f"AMB3R window {window_index + 1}/{len(windows)} registered {len(local_registered)} images")

        local_to_global = {local_idx: window_indices[local_idx] for local_idx in range(len(window_indices))}
        local_registered_global = [local_to_global[index] for index in local_registered]
        window_registered_counts.append(len(local_registered_global))

        if not registered:
            transform = identity_similarity_transform()
        else:
            overlap_globals = sorted(global_index for global_index in local_registered_global if global_index in registered)
            if len(overlap_globals) < 3:
                alignment_failures += 1
                raise FineFailure("AMB3R_WINDOW_ALIGNMENT_FAILED", f"AMB3R window {window_index + 1}/{len(windows)} has only {len(overlap_globals)} registered overlap frames")
            local_overlap_indices = [window_indices.index(global_index) for global_index in overlap_globals]
            local_centers = local["poses_c2w"][local_overlap_indices, :3, 3]
            global_centers = global_poses[overlap_globals, :3, 3]
            transform = estimate_similarity_transform(local_centers, global_centers)

        transformed_poses = apply_similarity_to_poses(local["poses_c2w"], transform)
        for local_index in local_registered:
            global_index = local_to_global[local_index]
            if global_index in registered:
                continue
            global_poses[global_index] = transformed_poses[local_index]
            global_intrinsics[global_index] = local["intrinsics"][local_index]
            registered.add(global_index)

            pts = local["points_all"][local_index].reshape(-1, 3)
            conf = local["confidence_all"][local_index].reshape(-1)
            colors = ((processed[global_index].image + 1.0) * 0.5).reshape(-1, 3)
            point_chunks.append(apply_similarity_to_points(pts, transform))
            confidence_chunks.append(conf)
            color_chunks.append(colors)
        torch.cuda.empty_cache()

    if not point_chunks:
        raise FineFailure("AMB3R_EMPTY_POINT_CLOUD", "AMB3R windowed reconstruction produced no valid 3D points")

    return {
        "poses_c2w": global_poses,
        "intrinsics": global_intrinsics,
        "points": np.concatenate(point_chunks, axis=0),
        "confidence": np.concatenate(confidence_chunks, axis=0),
        "colors": np.concatenate(color_chunks, axis=0),
        "registered_indices": sorted(registered),
        "memory_device": str(pipeline.device),
        "metrics": {
            "amb3r_windowed": True,
            "amb3r_window_count": len(windows),
            "amb3r_window_size": max(window.end - window.start for window in windows),
            "amb3r_window_overlap": overlap,
            "amb3r_window_registered_images": window_registered_counts,
            "amb3r_window_failed_count": failed_windows,
            "amb3r_window_alignment_failed_count": alignment_failures,
        },
    }


def run_amb3r_window_once(image_tensor, pipeline, window_indices: list[int]) -> dict[str, Any]:
    import torch

    device = pipeline.device
    with torch.no_grad():
        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            memory = pipeline.run(image_tensor[:, window_indices].to(device))
    return local_scene_arrays_from_memory(memory, len(window_indices), width=image_tensor.shape[-1], height=image_tensor.shape[-2])


def expand_amb3r_window(window: Amb3rWindow, image_count: int, overlap: int) -> Amb3rWindow:
    return Amb3rWindow(max(0, window.start - overlap), min(image_count, window.end + overlap))


def scene_data_from_memory(
    memory,
    processed: list[ProcessedAmb3rImage],
    global_indices: list[int],
    *,
    width: int,
    height: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    local = local_scene_arrays_from_memory(memory, len(global_indices), width=width, height=height)
    registered = [global_indices[index] for index in local["registered_indices"]]
    poses = np.zeros((len(processed), 4, 4), dtype=np.float32)
    intrinsics = np.zeros((len(processed), 3, 3), dtype=np.float32)
    for local_index, global_index in enumerate(global_indices):
        poses[global_index] = local["poses_c2w"][local_index]
        intrinsics[global_index] = local["intrinsics"][local_index]
    return {
        "poses_c2w": poses,
        "intrinsics": intrinsics,
        "points": local["points_all"][local["registered_indices"]].reshape(-1, 3),
        "confidence": local["confidence_all"][local["registered_indices"]].reshape(-1),
        "colors": np.stack([((processed[index].image + 1.0) * 0.5) for index in registered], axis=0).reshape(-1, 3),
        "registered_indices": registered,
        "memory_device": str(getattr(memory, "device", "unknown")),
        "metrics": metrics,
    }


def local_scene_arrays_from_memory(memory, image_count: int, *, width: int, height: int) -> dict[str, Any]:
    poses_c2w = to_numpy(memory.poses)
    intrinsics = to_numpy(memory.intrinsics) if hasattr(memory, "intrinsics") else default_intrinsics(image_count, width, height)
    points_all = to_numpy(memory.pts)
    confidence_all = to_numpy(memory.conf)
    unmapped_frames = {int(index) for index in getattr(memory, "unmapped_frames", set())}
    return {
        "poses_c2w": poses_c2w,
        "intrinsics": intrinsics,
        "points_all": points_all,
        "confidence_all": confidence_all,
        "registered_indices": valid_registered_indices(poses_c2w, image_count, unmapped_frames),
    }


def identity_similarity_transform() -> tuple[float, np.ndarray, np.ndarray]:
    return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)


def estimate_similarity_transform(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = src_centered.T @ dst_centered / max(1, src.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = float(np.sum(singular_values) / max(variance, 1e-8))
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def apply_similarity_to_poses(poses: np.ndarray, transform: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = transform
    transformed = np.asarray(poses, dtype=np.float32).copy()
    transformed[:, :3, :3] = rotation @ transformed[:, :3, :3]
    transformed[:, :3, 3] = scale * (transformed[:, :3, 3] @ rotation.T) + translation
    return transformed


def apply_similarity_to_points(points: np.ndarray, transform: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = transform
    pts = np.asarray(points, dtype=np.float32)
    return scale * (pts @ rotation.T) + translation


def prepare_amb3r_images(
    files: list[Path],
    images_dir: Path,
    *,
    width: int = AMB3R_WIDTH,
    height: int = AMB3R_HEIGHT,
    target_aspect: float | None = None,
) -> list[ProcessedAmb3rImage]:
    processed = []
    processed_width = int(width)
    processed_height = int(height)
    target_aspect = float(target_aspect or (width / float(height)))
    for index, path in enumerate(files):
        with Image.open(path) as original:
            full = ImageOps.exif_transpose(original).convert("RGB")
        original_width, original_height = full.size
        full.save(images_dir / f"{index:06d}.jpg", format="JPEG", quality=94)

        crop_width = original_width
        crop_height = original_height
        if original_width / float(original_height) > target_aspect:
            crop_width = int(round(original_height * target_aspect))
        else:
            crop_height = int(round(original_width / target_aspect))
        crop_left = max(0, (original_width - crop_width) // 2)
        crop_top = max(0, (original_height - crop_height) // 2)
        cropped = full.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
        resized = cropped.resize((processed_width, processed_height), Image.Resampling.BICUBIC)
        image = np.asarray(resized, dtype=np.float32) / 255.0
        processed.append(
            ProcessedAmb3rImage(
                path=path,
                width=int(original_width),
                height=int(original_height),
                processed_width=processed_width,
                processed_height=processed_height,
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
        value = value.detach().float().cpu()
    return np.asarray(value, dtype=np.float32)


def default_intrinsics(count: int, width: int = AMB3R_WIDTH, height: int = AMB3R_HEIGHT) -> np.ndarray:
    intrinsics = np.tile(np.eye(3, dtype=np.float32), (count, 1, 1))
    focal = 0.9 * max(width, height)
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = width / 2.0
    intrinsics[:, 1, 2] = height / 2.0
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
