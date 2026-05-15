from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.fine.types import FineFailure
from app.preview.utils import image_files

try:
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover - depends on worker image
    Image = None
    ImageOps = None
try:
    import cv2
except Exception:  # pragma: no cover - depends on worker image
    cv2 = None
try:
    import numpy as np
except Exception:  # pragma: no cover - depends on worker image
    np = None


Progress = Callable[[str, int, str], None]


@dataclass(slots=True)
class BlurAnalysis:
    mode: str
    mean_laplacian: float
    mean_gradient: float
    mean_fft_high_ratio: float
    rejected_images: int
    kept_images: int
    blurred_images: int = 0
    training_blur_frames: int = 0
    rejected_blur_frames: int = 0
    deblur_trigger_reason: str = "none"
    per_frame_blur: dict[str, dict[str, str | bool | float | None]] = field(default_factory=dict)

    def metrics(self) -> dict[str, int | float | str | dict[str, dict[str, str | bool | float | None]]]:
        return {
            "blur_mode": self.mode,
            "blur_mean_laplacian": round(self.mean_laplacian, 4),
            "blur_mean_gradient": round(self.mean_gradient, 4),
            "blur_mean_fft_high_ratio": round(self.mean_fft_high_ratio, 6),
            "blur_rejected_images": self.rejected_images,
            "blur_kept_images": self.kept_images,
            "blurred_images": self.blurred_images,
            "training_blur_frames": self.training_blur_frames,
            "rejected_blur_frames": self.rejected_blur_frames,
            "deblur_trigger_reason": self.deblur_trigger_reason,
            "blur_frame_registry": self.per_frame_blur,
        }


@dataclass(slots=True)
class BlurScore:
    path: Path
    laplacian: float
    gradient: float
    fft_high_ratio: float

    @property
    def quality(self) -> float:
        return self.laplacian + 0.1 * self.gradient + 600.0 * self.fft_high_ratio


@dataclass(slots=True)
class BlurClassification:
    blurred: bool
    kind: str


@dataclass(slots=True)
class SceneBuildResult:
    scene_dir: Path
    backend: str
    image_count: int
    registered_images: int
    point_count: int | None
    metrics: dict[str, Any]


def analyze_blur(input_dir: Path, *, reject_ratio: float = 0.15) -> BlurAnalysis:
    scores = score_blur_images(input_dir)
    return summarize_blur_scores(scores, reject_ratio=reject_ratio)


def prepare_fine_images(
    input_dir: Path,
    output_dir: Path,
    *,
    reject_ratio: float = 0.15,
    min_images: int = 3,
) -> tuple[Path, BlurAnalysis]:
    if Image is None or ImageOps is None:
        raise FineFailure("IMAGE_ANALYSIS_UNAVAILABLE", "Pillow is required for fine image normalization")
    scores = score_blur_images(input_dir)
    analysis = summarize_blur_scores(scores, reject_ratio=reject_ratio, min_images=min_images)
    classifications = classify_blur_scores(scores)
    median_quality = _median([item.quality for item in scores])
    reject_count = min(analysis.rejected_images, max(0, len(scores) - min_images))
    reject_candidates = [
        item
        for item in sorted(scores, key=lambda score: score.quality)
        if should_reject_for_training(item, classifications[item.path], median_quality=median_quality)
    ]
    rejected_paths = {item.path for item in reject_candidates[:reject_count]}
    kept = sorted([item for item in scores if item.path not in rejected_paths], key=lambda item: item.path.name)
    per_frame_blur: dict[str, dict[str, str | bool | float | None]] = {}
    kept_paths: set[Path] = set()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(kept):
        normalized_name = f"{index:06d}.jpg"
        kept_paths.add(item.path)
        classification = classifications[item.path]
        per_frame_blur[normalized_name] = {
            "source_image": item.path.name,
            "training_image": normalized_name,
            "training_stem": Path(normalized_name).stem,
            "rejected": False,
            "blurred": classification.blurred,
            "kind": classification.kind,
            "quality": round(item.quality, 6),
            "laplacian": round(item.laplacian, 6),
            "gradient": round(item.gradient, 6),
            "fft_high_ratio": round(item.fft_high_ratio, 8),
        }
        with Image.open(item.path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.save(output_dir / normalized_name, format="JPEG", quality=94)
    for item in sorted(scores, key=lambda score: score.path.name):
        if item.path in kept_paths:
            continue
        classification = classifications[item.path]
        per_frame_blur[f"rejected:{item.path.name}"] = {
            "source_image": item.path.name,
            "training_image": None,
            "training_stem": None,
            "rejected": True,
            "blurred": classification.blurred,
            "kind": classification.kind,
            "quality": round(item.quality, 6),
            "laplacian": round(item.laplacian, 6),
            "gradient": round(item.gradient, 6),
            "fft_high_ratio": round(item.fft_high_ratio, 8),
        }
    kept_blur = [classifications[item.path] for item in kept if classifications[item.path].blurred]
    all_blur = [classification for classification in classifications.values() if classification.blurred]
    mode = blur_mode_from_classifications(all_blur)
    training_blur_frames = len(kept_blur)
    return output_dir, BlurAnalysis(
        mode=mode,
        mean_laplacian=analysis.mean_laplacian,
        mean_gradient=analysis.mean_gradient,
        mean_fft_high_ratio=analysis.mean_fft_high_ratio,
        rejected_images=len(scores) - len(kept),
        kept_images=len(kept),
        blurred_images=analysis.blurred_images,
        training_blur_frames=training_blur_frames,
        rejected_blur_frames=analysis.rejected_blur_frames,
        deblur_trigger_reason="none" if not all_blur else f"blur_detected:{mode}",
        per_frame_blur=per_frame_blur,
    )


def score_blur_images(input_dir: Path) -> list[BlurScore]:
    if Image is None or ImageOps is None or cv2 is None or np is None:
        raise FineFailure("IMAGE_ANALYSIS_UNAVAILABLE", "Pillow, OpenCV, and NumPy are required for fine image blur analysis")
    files = image_files(input_dir)
    if not files:
        raise FineFailure("IMAGE_INPUT_NOT_FOUND", f"no supported image files found in {input_dir}")
    scores = []
    for path in files:
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("L")
            gray = np.asarray(image, dtype=np.uint8)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = float(np.sqrt(grad_x * grad_x + grad_y * grad_y).max())
        scores.append(BlurScore(path=path, laplacian=lap, gradient=gradient, fft_high_ratio=high_frequency_ratio(gray)))
    return scores


def summarize_blur_scores(scores: list[BlurScore], *, reject_ratio: float, min_images: int = 0) -> BlurAnalysis:
    laps = [item.laplacian for item in scores]
    gradients = [item.gradient for item in scores]
    fft_ratios = [item.fft_high_ratio for item in scores]
    mean_lap = sum(laps) / len(laps)
    mean_gradient = sum(gradients) / len(gradients)
    mean_fft = sum(fft_ratios) / len(fft_ratios)
    median_lap = _median(laps)
    median_fft = _median(fft_ratios)
    classifications = classify_blur_scores(scores, median_laplacian=median_lap, median_fft_high_ratio=median_fft)
    max_rejected = int(math.floor(len(scores) * max(0.0, min(0.45, reject_ratio))))
    max_rejected = min(max_rejected, max(0, len(scores) - max(0, min_images)))
    median_quality = _median([item.quality for item in scores])
    reject_candidates = [
        item
        for item in sorted(scores, key=lambda score: score.quality)
        if should_reject_for_training(item, classifications[item.path], median_quality=median_quality)
    ]
    rejected_scores = reject_candidates[:max_rejected]
    rejected_paths = {item.path for item in rejected_scores}
    kept_scores = [item for item in scores if item.path not in rejected_paths]
    rejected = len(rejected_scores)
    keep_count = len(kept_scores)
    kept_blur = [classifications[item.path] for item in kept_scores if classifications[item.path].blurred]
    rejected_blur = [classifications[item.path] for item in rejected_scores if classifications[item.path].blurred]
    all_blur = [classification for classification in classifications.values() if classification.blurred]
    mode = blur_mode_from_classifications(all_blur)
    training_blur_frames = len(kept_blur)
    trigger_reason = "none" if not all_blur else f"blur_detected:{mode}"
    return BlurAnalysis(
        mode=mode,
        mean_laplacian=mean_lap,
        mean_gradient=mean_gradient,
        mean_fft_high_ratio=mean_fft,
        rejected_images=rejected,
        kept_images=keep_count,
        blurred_images=sum(1 for item in classifications.values() if item.blurred),
        training_blur_frames=training_blur_frames,
        rejected_blur_frames=len(rejected_blur),
        deblur_trigger_reason=trigger_reason,
        per_frame_blur={
            item.path.name: {
                "source_image": item.path.name,
                "blurred": classifications[item.path].blurred,
                "kind": classifications[item.path].kind,
                "quality": round(item.quality, 6),
            }
            for item in scores
        },
    )


def classify_blur_scores(
    scores: list[BlurScore],
    *,
    median_laplacian: float | None = None,
    median_fft_high_ratio: float | None = None,
) -> dict[Path, BlurClassification]:
    if median_laplacian is None:
        median_laplacian = _median([item.laplacian for item in scores])
    if median_fft_high_ratio is None:
        median_fft_high_ratio = _median([item.fft_high_ratio for item in scores])
    return {
        item.path: classify_blur_score(
            item,
            median_laplacian=median_laplacian,
            median_fft_high_ratio=median_fft_high_ratio,
        )
        for item in scores
    }


def classify_blur_score(score: BlurScore, *, median_laplacian: float, median_fft_high_ratio: float) -> BlurClassification:
    defocus = score.laplacian < 80.0
    motion = score.gradient >= 40.0 and score.fft_high_ratio < 0.08
    relative_lap_blur = median_laplacian > 1e-6 and score.laplacian < median_laplacian * 0.55
    relative_fft_blur = median_fft_high_ratio > 1e-6 and score.fft_high_ratio < median_fft_high_ratio * 0.75
    moderate_defocus = (
        median_laplacian > 1e-6
        and median_fft_high_ratio > 1e-6
        and score.laplacian < max(180.0, median_laplacian * 0.65)
        and score.fft_high_ratio < median_fft_high_ratio * 0.90
    )
    relative = relative_lap_blur and relative_fft_blur and score.gradient >= 20.0
    if motion and defocus:
        return BlurClassification(True, "mixed")
    if motion:
        return BlurClassification(True, "motion")
    if defocus or moderate_defocus:
        return BlurClassification(True, "defocus")
    if relative:
        return BlurClassification(True, "mixed")
    return BlurClassification(False, "sharp")


def should_reject_for_training(score: BlurScore, classification: BlurClassification, *, median_quality: float) -> bool:
    if not classification.blurred:
        return False
    absolute_extreme = score.laplacian < 20.0 and score.gradient < 25.0 and score.fft_high_ratio < 0.03
    relative_extreme = median_quality > 1e-6 and score.quality < median_quality * 0.30
    return absolute_extreme or relative_extreme


def blur_mode_from_classifications(classifications: list[BlurClassification]) -> str:
    kinds = {item.kind for item in classifications if item.blurred}
    if not kinds:
        return "sharp"
    if len(kinds) > 1 or "mixed" in kinds:
        return "mixed"
    return next(iter(kinds))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)


def high_frequency_ratio(gray: np.ndarray) -> float:
    if cv2 is None or np is None:
        raise FineFailure("IMAGE_ANALYSIS_UNAVAILABLE", "OpenCV and NumPy are required for fine image blur analysis")
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA) if max(gray.shape) > 256 else gray
    spectrum = np.fft.fftshift(np.fft.fft2(small.astype(np.float32)))
    magnitude = np.abs(spectrum)
    height, width = magnitude.shape
    radius = min(height, width) * 0.18
    y, x = np.ogrid[:height, :width]
    mask = (x - width / 2.0) ** 2 + (y - height / 2.0) ** 2 >= radius * radius
    total = float(magnitude.sum())
    if total <= 1e-6:
        return 0.0
    return float(magnitude[mask].sum() / total)


def build_pycolmap_scene(
    input_dir: Path,
    scene_dir: Path,
    *,
    max_num_features: int,
    max_image_size: int,
    min_model_size: int,
    num_threads: int,
    matcher: str = "auto",
    min_registered_ratio: float | None = None,
    progress: Progress,
) -> SceneBuildResult:
    try:
        import pycolmap
    except Exception as exc:
        raise FineFailure("PYCOLMAP_UNAVAILABLE", f"pycolmap import failed: {exc}") from exc

    files = image_files(input_dir)
    if len(files) < 3:
        raise FineFailure("INSUFFICIENT_IMAGES", "COLMAP initialization requires at least 3 images")
    matcher = matcher.strip().lower()
    if matcher == "auto":
        matcher = "exhaustive" if len(files) <= 250 else "sequential"
    if matcher not in {"exhaustive", "sequential"}:
        raise FineFailure("UNSUPPORTED_COLMAP_MATCHER", f"Unsupported COLMAP matcher: {matcher}")

    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    distorted_dir = scene_dir / "distorted"
    images_dir = distorted_dir / "images"
    sparse_dir = distorted_dir / "sparse"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files):
        shutil.copy2(path, images_dir / f"{index:06d}.jpg")

    started = time.monotonic()
    database_path = distorted_dir / "database.db"
    progress("fine_colmap_features", 24, f"extracting COLMAP features from {len(files)} images")
    pycolmap.extract_features(
        database_path,
        images_dir,
        sift_options={
            "max_num_features": max_num_features,
            "max_image_size": max_image_size,
            "num_threads": num_threads,
        },
    )
    progress("fine_colmap_matching", 30, f"matching COLMAP features with {matcher} matcher")
    if matcher == "exhaustive":
        pycolmap.match_exhaustive(database_path)
    else:
        pycolmap.match_sequential(database_path)

    options = pycolmap.IncrementalPipelineOptions()
    options.min_num_matches = 15
    options.multiple_models = True
    options.max_num_models = 50
    options.max_model_overlap = 20
    options.min_model_size = min_model_size
    options.extract_colors = True
    options.num_threads = num_threads
    options.mapper.init_min_num_inliers = 30
    options.mapper.init_max_error = 8.0
    options.mapper.init_min_tri_angle = 5.0

    progress("fine_colmap_mapping", 36, "running COLMAP incremental mapping")
    pycolmap.incremental_mapping(database_path=database_path, image_path=images_dir, output_path=sparse_dir, options=options)
    recon_path = select_best_colmap_model(sparse_dir)
    if recon_path is None:
        raise FineFailure("COLMAP_RECONSTRUCTION_FAILED", "COLMAP did not produce a sparse reconstruction")
    reconstruction = pycolmap.Reconstruction(recon_path)
    registered = len(reconstruction.images)
    registered_ratio = registered / len(files)
    threshold = _resolve_min_registered_ratio(len(files), min_registered_ratio)
    if registered < min_model_size or registered_ratio < threshold:
        raise FineFailure(
            "COLMAP_RECONSTRUCTION_INCOMPLETE",
            f"COLMAP registered {registered}/{len(files)} images ({registered_ratio:.1%}), below threshold {threshold:.1%}",
        )

    point_count = len(reconstruction.points3D)
    progress("fine_colmap_undistort", 40, "undistorting COLMAP images for FastGS")
    pycolmap.undistort_images(
        output_path=str(scene_dir),
        input_path=str(recon_path),
        image_path=str(images_dir),
        output_type="COLMAP",
    )
    fastgs_sparse_dir = ensure_fastgs_sparse_zero(scene_dir / "sparse")
    fastgs_reconstruction = pycolmap.Reconstruction(fastgs_sparse_dir)
    validate_fastgs_colmap_scene(fastgs_reconstruction)
    elapsed = round(time.monotonic() - started, 3)
    return SceneBuildResult(
        scene_dir=scene_dir,
        backend="pycolmap",
        image_count=len(files),
        registered_images=registered,
        point_count=point_count,
        metrics={
            "sfm_backend": "pycolmap",
            "sfm_elapsed_seconds": elapsed,
            "sfm_registered_images": registered,
            "sfm_registered_ratio": registered_ratio,
            "sfm_min_registered_ratio": threshold,
            "sfm_sparse_points": point_count,
            "sfm_undistorted": True,
            "colmap_matcher": matcher,
            "colmap_sift_max_num_features": max_num_features,
            "colmap_max_image_size": max_image_size,
        },
    )


def _resolve_min_registered_ratio(image_count: int, override: float | None) -> float:
    if override is not None:
        return max(0.30, min(0.95, float(override)))
    if image_count <= 250:
        return 0.70
    if image_count <= 1000:
        return 0.65
    return 0.60


def select_best_colmap_model(sparse_dir: Path) -> Path | None:
    candidates = [path for path in sparse_dir.iterdir() if path.is_dir() and (path / "images.bin").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "images.bin").stat().st_size)


def ensure_fastgs_sparse_zero(sparse_dir: Path) -> Path:
    model_dir = sparse_dir / "0"
    if (model_dir / "images.bin").exists():
        return model_dir
    if (sparse_dir / "images.bin").exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        for path in sparse_dir.iterdir():
            if path.is_file():
                shutil.move(str(path), model_dir / path.name)
        return model_dir
    raise FineFailure("COLMAP_UNDISTORT_FAILED", f"COLMAP undistortion did not create a sparse model in {sparse_dir}")


def validate_fastgs_colmap_scene(reconstruction) -> None:
    cameras = getattr(reconstruction, "cameras", None) or {}
    for camera in cameras.values():
        model = _camera_model_name(getattr(camera, "model", ""))
        params = [float(value) for value in getattr(camera, "params", [])]
        width = float(getattr(camera, "width", 0) or 0)
        height = float(getattr(camera, "height", 0) or 0)
        if model == "SIMPLE_PINHOLE" and len(params) >= 3:
            fov_params = params[:1]
            cx, cy = params[1], params[2]
        elif model == "PINHOLE" and len(params) >= 4:
            fov_params = params[:2]
            cx, cy = params[2], params[3]
        else:
            raise FineFailure(
                "COLMAP_CAMERA_MODEL_UNSUPPORTED",
                f"FastGS requires undistorted SIMPLE_PINHOLE/PINHOLE cameras, got {model or camera.model}",
            )
        if width <= 0 or height <= 0 or any(value <= 0 for value in fov_params):
            raise FineFailure("COLMAP_CAMERA_INVALID", f"Invalid FastGS camera intrinsics for model {model}")
        if not _principal_point_is_plausible(cx, cy, width, height):
            raise FineFailure(
                "COLMAP_CAMERA_INVALID",
                f"Invalid FastGS principal point for {model}: cx={cx:.6g}, cy={cy:.6g}, size={width:.0f}x{height:.0f}",
            )


def _camera_model_name(model: object) -> str:
    return str(model).split(".")[-1].upper()


def _principal_point_is_plausible(cx: float, cy: float, width: float, height: float) -> bool:
    return width * 0.05 <= cx <= width * 0.95 and height * 0.05 <= cy <= height * 0.95
