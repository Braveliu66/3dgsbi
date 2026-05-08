from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.fine.types import FineFailure
from app.preview.utils import image_files


Progress = Callable[[str, int, str], None]


@dataclass(slots=True)
class BlurAnalysis:
    mode: str
    mean_laplacian: float
    mean_gradient: float
    mean_fft_high_ratio: float
    rejected_images: int
    kept_images: int

    def metrics(self) -> dict[str, int | float | str]:
        return {
            "blur_mode": self.mode,
            "blur_mean_laplacian": round(self.mean_laplacian, 4),
            "blur_mean_gradient": round(self.mean_gradient, 4),
            "blur_mean_fft_high_ratio": round(self.mean_fft_high_ratio, 6),
            "blur_rejected_images": self.rejected_images,
            "blur_kept_images": self.kept_images,
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
class SceneBuildResult:
    scene_dir: Path
    backend: str
    image_count: int
    registered_images: int
    point_count: int | None
    metrics: dict[str, int | float | str | None]


def analyze_blur(input_dir: Path, *, reject_ratio: float = 0.15) -> BlurAnalysis:
    scores = score_blur_images(input_dir)
    return summarize_blur_scores(scores, reject_ratio=reject_ratio)


def prepare_mobile_images(
    input_dir: Path,
    output_dir: Path,
    *,
    reject_ratio: float = 0.15,
    min_images: int = 3,
) -> tuple[Path, BlurAnalysis]:
    scores = score_blur_images(input_dir)
    analysis = summarize_blur_scores(scores, reject_ratio=reject_ratio)
    reject_count = min(analysis.rejected_images, max(0, len(scores) - min_images))
    keep_count = max(min_images, len(scores) - reject_count)
    kept = sorted(sorted(scores, key=lambda item: item.quality, reverse=True)[:keep_count], key=lambda item: item.path.name)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(kept):
        with Image.open(item.path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.save(output_dir / f"{index:06d}.jpg", format="JPEG", quality=94)
    return output_dir, BlurAnalysis(
        mode=analysis.mode,
        mean_laplacian=analysis.mean_laplacian,
        mean_gradient=analysis.mean_gradient,
        mean_fft_high_ratio=analysis.mean_fft_high_ratio,
        rejected_images=len(scores) - len(kept),
        kept_images=len(kept),
    )


def score_blur_images(input_dir: Path) -> list[BlurScore]:
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


def summarize_blur_scores(scores: list[BlurScore], *, reject_ratio: float) -> BlurAnalysis:
    laps = np.array([item.laplacian for item in scores], dtype=np.float32)
    gradients = np.array([item.gradient for item in scores], dtype=np.float32)
    fft_ratios = np.array([item.fft_high_ratio for item in scores], dtype=np.float32)
    rejected = int(math.floor(len(scores) * max(0.0, min(0.45, reject_ratio))))

    mean_lap = float(laps.mean())
    mean_gradient = float(gradients.mean())
    mean_fft = float(fft_ratios.mean())
    if mean_lap >= 120.0 and mean_fft >= 0.08:
        mode = "sharp"
    elif mean_gradient >= 40.0 and mean_fft < 0.08:
        mode = "motion"
    elif mean_lap < 80.0:
        mode = "defocus"
    else:
        mode = "mixed"
    return BlurAnalysis(
        mode=mode,
        mean_laplacian=mean_lap,
        mean_gradient=mean_gradient,
        mean_fft_high_ratio=mean_fft,
        rejected_images=rejected,
        kept_images=max(0, len(scores) - rejected),
    )


def high_frequency_ratio(gray: np.ndarray) -> float:
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
    progress: Progress,
) -> SceneBuildResult:
    try:
        import pycolmap
    except Exception as exc:
        raise FineFailure("PYCOLMAP_UNAVAILABLE", f"pycolmap import failed: {exc}") from exc

    files = image_files(input_dir)
    if len(files) < 3:
        raise FineFailure("INSUFFICIENT_IMAGES", "COLMAP initialization requires at least 3 images")

    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    images_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files):
        shutil.copy2(path, images_dir / f"{index:06d}.jpg")

    started = time.monotonic()
    database_path = scene_dir / "database.db"
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
    progress("fine_colmap_matching", 30, "matching COLMAP features")
    pycolmap.match_exhaustive(database_path)

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
    if recon_path.name != "0":
        target = sparse_dir / "0"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(recon_path, target)
        recon_path = target

    reconstruction = pycolmap.Reconstruction(recon_path)
    normalize_colmap_cameras(reconstruction)
    reconstruction.write(recon_path)
    registered = len(reconstruction.images)
    if registered < min_model_size:
        raise FineFailure("COLMAP_RECONSTRUCTION_INCOMPLETE", f"COLMAP registered {registered}/{len(files)} images")

    point_count = len(reconstruction.points3D)
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
            "sfm_sparse_points": point_count,
            "colmap_sift_max_num_features": max_num_features,
            "colmap_max_image_size": max_image_size,
        },
    )


def select_best_colmap_model(sparse_dir: Path) -> Path | None:
    candidates = [path for path in sparse_dir.iterdir() if path.is_dir() and (path / "images.bin").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "images.bin").stat().st_size)


def normalize_colmap_cameras(reconstruction) -> None:
    import numpy as np

    for camera in reconstruction.cameras.values():
        model = str(camera.model)
        params = np.asarray(camera.params, dtype=np.float64)
        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"} and params.size >= 3:
            camera.model = "SIMPLE_PINHOLE"
            camera.params = params[:3]
        elif model == "PINHOLE" and params.size >= 4:
            continue
        elif params.size >= 4:
            camera.model = "PINHOLE"
            camera.params = params[:4]
        elif params.size >= 3:
            camera.model = "SIMPLE_PINHOLE"
            camera.params = params[:3]
