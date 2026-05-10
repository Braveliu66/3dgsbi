from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.fine.amb3r_sfm import AMB3R_COMMIT, amb3r_weight_path, build_amb3r_colmap_scene
from app.fine.local_3dgs.scene_quality import assess_sfm_scene_quality
from app.fine.local_3dgs.sparse_compensation import compensate_sparse_point_cloud
from app.fine.option_utils import read_float, read_int
from app.fine.preprocess import build_pycolmap_scene, prepare_mobile_images
from app.fine.types import FineContext, FineFailure, FineResult
from app.fine.video.pipeline import VIDEO_PIPELINE_NAME, run_video_artdeco_speed3r_pipeline
from app.preview.io.spz import convert_ply_to_spz
from app.preview.types import SOURCE_COMMITS
from app.preview.types import PreviewFailure
from app.preview.utils import image_files


PIPELINE_NAME = "mobilegs_lmrs"

SOURCE_COMMITS_FINE = {
    "AMB3R": AMB3R_COMMIT,
    "Spark": SOURCE_COMMITS["Spark"],
    "LM-RS": "cb40c7c06c2a60f8314ce095ad7b4513fbb33319",
    "LM-RS Rasterizer": "c2529d3bb13bc38271710785c015a89d9d623237",
    "FastGS": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
    "Deblurring-3DGS": "e63366b8581c0fde2fda0ab1aea99518da2e2f10",
}

def run_fine_pipeline(ctx: FineContext) -> FineResult:
    settings = get_settings()
    pipeline = normalize_fine_pipeline(ctx.pipeline)
    if pipeline == VIDEO_PIPELINE_NAME:
        return run_video_artdeco_speed3r_pipeline(ctx, settings=settings, lod_builder=build_lod_rad_if_available)
    if pipeline != PIPELINE_NAME:
        raise FineFailure("UNSUPPORTED_FINE_PIPELINE", f"Unsupported fine pipeline: {ctx.pipeline}")
    assert_runtime_ready()

    image_count = len(image_files(ctx.input_dir))
    if image_count < 3:
        raise FineFailure("INSUFFICIENT_IMAGES", "Fine reconstruction requires at least 3 images")

    iterations = read_int(ctx.options.get("fine_iterations"), settings.fine_iterations, minimum=500, maximum=60_000)
    explicit_lm_start_iter = ctx.options.get("fine_lm_start_iter")
    reject_ratio = read_float(ctx.options.get("fine_blur_reject_ratio"), 0.15, minimum=0.0, maximum=0.45)
    colmap_features = read_int(ctx.options.get("fine_sift_max_num_features"), 8192, minimum=1024, maximum=32768)
    colmap_max_size = read_int(ctx.options.get("fine_colmap_max_image_size"), 1600, minimum=512, maximum=3200)
    colmap_threads = read_int(ctx.options.get("fine_colmap_threads"), 8, minimum=1, maximum=32)

    ctx_progress(ctx, "fine_blur_analysis", 20, "analyzing image sharpness and filtering lowest quality frames")
    train_input_dir, blur = prepare_mobile_images(
        ctx.input_dir,
        ctx.work_dir / "fine_input",
        reject_ratio=reject_ratio,
        min_images=3,
    )
    blur_mode = str(ctx.options.get("fine_blur_mode") or blur.mode)
    lm_default = iterations
    lm_start_iter = read_int(explicit_lm_start_iter, lm_default, minimum=1, maximum=iterations)

    scene_dir = ctx.work_dir / "fine_scene"
    output_dir = ctx.work_dir / "fine_mobilegs"
    scene_result = build_scene(ctx, train_input_dir, scene_dir, colmap_features, colmap_max_size, colmap_threads)
    if scene_result.backend == "amb3r_sfm_colmap_no_exif":
        scene_result.metrics.update(assess_sfm_scene_quality(scene_result.scene_dir, prefix="amb3r").metrics)
    if read_bool(ctx.options.get("fine_edgs_enabled"), True):
        scene_result.metrics.update(
            {
                "sparse_compensation_enabled": False,
                "sparse_compensation_reason": "disabled_by_edgs",
            }
        )
    else:
        scene_result.metrics.update(compensate_sparse_point_cloud(scene_result.scene_dir, ctx.options).metrics)

    ctx_progress(ctx, "fine_mobilegs_train_start", 42, f"training MobileGS with {scene_result.backend} initialization")
    from app.fine.mobilegs_trainer import train_mobile_3dgs

    train_result = train_mobile_3dgs(
        scene_dir=scene_result.scene_dir,
        output_dir=output_dir,
        iterations=iterations,
        lm_start_iter=lm_start_iter,
        blur_mode=blur_mode,
        options=ctx.options,
        progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
    )

    ply_path = Path(train_result.ply_path)
    if not ply_path.exists() or ply_path.stat().st_size <= 0:
        raise FineFailure("ARTIFACT_NOT_FOUND", f"Fine runner did not create non-empty PLY: {ply_path}")

    ctx.final_ply.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ply_path, ctx.final_ply)

    ctx_progress(ctx, "final_spz_converting", 88, "converting final.ply to Spark-readable final_web.spz")
    try:
        splat_count = convert_ply_to_spz(ctx.final_ply, ctx.final_spz)
    except PreviewFailure as exc:
        raise FineFailure(exc.code, exc.message) from exc

    warnings = []
    lod_rad = build_lod_rad_if_available(ctx)
    if ctx.lod_rad and lod_rad is None:
        warnings.append("RAD LOD builder is not configured; final_lod.rad was not generated.")

    metrics = {
        "pipeline": PIPELINE_NAME,
        "algorithm": "mobilegs_amb3r_deblurmlp_lmrs",
        "source_version": ctx.source_version,
        "source_commits": SOURCE_COMMITS_FINE,
        "input_images": image_count,
        "training_images": len(image_files(train_input_dir)),
        "iterations": iterations,
        "lm_start_iter": lm_start_iter,
        "splat_count": splat_count,
        "final_ply_bytes": ctx.final_ply.stat().st_size,
        "final_spz_bytes": ctx.final_spz.stat().st_size,
        "lod_rad_bytes": lod_rad.stat().st_size if lod_rad else None,
        "warnings": warnings,
        **blur.metrics(),
        **scene_result.metrics,
        **train_result.metrics,
    }
    ctx.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    ctx.metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    ctx_progress(ctx, "fine_outputs_ready", 90, "validated final.ply and final_web.spz", metrics)
    return FineResult(
        final_ply=ctx.final_ply,
        final_spz=ctx.final_spz,
        metrics_json=ctx.metrics_json,
        lod_rad=lod_rad,
        splat_count=splat_count,
        source_commits=SOURCE_COMMITS_FINE,
        metrics=metrics,
    )


def build_scene(
    ctx: FineContext,
    input_dir: Path,
    scene_dir: Path,
    colmap_features: int,
    colmap_max_size: int,
    colmap_threads: int,
):
    sfm_backend = str(ctx.options.get("fine_sfm_backend") or "amb3r").strip().lower()
    if sfm_backend in {"amb3r", "amb3r_sfm", "litevggt"}:
        result = build_amb3r_colmap_scene(
            input_dir,
            scene_dir,
            checkpoint_path=amb3r_weight_path(ctx.model_cache_dir),
            keep_ratio=read_float(ctx.options.get("fine_amb3r_keep_ratio", ctx.options.get("fine_litevggt_keep_ratio")), 0.12, minimum=0.001, maximum=1.0),
            max_points=read_int(ctx.options.get("fine_amb3r_max_points", ctx.options.get("fine_litevggt_max_points")), 1_000_000, minimum=10_000, maximum=15_000_000),
            progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
            options=ctx.options,
        )
        if sfm_backend == "litevggt":
            result.metrics["sfm_backend_requested_alias"] = "litevggt_deprecated_maps_to_amb3r"
        return result
    if sfm_backend != "pycolmap":
        raise FineFailure("UNSUPPORTED_FINE_SFM_BACKEND", f"Unsupported fine SfM backend: {sfm_backend}")

    result = build_pycolmap_scene(
        input_dir,
        scene_dir,
        max_num_features=colmap_features,
        max_image_size=colmap_max_size,
        min_model_size=max(3, min(10, len(image_files(input_dir)))),
        num_threads=colmap_threads,
        progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
    )
    result.metrics["sfm_backend_requested"] = "pycolmap_diagnostic"
    return result


def normalize_fine_pipeline(value: str | None) -> str:
    normalized = (value or PIPELINE_NAME).strip().lower()
    aliases = {
        "fine_fused_quality": PIPELINE_NAME,
        "fused_quality": PIPELINE_NAME,
        "fused_quality_3dgs": PIPELINE_NAME,
        "mobilegs": PIPELINE_NAME,
        "mobilegs_lmrs": PIPELINE_NAME,
        "video_artdeco_litevggt": VIDEO_PIPELINE_NAME,
        "video_litevggt": VIDEO_PIPELINE_NAME,
        "artdeco_litevggt": VIDEO_PIPELINE_NAME,
        "video_artdeco_speed3r": VIDEO_PIPELINE_NAME,
        PIPELINE_NAME: PIPELINE_NAME,
        VIDEO_PIPELINE_NAME: VIDEO_PIPELINE_NAME,
    }
    return aliases.get(normalized, PIPELINE_NAME)


def assert_runtime_ready() -> None:
    try:
        import torch
    except Exception as exc:
        raise FineFailure("TORCH_UNAVAILABLE", f"PyTorch import failed: {exc}") from exc
    if not torch.cuda.is_available():
        raise FineFailure("GPU_RESOURCE_UNAVAILABLE", "CUDA GPU is required for fine reconstruction")
    missing = []
    for module_name in ("diff_gaussian_rasterization", "simple_knn", "fused_ssim"):
        try:
            __import__(module_name)
        except Exception as exc:
            missing.append(f"{module_name}: {exc}")
    if missing:
        raise FineFailure("FINE_RUNTIME_UNAVAILABLE", f"MobileGS fine runtime dependency missing: {'; '.join(missing)}")
    rasterizer = __import__("diff_gaussian_rasterization")
    if not hasattr(rasterizer, "GaussianRasterizer"):
        raise FineFailure("FINE_RUNTIME_UNAVAILABLE", "diff_gaussian_rasterization is missing GaussianRasterizer")


def deblur_mlp_enabled_by_default(blur_mode: str, options: dict[str, Any]) -> bool:
    value = str(options.get("fine_deblur_enabled", "auto")).lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return blur_mode != "sharp"
    return blur_mode in {"motion", "defocus", "mixed"}


def read_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return fallback


def build_lod_rad_if_available(ctx: FineContext) -> Path | None:
    if ctx.lod_rad is None:
        return None
    builder = os.getenv("SPARK_RAD_CLI")
    if not builder:
        return None
    command = ["node", builder, "convert", str(ctx.final_ply), str(ctx.lod_rad)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if completed.returncode != 0:
        raise FineFailure("RAD_CONVERSION_FAILED", (completed.stderr or completed.stdout or "").strip() or "RAD converter failed")
    if not ctx.lod_rad.exists() or ctx.lod_rad.stat().st_size <= 0:
        raise FineFailure("RAD_NOT_FOUND", f"RAD converter did not create non-empty output: {ctx.lod_rad}")
    return ctx.lod_rad


def ctx_progress(ctx: FineContext, stage: str, progress: int, message: str | None = None, metrics: dict[str, Any] | None = None) -> None:
    if ctx.progress:
        ctx.progress(stage, progress, message, metrics)
