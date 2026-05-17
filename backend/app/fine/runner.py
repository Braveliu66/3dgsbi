from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.fine.fastgs_defaults import (
    COLMAP_MATCHER,
    COLMAP_MAX_IMAGE_SIZE,
    COLMAP_MIN_REGISTERED_RATIO,
    COLMAP_SIFT_MAX_NUM_FEATURES,
    COLMAP_THREADS,
    DEFAULT_FINE_SCENE_PROFILE,
    FINE_IMAGE_MAX_SIDE,
    FINE_SCENE_PROFILE_MAX_SIDES,
    FASTGS_DEBLUR_BLURRED_VIEWS_ONLY,
    FASTGS_DEBLUR_ENABLED,
    FASTGS_DEBLUR_MODE,
    FASTGS_RESOLUTION,
)
from app.fine.option_utils import read_float, read_int
from app.fine.official_fastgs_big_trainer import train_official_fastgs_big
from app.fine.preprocess import build_pycolmap_scene, prepare_fine_images
from app.fine.types import FineContext, FineFailure, FineResult
from app.fine.viewer_meta import read_ply_xyz_bounds, write_far_noise_filtered_ply, write_final_viewer_meta_json, write_scaled_viewer_ply
from app.preview.io.spz import convert_ply_to_spz
from app.preview.types import PreviewFailure
from app.preview.utils import image_files


PIPELINE_NAME = "official_fastgs_big"

SOURCE_COMMITS_FINE = {
    "FastGS": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
    "diff_gaussian_rasterization_fastgs": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
    "simple-knn": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
    "fused-ssim": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
    "pycolmap": "3.12.6",
}

def run_fine_pipeline(ctx: FineContext) -> FineResult:
    settings = get_settings()
    pipeline = normalize_fine_pipeline(ctx.pipeline)
    print(
        "[fine-runner] start "
        f"task_id={ctx.task_id} project_id={ctx.project_id} requested_pipeline={ctx.pipeline} "
        f"normalized_pipeline={pipeline} input_dir={ctx.input_dir} work_dir={ctx.work_dir} "
        f"source_version={ctx.source_version} options={_format_for_log(ctx.options)}",
        flush=True,
    )
    if pipeline != PIPELINE_NAME:
        raise FineFailure("UNSUPPORTED_FINE_PIPELINE", f"Unsupported fine pipeline: {ctx.pipeline}")
    if read_bool(ctx.options.get("fine_edgs_enabled"), False):
        raise FineFailure("UNSUPPORTED_FINE_OPTION", "EDGS/RoMA dense initialization has been removed from this worker image")
    assert_runtime_ready()

    image_count = len(image_files(ctx.input_dir))
    print(
        "[fine-runner] input summary "
        f"input_dir={ctx.input_dir} image_count={image_count} first_images={_first_image_names(ctx.input_dir)}",
        flush=True,
    )
    if image_count < 8:
        raise FineFailure("INSUFFICIENT_IMAGES", "FastGS-Big fine reconstruction requires at least 8 images")

    iterations = read_int(ctx.options.get("fine_iterations"), settings.fine_iterations, minimum=5_000, maximum=60_000)
    fine_scene_profile = resolve_fine_scene_profile(ctx.options)
    profile_image_max_side = FINE_SCENE_PROFILE_MAX_SIDES[fine_scene_profile]
    default_image_max_side = min(profile_image_max_side, settings.fine_image_max_side, FINE_IMAGE_MAX_SIDE)
    image_max_side = read_int(ctx.options.get("fine_image_max_side"), default_image_max_side, minimum=512, maximum=FINE_IMAGE_MAX_SIDE)
    reject_ratio = read_float(ctx.options.get("fine_blur_reject_ratio"), 0.10, minimum=0.0, maximum=0.45)
    colmap_features = read_int(ctx.options.get("fine_sift_max_num_features"), COLMAP_SIFT_MAX_NUM_FEATURES, minimum=1024, maximum=32768)
    colmap_max_size = read_int(ctx.options.get("fine_colmap_max_image_size"), min(image_max_side, COLMAP_MAX_IMAGE_SIZE), minimum=512, maximum=FINE_IMAGE_MAX_SIDE)
    colmap_threads = read_int(ctx.options.get("fine_colmap_threads"), COLMAP_THREADS, minimum=1, maximum=32)
    colmap_matcher = str(ctx.options.get("fine_colmap_matcher") or COLMAP_MATCHER).strip().lower()
    min_registered_ratio = _optional_float(
        ctx.options,
        "fine_min_registered_ratio",
        fallback=COLMAP_MIN_REGISTERED_RATIO,
        minimum=0.30,
        maximum=0.95,
    )

    ctx_progress(ctx, "fine_blur_analysis", 20, "analyzing image sharpness and filtering lowest quality frames")
    print(
        "[fine-runner] blur analysis start "
        f"reject_ratio={reject_ratio} min_images=3 input_dir={ctx.input_dir}",
        flush=True,
    )
    train_input_dir, blur = prepare_fine_images(
        ctx.input_dir,
        ctx.work_dir / "fine_input",
        reject_ratio=reject_ratio,
        min_images=3,
    )
    print(
        "[fine-runner] blur analysis complete "
        f"train_input_dir={train_input_dir} blur_metrics={_format_for_log(blur.metrics())}",
        flush=True,
    )
    blur_mode = FASTGS_DEBLUR_MODE
    print(
        "[fine-runner] resolved training params "
        f"iterations={iterations} blur_mode={blur_mode} fine_scene_profile={fine_scene_profile} image_max_side={image_max_side} "
        f"sfm_backend={ctx.options.get('fine_sfm_backend') or 'pycolmap'} colmap_features={colmap_features} "
        f"colmap_max_size={colmap_max_size} colmap_threads={colmap_threads} colmap_matcher={colmap_matcher}",
        flush=True,
    )

    scene_dir = ctx.work_dir / "fine_scene"
    output_dir = ctx.work_dir / "fine_fastgs_big"
    scene_result = build_scene(
        ctx,
        train_input_dir,
        scene_dir,
        colmap_features,
        colmap_max_size,
        colmap_threads,
        matcher=colmap_matcher,
        min_registered_ratio=min_registered_ratio,
    )
    print(
        "[fine-runner] scene build complete "
        f"backend={scene_result.backend} scene_dir={scene_result.scene_dir} "
        f"image_count={scene_result.image_count} registered_images={scene_result.registered_images} "
        f"point_count={scene_result.point_count} metrics={_format_for_log(scene_result.metrics)}",
        flush=True,
    )

    ctx_progress(ctx, "fine_gaussian_train_start", 42, f"training official FastGS-Big with {scene_result.backend} initialization")
    blur_registry_path = output_dir / "blur_frame_registry.json"
    blur_registry_path.parent.mkdir(parents=True, exist_ok=True)
    blur_registry_path.write_text(
        json.dumps({"frames": blur.per_frame_blur}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    resolved_deblur_mode, deblur_mode_source = resolve_fine_deblur_mode(ctx.options, blur.mode, blur.per_frame_blur)
    train_options = {
        **ctx.options,
        "_fine_scene_backend": scene_result.backend,
        "fine_scene_profile": fine_scene_profile,
        "fine_image_max_side": image_max_side,
        "fine_deblur_mode": resolved_deblur_mode,
        "fine_deblur_mode_source": deblur_mode_source,
        "fine_deblur_blur_registry": str(blur_registry_path),
        "fine_deblur_blurred_views_only": ctx.options.get("fine_deblur_blurred_views_only") or FASTGS_DEBLUR_BLURRED_VIEWS_ONLY,
        "fine_train_resolution": ctx.options.get("fine_train_resolution") or min(image_max_side, FASTGS_RESOLUTION),
    }
    print(
        "[fine-runner] gaussian training start "
        f"scene_dir={scene_result.scene_dir} output_dir={output_dir} iterations={iterations} "
        f"blur_mode={blur_mode} blur_registry={blur_registry_path} train_options={_format_for_log(train_options)}",
        flush=True,
    )

    train_result = train_official_fastgs_big(
        scene_dir=scene_result.scene_dir,
        output_dir=output_dir,
        iterations=iterations,
        options=train_options,
        progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
    )
    print(
        "[fine-runner] gaussian training complete "
        f"ply_path={train_result.ply_path} metrics={_format_for_log(train_result.metrics)}",
        flush=True,
    )

    ply_path = Path(train_result.ply_path)
    if not ply_path.exists() or ply_path.stat().st_size <= 0:
        raise FineFailure("ARTIFACT_NOT_FOUND", f"Fine runner did not create non-empty PLY: {ply_path}")

    filtered_ply = ctx.work_dir / "final_filtered.ply"
    far_noise_metrics = write_far_noise_filtered_ply(ply_path, filtered_ply, profile=fine_scene_profile)

    ctx.final_ply.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(filtered_ply, ctx.final_ply)
    print(
        "[fine-runner] copied final ply "
        f"source={filtered_ply} target={ctx.final_ply} bytes={ctx.final_ply.stat().st_size} "
        f"far_noise_metrics={_format_for_log(far_noise_metrics)}",
        flush=True,
    )

    viewer_ply = ctx.work_dir / "final_viewer.ply"
    bounds = read_ply_xyz_bounds(ctx.final_ply)
    viewer_scale_multiplier = read_float(ctx.options.get("fine_viewer_scale_multiplier"), 0.55, minimum=0.05, maximum=1.0)
    default_max_scale = max(1e-6, min(float(bounds["radius"]) * 0.009, 1.0))
    viewer_scale_max = read_float(ctx.options.get("fine_viewer_scale_max"), default_max_scale, minimum=1e-6, maximum=10.0)
    viewer_scale_metrics = write_scaled_viewer_ply(
        ctx.final_ply,
        viewer_ply,
        scale_multiplier=viewer_scale_multiplier,
        max_scale=viewer_scale_max,
    )
    print(
        "[fine-runner] viewer ply ready "
        f"source={ctx.final_ply} viewer_ply={viewer_ply} metrics={_format_for_log(viewer_scale_metrics)}",
        flush=True,
    )

    viewer_meta_payload = None
    if ctx.viewer_meta_json:
        viewer_meta_payload = write_final_viewer_meta_json(
            ctx.viewer_meta_json,
            final_ply=ctx.final_ply,
            scene_dir=scene_result.scene_dir,
            preferred_image_names=first_clear_training_images(blur.per_frame_blur),
        )
        print(
            "[fine-runner] viewer meta json written "
            f"path={ctx.viewer_meta_json} bytes={ctx.viewer_meta_json.stat().st_size} "
            f"recommended_view={viewer_meta_payload.get('recommended_view')}",
            flush=True,
        )

    ctx_progress(ctx, "final_spz_converting", 88, "converting viewer-scaled final.ply to Spark-readable final_web.spz")
    try:
        splat_count = convert_ply_to_spz(viewer_ply, ctx.final_spz)
    except PreviewFailure as exc:
        raise FineFailure(exc.code, exc.message) from exc
    print(
        "[fine-runner] final SPZ conversion complete "
        f"final_ply={ctx.final_ply} viewer_ply={viewer_ply} final_ply_bytes={ctx.final_ply.stat().st_size} "
        f"final_spz={ctx.final_spz} final_spz_bytes={ctx.final_spz.stat().st_size if ctx.final_spz.exists() else None} "
        f"splat_count={splat_count}",
        flush=True,
    )

    warnings = []
    lod_rad = build_lod_rad_if_available(ctx)
    if ctx.lod_rad and lod_rad is None:
        warnings.append("RAD LOD builder is not configured; final_lod.rad was not generated.")

    effective_algorithms = [scene_result.backend, "official_fastgs_big", "diff_gaussian_rasterization_fastgs"]
    effective_deblur_mode = str(train_options.get("fine_deblur_mode") or FASTGS_DEBLUR_MODE)
    if deblur_mlp_enabled_by_default(effective_deblur_mode, train_options):
        effective_algorithms.append("Deblurring-3DGS_GTnet_fastgs")
    metrics = {
        "pipeline": PIPELINE_NAME,
        "algorithm": f"{scene_result.backend}_official_fastgs_big",
        "requested_algorithms": [scene_result.backend, "FastGS-Big"],
        "effective_algorithms": effective_algorithms,
        "source_version": ctx.source_version,
        "source_commits": SOURCE_COMMITS_FINE,
        "artifact_converter": "Spark SPZ",
        "input_images": image_count,
        "training_images": len(image_files(train_input_dir)),
        "fine_scene_profile": fine_scene_profile,
        "fine_image_max_side": image_max_side,
        "iterations": iterations,
        "deblur_mode_source": deblur_mode_source,
        "splat_count": splat_count,
        "final_ply_bytes": ctx.final_ply.stat().st_size,
        "final_viewer_ply_bytes": viewer_ply.stat().st_size,
        "final_spz_bytes": ctx.final_spz.stat().st_size,
        "viewer_meta_json_bytes": ctx.viewer_meta_json.stat().st_size if ctx.viewer_meta_json and ctx.viewer_meta_json.exists() else None,
        "lod_rad_bytes": lod_rad.stat().st_size if lod_rad else None,
        "warnings": warnings,
        **far_noise_metrics,
        **viewer_scale_metrics,
        **blur.metrics(),
        **scene_result.metrics,
        **train_result.metrics,
    }
    ctx.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    ctx.metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[fine-runner] metrics json written "
        f"path={ctx.metrics_json} bytes={ctx.metrics_json.stat().st_size} metrics={_format_for_log(metrics)}",
        flush=True,
    )
    ctx_progress(ctx, "fine_outputs_ready", 90, "validated final.ply and final_web.spz", metrics)
    return FineResult(
        final_ply=ctx.final_ply,
        final_spz=ctx.final_spz,
        metrics_json=ctx.metrics_json,
        viewer_meta_json=ctx.viewer_meta_json if ctx.viewer_meta_json and ctx.viewer_meta_json.exists() else None,
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
    matcher: str = "auto",
    min_registered_ratio: float | None = None,
):
    sfm_backend = str(ctx.options.get("fine_sfm_backend") or "pycolmap").strip().lower()
    print(
        "[fine-runner] scene build start "
        f"requested_sfm_backend={sfm_backend} input_dir={input_dir} scene_dir={scene_dir} "
        f"colmap_features={colmap_features} colmap_max_size={colmap_max_size} colmap_threads={colmap_threads} matcher={matcher}",
        flush=True,
    )
    if sfm_backend not in {"pycolmap", "colmap"}:
        raise FineFailure("UNSUPPORTED_FINE_SFM_BACKEND", f"Unsupported fine SfM backend: {sfm_backend}")

    result = build_pycolmap_scene(
        input_dir,
        scene_dir,
        max_num_features=colmap_features,
        max_image_size=colmap_max_size,
        min_model_size=max(3, min(10, len(image_files(input_dir)))),
        num_threads=colmap_threads,
        matcher=matcher,
        min_registered_ratio=min_registered_ratio,
        progress=lambda stage, progress, message: ctx_progress(ctx, stage, progress, message),
    )
    if sfm_backend == "colmap":
        result.metrics["sfm_backend_requested_alias"] = "colmap_maps_to_pycolmap"
    return result


def normalize_fine_pipeline(value: str | None) -> str:
    normalized = (value or PIPELINE_NAME).strip().lower()
    aliases = {
        PIPELINE_NAME: PIPELINE_NAME,
    }
    return aliases.get(normalized, normalized)


def resolve_fine_scene_profile(options: dict[str, Any]) -> str:
    profile = str(
        options.get("fine_scene_profile") or options.get("preview_scene_profile") or DEFAULT_FINE_SCENE_PROFILE
    ).strip().lower()
    if profile not in FINE_SCENE_PROFILE_MAX_SIDES:
        return DEFAULT_FINE_SCENE_PROFILE
    return profile


def assert_runtime_ready() -> None:
    print("[fine-runner] checking CUDA and official FastGS-Big modules", flush=True)
    try:
        import torch
    except Exception as exc:
        raise FineFailure("TORCH_UNAVAILABLE", f"PyTorch import failed: {exc}") from exc
    if not torch.cuda.is_available():
        raise FineFailure("GPU_RESOURCE_UNAVAILABLE", "CUDA GPU is required for fine reconstruction")
    print(
        "[fine-runner] torch CUDA ready "
        f"torch_version={getattr(torch, '__version__', 'unknown')} cuda_version={getattr(torch.version, 'cuda', None)} "
        f"device_count={torch.cuda.device_count()} current_device={torch.cuda.current_device() if torch.cuda.device_count() else None} "
        f"device_name={torch.cuda.get_device_name(0) if torch.cuda.device_count() else None}",
        flush=True,
    )
    missing = []
    for module_name in ("diff_gaussian_rasterization_fastgs", "simple_knn", "fused_ssim"):
        try:
            __import__(module_name)
            print(f"[fine-runner] module import ok module={module_name}", flush=True)
        except Exception as exc:
            missing.append(f"{module_name}: {exc}")
    if missing:
        raise FineFailure("FINE_RUNTIME_UNAVAILABLE", f"Fine reconstruction runtime dependency missing: {'; '.join(missing)}")
    fastgs_rasterizer = __import__("diff_gaussian_rasterization_fastgs")
    if not hasattr(fastgs_rasterizer, "GaussianRasterizer"):
        raise FineFailure("FINE_RUNTIME_UNAVAILABLE", "diff_gaussian_rasterization_fastgs is missing GaussianRasterizer")


def deblur_mlp_enabled_by_default(blur_mode: str, options: dict[str, Any]) -> bool:
    value = str(options.get("fine_deblur_enabled", FASTGS_DEBLUR_ENABLED)).lower()
    if value in {"0", "false", "no", "off"}:
        return False
    return str(blur_mode or FASTGS_DEBLUR_MODE).strip().lower() != "sharp"


def resolve_fine_deblur_mode(
    options: dict[str, Any],
    _detected_mode: str,
    _per_frame_blur: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    explicit = str(options.get("fine_deblur_mode") or "").strip().lower()
    if explicit in {"sharp", "defocus", "motion", "mixed"}:
        return explicit, "override"
    return FASTGS_DEBLUR_MODE, "default_mixed"


def first_clear_training_images(per_frame_blur: dict[str, dict[str, Any]]) -> list[str]:
    clear: list[str] = []
    for item in per_frame_blur.values():
        if item.get("rejected") or item.get("blurred"):
            continue
        training_image = item.get("training_image")
        if isinstance(training_image, str) and training_image:
            clear.append(training_image)
    return clear


def read_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return fallback


def _optional_float(options: dict[str, Any], key: str, *, fallback: float | None = None, minimum: float, maximum: float) -> float | None:
    if key not in options or options.get(key) in {None, ""}:
        return fallback
    return read_float(options.get(key), fallback if fallback is not None else minimum, minimum=minimum, maximum=maximum)


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


def _first_image_names(input_dir: Path, limit: int = 8) -> str:
    files = image_files(input_dir)
    names = [path.name for path in files[:limit]]
    suffix = "" if len(files) <= limit else f", ... +{len(files) - limit}"
    return "[" + ", ".join(names) + suffix + "]"


def _format_for_log(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)
