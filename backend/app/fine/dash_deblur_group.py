from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.fine.option_utils import read_float, read_int
from app.fine.types import FineFailure
from app.preview.io.spz import convert_ply_to_spz
from app.preview.types import PreviewFailure


Progress = Callable[[str, int, str], None]
BLUR_CODE_DIM = 8


@dataclass(frozen=True, slots=True)
class DashDeblurGroupPaths:
    repo_dir: Path
    train_py: Path
    python: str
    trainer_flavor: str = "dash_deblur_group"


@dataclass(frozen=True, slots=True)
class DashDeblurGroupResult:
    final_ply: Path
    final_spz: Path | None
    output_dir: Path
    config_path: Path
    splat_count: int
    source_commit: str
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EffectiveDeblurMode:
    requested: str
    effective: str
    confidence: str
    reason: str
    motion_frames: int
    defocus_frames: int
    mixed_frames: int
    sharp_frames: int


INDOOR_MOTION = {
    "iterations": 30_000,
    "resolution": -1,
    "white_background": False,
    "eval": True,
    "deblur": 1,
    "use_pos": 1,
    "num_moments": 4,
    "hidden": 3,
    "width": 64,
    "gtnet_lr": 0.001,
    "position_lr_final": 0.000016,
    "percent_dense": 0.01,
    "lambda_dssim": 0.2,
    "lambda_s": 0.01,
    "lambda_p": 0.01,
    "max_clamp": 1.10,
    "per_image_blur": 0,
    "blur_label_path": "",
    "blur_code_dim": BLUR_CODE_DIM,
    "lambda_code": 0.0001,
    "lambda_delta": 0.001,
    "sharp_weight": 1.0,
    "motion_weight": 1.0,
    "defocus_weight": 1.0,
    "pre_deblur_warmup_enable": True,
    "pre_deblur_warmup_iters": 500,
    "pre_deblur_warmup_densify": True,
    "pre_deblur_warmup_densify_from_iter": 100,
    "luminance_enable": True,
    "luminance_start_iter": 1000,
    "luminance_lr": 0.001,
    "luminance_lambda_gain": 0.005,
    "luminance_lambda_bias": 0.01,
    "luminance_mode": "exposure_gain_bias",
    "luminance_per_channel": False,
    "luminance_matrix_enable": False,
    "luminance_curve_enable": False,
    "gdags_stats_enable": True,
    "gdags_enable": False,
    "gdags_start_iter": 3000,
    "gdags_probe_interval": 100,
    "gdags_probe_use_autograd_grad": True,
    "gdags_clone_enable": False,
    "gdags_split_enable": False,
    "gdags_prune_enable": False,
    "gdags_newborn_protect_iters": 1000,
    "pc_name": "points3D",
    "renderer_backend": "original",
    "renderer_backend_deblur": "original",
    "densify_from_iter": 500,
    "densify_until_iter": 3000,
    "densification_interval": 100,
    "densify_grad_threshold": 0.0005,
    "densify_prune_threshold": 0.01,
    "densify_with_depth": 1,
    "prune_range": 3,
    "pts_iter": 999999,
    "pts_rate": 0.0,
    "pts_dist": 2,
    "pts_N_intpl": 4,
    "pts_N_pts": 0,
    "pts_add_bound": 10,
}

INDOOR_MIX = {
    **INDOOR_MOTION,
}

INDOOR_DEFOCUS = {
    **INDOOR_MOTION,
    "use_pos": 0,
    "densify_grad_threshold": 0.0002,
    "densify_prune_threshold": 0.005,
}

OUTDOOR_MOTION = {
    **INDOOR_MOTION,
}

OUTDOOR_MIX = {
    **OUTDOOR_MOTION,
}

OUTDOOR_DEFOCUS = {
    **OUTDOOR_MOTION,
    "use_pos": 0,
    "densify_grad_threshold": 0.0002,
    "densify_prune_threshold": 0.005,
}

CONFIG_PRESETS = {
    ("indoor", "motion"): INDOOR_MOTION,
    ("indoor", "mix"): INDOOR_MIX,
    ("indoor", "defocus"): INDOOR_DEFOCUS,
    ("outdoor", "motion"): OUTDOOR_MOTION,
    ("outdoor", "mix"): OUTDOOR_MIX,
    ("outdoor", "defocus"): OUTDOOR_DEFOCUS,
}

CONFIG_KEY_ORDER = list(INDOOR_MOTION)
MODE_LOCKED_KEYS = {
    "deblur",
    "use_pos",
    "num_moments",
    "hidden",
    "lambda_s",
    "lambda_p",
    "max_clamp",
    "per_image_blur",
    "blur_label_path",
    "lambda_code",
    "lambda_delta",
    "sharp_weight",
    "densify_grad_threshold",
    "densify_prune_threshold",
    "densify_with_depth",
}
INT_KEYS = {
    "iterations",
    "resolution",
    "deblur",
    "use_pos",
    "num_moments",
    "hidden",
    "width",
    "densify_from_iter",
    "densify_until_iter",
    "densification_interval",
    "densify_with_depth",
    "prune_range",
    "pts_iter",
    "pts_dist",
    "pts_N_intpl",
    "pts_N_pts",
    "pts_add_bound",
    "per_image_blur",
    "blur_code_dim",
    "pre_deblur_warmup_iters",
    "pre_deblur_warmup_densify_from_iter",
    "luminance_start_iter",
    "gdags_start_iter",
    "gdags_probe_interval",
    "gdags_newborn_protect_iters",
}
FLOAT_KEYS = {
    "gtnet_lr",
    "position_lr_final",
    "percent_dense",
    "lambda_dssim",
    "lambda_s",
    "lambda_p",
    "max_clamp",
    "lambda_code",
    "lambda_delta",
    "sharp_weight",
    "motion_weight",
    "defocus_weight",
    "luminance_lr",
    "luminance_lambda_gain",
    "luminance_lambda_bias",
    "densify_grad_threshold",
    "densify_prune_threshold",
    "pts_rate",
}
BOOL_KEYS = {
    "white_background",
    "eval",
    "luminance_enable",
    "pre_deblur_warmup_enable",
    "pre_deblur_warmup_densify",
    "luminance_per_channel",
    "luminance_matrix_enable",
    "luminance_curve_enable",
    "gdags_stats_enable",
    "gdags_enable",
    "gdags_probe_use_autograd_grad",
    "gdags_clone_enable",
    "gdags_split_enable",
    "gdags_prune_enable",
}
STRING_KEYS: set[str] = {"blur_label_path", "pc_name", "renderer_backend", "renderer_backend_deblur", "luminance_mode"}


def run_dash_deblur_group_training(
    *,
    scene_dir: Path,
    work_dir: Path,
    final_ply: Path,
    final_spz: Path | None,
    options: dict[str, Any],
    repo_cache_dir: Path,
    blur_analysis: Any | None = None,
    progress: Progress | None = None,
) -> DashDeblurGroupResult:
    paths = resolve_runtime_paths(options, repo_cache_dir)
    runtime_dir = work_dir / "dash_deblur_group"
    output_dir = runtime_dir / "model"
    config_path = runtime_dir / "train_config.txt"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    deblur_mode = resolve_effective_deblur_mode(options, blur_analysis)
    config = build_training_config(options, blur_analysis=blur_analysis)
    label_counts = {"motion": 0, "defocus": 0, "sharp": 0}
    labels: dict[str, str] = {}
    if normalize_deblur_mode(options) != "sharp":
        label_path = runtime_dir / "blur_labels.json"
        labels, label_counts = write_blur_label_file(label_path, blur_analysis)
    if int(config.get("deblur", 0) or 0) != 0 and labels:
        config["blur_label_path"] = str(runtime_dir / "blur_labels.json")
        config["per_image_blur"] = 1
        if label_counts["motion"] + label_counts["defocus"] == 0:
            config["deblur"] = 0
            config["per_image_blur"] = 0
    else:
        config["per_image_blur"] = 0
    write_training_config(config_path, config)
    print(
        "[dash_deblur_group] training config "
        f"pc_name={config['pc_name']} renderer_backend={config['renderer_backend']} "
        f"renderer_backend_deblur={config['renderer_backend_deblur']} iterations={config['iterations']}",
        flush=True,
    )
    if config_uses_gsplat(config):
        prewarm_gsplat_kernels(paths, progress=progress, label=paths.trainer_flavor)

    command = build_training_command(
        paths=paths,
        scene_dir=scene_dir,
        output_dir=output_dir,
        config_path=config_path,
        expname=str(options.get("fine_expname") or "dash_deblur_group"),
        config=config,
    )
    run_training_process(command, cwd=paths.repo_dir, iterations=int(config["iterations"]), progress=progress, label=paths.trainer_flavor)

    produced_ply = locate_final_ply(output_dir, expected_iteration=int(config["iterations"]))
    if produced_ply is None:
        raise FineFailure("FINE_TRAINING_OUTPUT_MISSING", f"DashDeblurGroupGS did not produce a final PLY under {output_dir}")

    final_ply.parent.mkdir(parents=True, exist_ok=True)
    raw_final_ply = runtime_dir / "final_raw.ply"
    shutil.copy2(produced_ply, raw_final_ply)
    try:
        from app.fine.viewer_meta import write_far_noise_filtered_ply

        far_noise_metrics = write_far_noise_filtered_ply(
            raw_final_ply,
            final_ply,
            profile=str(options.get("fine_scene_profile") or options.get("preview_scene_profile") or "mixed_balanced"),
        )
    except Exception as exc:
        raise FineFailure("FINE_OUTPUT_FILTER_FAILED", f"failed to filter final Gaussian PLY: {exc}") from exc
    splat_count = read_ply_vertex_count(final_ply)

    resolved_spz: Path | None = None
    if final_spz and read_bool(options.get("fine_spz_enabled"), True):
        try:
            convert_ply_to_spz(final_ply, final_spz)
        except PreviewFailure as exc:
            raise FineFailure(exc.code, exc.message) from exc
        resolved_spz = final_spz

    source_commit = git_commit(paths.repo_dir)
    metrics = {
        "fine_training_backend": "dash_deblur_group_gs",
        "fine_training_flavor": paths.trainer_flavor,
        "fine_trainer_repo": str(paths.repo_dir),
        "fine_train_entrypoint": str(paths.train_py),
        "fine_train_output_dir": str(output_dir),
        "fine_train_config": str(config_path),
        "fine_train_metrics_csv": str(output_dir / "experiment_metrics.csv"),
        "fine_train_psnr_log": str(output_dir / "psnr.txt"),
        "fine_train_final_metrics": str(output_dir / "final_metrics.txt"),
        "fine_deblur_mode": deblur_mode.effective,
        "fine_deblur_mode_requested": deblur_mode.requested,
        "fine_deblur_mode_effective": deblur_mode.effective,
        "deblur_auto_confidence": deblur_mode.confidence,
        "deblur_auto_reason": deblur_mode.reason,
        "deblur_auto_motion_frames": deblur_mode.motion_frames,
        "deblur_auto_defocus_frames": deblur_mode.defocus_frames,
        "deblur_auto_mixed_frames": deblur_mode.mixed_frames,
        "deblur_auto_sharp_frames": deblur_mode.sharp_frames,
        "deblur": int(config["deblur"]),
        "deblur_strategy": _deblur_strategy(config),
        "deblur_applied_images": _deblur_applied_image_count(config, blur_analysis, label_counts),
        "deblur_label_motion_images": label_counts["motion"],
        "deblur_label_defocus_images": label_counts["defocus"],
        "deblur_label_sharp_images": label_counts["sharp"],
        "blur_code_dim": int(config["blur_code_dim"]),
        "pre_deblur_warmup_enable": read_bool(config.get("pre_deblur_warmup_enable"), False),
        "pre_deblur_warmup_iters": int(config["pre_deblur_warmup_iters"]),
        "luminance_enable": read_bool(config.get("luminance_enable"), False),
        "gdags_stats_enable": read_bool(config.get("gdags_stats_enable"), False),
        "gdags_enable": read_bool(config.get("gdags_enable"), False),
        "resolution": int(config["resolution"]),
        "use_pos": int(config["use_pos"]),
        "num_moments": int(config["num_moments"]),
        "densify_with_depth": int(config["densify_with_depth"]),
        "pc_name": str(config["pc_name"]),
        "renderer_backend": str(config["renderer_backend"]),
        "renderer_backend_deblur": str(config["renderer_backend_deblur"]),
        "fine_eap_enabled": read_bool(options.get("fine_eap_enabled"), True),
        "fine_gsplat_enabled": read_bool(options.get("fine_gsplat_enabled"), False),
        "densify_until_iter": int(config["densify_until_iter"]),
        "pts_iter": int(config["pts_iter"]),
        "pts_rate": float(config["pts_rate"]),
        "pts_N_pts": int(config["pts_N_pts"]),
        "iterations": int(config["iterations"]),
        "splat_count": splat_count,
        "final_spz_enabled": resolved_spz is not None,
        **far_noise_metrics,
    }
    return DashDeblurGroupResult(
        final_ply=final_ply,
        final_spz=resolved_spz,
        output_dir=output_dir,
        config_path=config_path,
        splat_count=splat_count,
        source_commit=source_commit,
        metrics=metrics,
    )


def _deblur_strategy(config: dict[str, Any]) -> str:
    if int(config.get("deblur", 0) or 0) == 0:
        return "disabled"
    if int(config.get("per_image_blur", 0) or 0) != 0:
        return "per_image_blur_type"
    return "all_training_images"


def write_blur_label_file(path: Path, blur_analysis: Any | None) -> tuple[dict[str, str], dict[str, int]]:
    label_records = build_blur_label_records(blur_analysis)
    labels = {name: str(record["blur_type"]) for name, record in label_records.items()}
    counts = {"motion": 0, "defocus": 0, "sharp": 0}
    for value in labels.values():
        counts[value] += 1
    if labels:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "labels": label_records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return labels, counts


def build_blur_label_records(blur_analysis: Any | None) -> dict[str, dict[str, Any]]:
    registry = getattr(blur_analysis, "per_frame_blur", None)
    if not isinstance(registry, dict):
        return {}
    labels: dict[str, dict[str, Any]] = {}
    numeric_keys = (
        "deblurweight",
        "deblur_weight",
        "blur_weight",
        "blurry_patch_ratio",
        "raw_score",
        "sharp_score",
        "laplacian",
        "gradient",
        "fft_high_ratio",
    )
    for key, item in registry.items():
        if not isinstance(item, dict) or item.get("rejected"):
            continue
        training_image = item.get("training_image") or key
        if training_image is None or str(training_image).startswith("rejected:"):
            continue
        record = {"blur_type": normalize_blur_label(item)}
        for numeric_key in numeric_keys:
            value = _finite_float_or_none(item.get(numeric_key))
            if value is not None:
                record[numeric_key] = value
        labels[str(training_image)] = record
    return labels


def build_blur_labels(blur_analysis: Any | None) -> dict[str, str]:
    return {name: str(record["blur_type"]) for name, record in build_blur_label_records(blur_analysis).items()}


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def normalize_blur_label(item: dict[str, Any]) -> str:
    detector_label = str(item.get("detector_label") or "").strip().lower()
    detector_blur_type = str(item.get("detector_blur_type") or "").strip().lower()
    if detector_label == "sharp" or detector_blur_type in {"none", "sharp"}:
        return "sharp"
    if detector_blur_type in {"defocus", "defocus_blur"}:
        return "defocus"
    if detector_label in {"blurry", "uncertain"} or detector_blur_type in {"motion", "motion_blur", "blur_unknown", "uncertain"}:
        return "motion"

    if not bool(item.get("blurred")):
        return "sharp"
    kind = str(item.get("kind") or item.get("blur_type") or "").strip().lower()
    if kind in {"defocus", "defocus_blur"}:
        return "defocus"
    if kind in {"sharp", "none"}:
        return "sharp"
    return "motion"


def _deblur_applied_image_count(config: dict[str, Any], blur_analysis: Any | None, label_counts: dict[str, int] | None = None) -> int:
    if int(config.get("deblur", 0) or 0) == 0:
        return 0
    if int(config.get("per_image_blur", 0) or 0) != 0 and label_counts is not None:
        return int(label_counts.get("motion", 0)) + int(label_counts.get("defocus", 0))
    for attr in ("kept_images", "input_images", "normalized_image_count"):
        value = int(getattr(blur_analysis, attr, 0) or 0)
        if value > 0:
            return value
    return 0


def resolve_runtime_paths(options: dict[str, Any], repo_cache_dir: Path) -> DashDeblurGroupPaths:
    repo_value = str(options.get("fine_trainer_repo") or os.getenv("DASH_DEBLUR_GROUP_REPO") or "").strip()
    repo_dir = Path(repo_value) if repo_value else default_embedded_trainer_dir(repo_cache_dir)
    repo_dir = repo_dir.expanduser().resolve()
    requested_flavor = str(options.get("fine_training_flavor") or os.getenv("DASH_DEBLUR_GROUP_FLAVOR") or "auto").strip().lower()
    trainer_flavor = detect_trainer_flavor(repo_dir) if requested_flavor in {"", "auto"} else requested_flavor
    if trainer_flavor != "dash_deblur_group":
        raise FineFailure("FINE_TRAINER_FLAVOR_INVALID", f"unsupported fine training flavor: {trainer_flavor}")
    explicit_entrypoint = str(options.get("fine_train_entrypoint") or "").strip()
    entrypoint = explicit_entrypoint or "train.py"
    train_py = (repo_dir / entrypoint).resolve()
    if repo_dir not in train_py.parents and repo_dir != train_py.parent:
        raise FineFailure("FINE_TRAINER_ENTRYPOINT_INVALID", f"fine train entrypoint escapes repo: {entrypoint}")
    if not train_py.exists():
        raise FineFailure(
            "FINE_TRAINER_NOT_CONFIGURED",
            f"fine trainer entrypoint not found at {train_py}. Set DASH_DEBLUR_GROUP_REPO or fine_trainer_repo.",
        )
    python = str(options.get("fine_train_python") or os.getenv("DASH_DEBLUR_GROUP_PYTHON") or sys.executable)
    return DashDeblurGroupPaths(repo_dir=repo_dir, train_py=train_py, python=python, trainer_flavor=trainer_flavor)


def default_embedded_trainer_dir(repo_cache_dir: Path) -> Path:
    container_path = Path("/opt/dash_deblur_group_gs")
    if container_path.exists():
        return container_path
    workspace_path = Path(__file__).resolve().parents[3] / "worker" / "trainer" / "dash_deblur_group_gs"
    if workspace_path.exists():
        return workspace_path
    return repo_cache_dir / "DashDeblurGroupGS"


def build_training_config(options: dict[str, Any], blur_analysis: Any | None = None) -> dict[str, Any]:
    scene_type = normalize_scene_type(options)
    deblur_mode = resolve_effective_deblur_mode(options, blur_analysis).effective
    base_mode = deblur_mode if deblur_mode in {"motion", "mix", "defocus"} else "motion"
    config = dict(CONFIG_PRESETS[(scene_type, base_mode)])
    if deblur_mode == "sharp":
        config.update({"deblur": 0, "lambda_s": 0.0, "lambda_p": 0.0, "num_moments": 1})

    if "fine_iterations" in options:
        options = {**options, "iterations": options["fine_iterations"]}
    for key in CONFIG_KEY_ORDER:
        if key not in options:
            continue
        if key in INT_KEYS:
            config[key] = read_int(options.get(key), int(config[key]), minimum=-1, maximum=10_000_000)
        elif key in FLOAT_KEYS:
            config[key] = read_float(options.get(key), float(config[key]), minimum=0.0, maximum=1_000_000.0)
        elif key in BOOL_KEYS:
            config[key] = read_bool(options.get(key), bool(config[key]))
        elif key in STRING_KEYS:
            config[key] = str(options.get(key) or config[key]).strip()
    apply_mode_locked_config(config, scene_type, deblur_mode)
    config["blur_code_dim"] = BLUR_CODE_DIM
    config["pc_name"] = "points3D_eap" if read_bool(options.get("fine_eap_enabled"), True) else "points3D"
    config["renderer_backend"] = "gsplat" if read_bool(options.get("fine_gsplat_enabled"), False) else "original"
    config["renderer_backend_deblur"] = "original"
    apply_training_safety_caps(config)
    return config


def apply_training_safety_caps(config: dict[str, Any]) -> None:
    iterations = max(1, int(config.get("iterations", 1) or 1))
    densify_from_iter = max(0, int(config.get("densify_from_iter", 0) or 0))
    densification_interval = max(1, int(config.get("densification_interval", 1) or 1))
    target_densify_until = min(iterations, max(int(iterations * 0.6), densify_from_iter + densification_interval))
    if int(config.get("densify_until_iter", 0) or 0) > target_densify_until:
        config["densify_until_iter"] = target_densify_until

    pts_iter = int(config.get("pts_iter", 0) or 0)
    pts_rate = float(config.get("pts_rate", 0.0) or 0.0)
    pts_n_pts = int(config.get("pts_N_pts", 0) or 0)
    if pts_iter == 2500 and abs(pts_rate - 1.1) < 1e-9 and pts_n_pts == 200000:
        config["pts_iter"] = 999999
        config["pts_rate"] = 0.0
        config["pts_N_pts"] = 0
    elif pts_rate <= 0.0 and pts_n_pts <= 0:
        config["pts_iter"] = 999999


def apply_mode_locked_config(config: dict[str, Any], scene_type: str, deblur_mode: str) -> None:
    if deblur_mode == "sharp":
        config.update({"deblur": 0, "use_pos": 0, "lambda_s": 0.0, "lambda_p": 0.0, "num_moments": 1})
        return
    base_mode = deblur_mode if deblur_mode in {"motion", "mix", "defocus"} else "motion"
    preset = CONFIG_PRESETS[(scene_type, base_mode)]
    for key in MODE_LOCKED_KEYS:
        config[key] = preset[key]


def write_training_config(path: Path, config: dict[str, Any]) -> None:
    lines = ["# Generated by backend/app/fine/dash_deblur_group.py"]
    for key in CONFIG_KEY_ORDER:
        value = config[key]
        if isinstance(value, bool):
            rendered = "True" if value else "False"
        else:
            rendered = str(value)
        lines.append(f"{key} = {rendered}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def detect_trainer_flavor(repo_dir: Path) -> str:
    train_py = repo_dir / "train.py"
    if file_contains(train_py, ("deblur", "--config")):
        return "dash_deblur_group"
    return "dash_deblur_group"


def file_contains(path: Path, needles: tuple[str, ...]) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")[:262_144]
    except OSError:
        return False
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def build_training_command(
    *,
    paths: DashDeblurGroupPaths,
    scene_dir: Path,
    output_dir: Path,
    config_path: Path,
    expname: str,
    config: dict[str, Any],
) -> list[str]:
    iterations = int(config["iterations"])
    test_iterations = training_metric_iterations(iterations)
    save_iterations = training_save_iterations(iterations)
    return [
        paths.python,
        "-u",
        str(paths.train_py),
        "-s",
        str(scene_dir),
        "--model_path",
        str(output_dir),
        "--expname",
        expname,
        "--config",
        str(config_path),
        "--test_iterations",
        *[str(iteration) for iteration in test_iterations],
        "--save_iterations",
        *[str(iteration) for iteration in save_iterations],
    ]


def training_metric_iterations(iterations: int, interval: int = 500) -> list[int]:
    iterations = max(1, int(iterations))
    interval = max(1, int(interval))
    values = list(range(interval, iterations + 1, interval))
    if iterations not in values:
        values.append(iterations)
    return values


def training_save_iterations(iterations: int, interval: int = 10_000) -> list[int]:
    iterations = max(1, int(iterations))
    interval = max(1, int(interval))
    values = list(range(interval, iterations + 1, interval))
    if iterations not in values:
        values.append(iterations)
    return values


GSPLAT_PREWARM_SCRIPT = r"""
import torch
from gsplat import rasterization

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available for gsplat precompile")

device = torch.device("cuda")
dtype = torch.float32
means = torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=dtype)
quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
scales = torch.full((1, 3), 0.01, device=device, dtype=dtype)
opacities = torch.ones((1,), device=device, dtype=dtype)
colors = torch.ones((1, 1, 3), device=device, dtype=dtype)
viewmats = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
Ks = torch.tensor([[[32.0, 0.0, 8.0], [0.0, 32.0, 8.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
backgrounds = torch.zeros((1, 3), device=device, dtype=dtype)

rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=viewmats,
    Ks=Ks,
    width=16,
    height=16,
    sh_degree=0,
    packed=False,
    backgrounds=backgrounds,
    camera_model="pinhole",
)
torch.cuda.synchronize()
print("gsplat CUDA kernels ready", flush=True)
"""


def config_uses_gsplat(config: dict[str, Any]) -> bool:
    return str(config.get("renderer_backend") or "").strip().lower() == "gsplat" or str(config.get("renderer_backend_deblur") or "").strip().lower() == "gsplat"


def gsplat_kernels_are_precompiled() -> bool:
    marker = Path(os.getenv("GSPLAT_PRECOMPILED_MARKER") or "/opt/torch_extensions/.gsplat_precompiled")
    extension_dir = Path(os.getenv("TORCH_EXTENSIONS_DIR") or "")
    try:
        return marker.exists() and extension_dir.resolve() == marker.parent.resolve()
    except OSError:
        return False


def prewarm_gsplat_kernels(paths: DashDeblurGroupPaths, *, progress: Progress | None = None, label: str = "dash_deblur_group") -> None:
    if gsplat_kernels_are_precompiled():
        print(f"[{label}] gsplat kernels ready (precompiled)", flush=True)
        if progress:
            progress("fine_training_preflight", 42, "gsplat kernels ready")
        return
    print(f"[{label}] precompiling gsplat CUDA kernels", flush=True)
    if progress:
        progress("fine_training_preflight", 42, "precompiling gsplat CUDA kernels")
    command = [paths.python, "-u", "-c", GSPLAT_PREWARM_SCRIPT]
    try:
        completed = subprocess.run(
            command,
            cwd=str(paths.repo_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            check=False,
        )
    except FileNotFoundError as exc:
        raise FineFailure("FINE_TRAINING_PYTHON_NOT_FOUND", f"fine training python not found: {paths.python}") from exc
    except subprocess.TimeoutExpired as exc:
        raise FineFailure("FINE_GSPLAT_PRECOMPILE_TIMEOUT", "gsplat CUDA kernel precompile timed out") from exc
    output = (completed.stdout or "").strip()
    if output:
        for line in output.splitlines():
            print(f"[{label}] gsplat precompile: {line}", flush=True)
    if completed.returncode != 0:
        raise FineFailure("FINE_GSPLAT_PRECOMPILE_FAILED", output or "gsplat CUDA kernel precompile failed")
    print(f"[{label}] gsplat kernels ready", flush=True)
    if progress:
        progress("fine_training_preflight", 42, "gsplat kernels ready")


def run_training_process(command: list[str], *, cwd: Path, iterations: int, progress: Progress | None, label: str = "dash_deblur_group") -> None:
    print(f"[{label}] command " + " ".join(command), flush=True)
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise FineFailure("FINE_TRAINING_PYTHON_NOT_FOUND", f"fine training python not found: {command[0]}") from exc

    assert process.stdout is not None
    tail: list[str] = []
    last_progress = 44
    for line in process.stdout:
        for text in split_training_output(line):
            print(text, flush=True)
            tail.append(text)
            tail = tail[-80:]
            parsed = parse_training_progress(text, iterations)
            if parsed is not None:
                parsed_iter, parsed_total = parsed
                if parsed_iter % 200 != 0 and parsed_iter != parsed_total:
                    continue
                value = 44 + int(min(1.0, parsed_iter / parsed_total) * 34)
                if progress and value >= last_progress:
                    last_progress = max(last_progress, value)
                    progress("fine_training", value, text)
    return_code = process.wait()
    if return_code != 0:
        message = "\n".join(tail) or f"{label} exited with {return_code}"
        raise FineFailure("FINE_TRAINING_FAILED", message)
    if progress:
        progress("fine_training", 78, f"{label} training complete")


def locate_final_ply(output_dir: Path, *, expected_iteration: int | None = None) -> Path | None:
    candidates: list[Path] = []
    if expected_iteration is not None:
        candidates.append(output_dir / "point_cloud" / f"iteration_{expected_iteration}" / "point_cloud.ply")
    candidates.extend([output_dir / "final.ply", output_dir / "point_cloud.ply"])
    candidates.extend(output_dir.rglob("point_cloud.ply") if output_dir.exists() else [])
    candidates.extend(output_dir.rglob("*.ply") if output_dir.exists() else [])
    existing = [path for path in candidates if path.exists() and path.is_file() and path.stat().st_size > 0]
    if not existing:
        return None
    return sorted(existing, key=ply_rank, reverse=True)[0]


def ply_rank(path: Path) -> tuple[int, float]:
    match = re.search(r"iteration_(\d+)", str(path))
    iteration = int(match.group(1)) if match else -1
    return iteration, path.stat().st_mtime


def parse_iteration(line: str) -> int | None:
    patterns = [
        r"\biteration\s*[:=]?\s*(\d+)\b",
        r"\biter\s*[:=]?\s*(\d+)\b",
        r"\b(\d+)\s*/\s*\d+\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def parse_training_progress(line: str, fallback_iterations: int) -> tuple[int, int] | None:
    matches = re.findall(r"\b(\d+)\s*/\s*(\d+)\s*\[[^\]]+\]", line)
    if matches:
        current, total = matches[-1]
        return int(current), max(1, int(total))
    parsed_iter = parse_iteration(line)
    if parsed_iter is None or fallback_iterations <= 0:
        return None
    return parsed_iter, fallback_iterations


def split_training_output(line: str) -> list[str]:
    normalized = line.rstrip("\n").replace("\r", "\n")
    normalized = re.sub(r"(?<!^)(\[ITER\s+\d+\])", r"\n\1", normalized)
    return [part.strip() for part in normalized.split("\n") if part.strip()]


def read_ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii", errors="strict").strip()
        if first != "ply":
            raise FineFailure("PLY_INVALID", f"not a PLY file: {path}")
        while True:
            line_bytes = handle.readline()
            if not line_bytes:
                raise FineFailure("PLY_INVALID", f"PLY header terminator missing: {path}")
            line = line_bytes.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise FineFailure("PLY_INVALID", f"PLY vertex count missing: {path}")


def normalize_scene_type(options: dict[str, Any]) -> str:
    value = str(options.get("fine_scene_type") or options.get("scene_type") or "indoor").strip().lower()
    return "outdoor" if value in {"outdoor", "outside", "outdoor_fast_clean"} else "indoor"


def normalize_deblur_mode(options: dict[str, Any]) -> str:
    enabled = str(options.get("fine_deblur_enabled", "true")).strip().lower()
    if enabled in {"0", "false", "no", "off", "sharp"}:
        return "sharp"
    mode_key = "fine_deblur_mode_requested" if "fine_deblur_mode_requested" in options else "fine_deblur_mode"
    if mode_key in options:
        value = str(options.get(mode_key) or "motion").strip().lower()
        if value in {"mix", "auto", "automatic"}:
            return "motion"
        if value in {"defocus", "2"}:
            return "defocus"
        if value in {"sharp", "0", "none", "off"}:
            return "sharp"
        return "motion"
    try:
        code = int(options.get("deblur"))
    except (TypeError, ValueError):
        code = 1
    if code == 2:
        return "defocus"
    if code == 0:
        return "sharp"
    return "motion"


def resolve_effective_deblur_mode(options: dict[str, Any], blur_analysis: Any | None = None) -> EffectiveDeblurMode:
    requested = normalize_deblur_mode(options)
    counts = count_training_blur_kinds(blur_analysis)
    if requested == "sharp":
        effective = "sharp"
        reason = "user_selected"
    elif counts["motion"] > 0:
        effective = "motion"
        reason = "per_image_blur_labels"
    elif counts["defocus"] > 0:
        effective = "defocus"
        reason = "per_image_blur_labels"
    elif counts["sharp"] > 0:
        effective = "sharp"
        reason = "per_image_blur_labels"
    else:
        effective = requested
        reason = "user_selected"
    return EffectiveDeblurMode(
        requested,
        effective,
        "explicit",
        reason,
        counts["motion"],
        counts["defocus"],
        0,
        counts["sharp"],
    )


def count_training_blur_kinds(blur_analysis: Any | None) -> dict[str, int]:
    counts = {"motion": 0, "defocus": 0, "mixed": 0, "sharp": 0}
    registry = getattr(blur_analysis, "per_frame_blur", None)
    if not isinstance(registry, dict):
        raw_mode = getattr(blur_analysis, "mode", None)
        if raw_mode is None:
            return counts
        mode = str(raw_mode or "sharp").lower()
        if mode == "defocus":
            counts["defocus"] += int(getattr(blur_analysis, "training_blur_frames", 0) or 0) or 1
        elif mode in {"motion", "mixed"}:
            counts["motion"] += int(getattr(blur_analysis, "training_blur_frames", 0) or 0) or 1
        elif mode == "sharp":
            counts["sharp"] += int(getattr(blur_analysis, "kept_images", 0) or 0) or 1
        return counts
    for item in registry.values():
        if not isinstance(item, dict) or item.get("rejected"):
            continue
        label = normalize_blur_label(item)
        if label == "sharp":
            counts["sharp"] += 1
        elif label == "defocus":
            counts["defocus"] += 1
        else:
            counts["motion"] += 1
    return counts


def deblur_mode_from_config(config: dict[str, Any]) -> str:
    code = int(config.get("deblur", 1))
    if code == 0:
        return "sharp"
    if not read_bool(config.get("use_pos"), True):
        return "defocus"
    if code >= 3:
        return "mix"
    if code == 2:
        return "defocus"
    return "motion"


def read_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return fallback


def git_commit(repo_dir: Path) -> str:
    marker = repo_dir / "UPSTREAM_SOURCE.txt"
    if marker.exists():
        match = re.search(r"^Commit:\s*([0-9a-fA-F]{7,40})\s*$", marker.read_text(encoding="utf-8", errors="ignore"), re.MULTILINE)
        if match:
            return match.group(1)
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            timeout=5,
            check=False,
        )
    except Exception:
        return "unknown"
    value = (completed.stdout or "").strip()
    return value if completed.returncode == 0 and value else "unknown"
