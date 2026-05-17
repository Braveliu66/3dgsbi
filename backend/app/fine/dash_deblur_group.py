from __future__ import annotations

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
    "iterations": 24000,
    "resolution": 2,
    "white_background": False,
    "eval": True,
    "deblur": 1,
    "use_pos": 1,
    "num_moments": 4,
    "hidden": 3,
    "width": 64,
    "gtnet_lr": 0.001,
    "lambda_s": 0.01,
    "lambda_p": 0.008,
    "max_clamp": 1.08,
    "densify_from_iter": 800,
    "densify_until_iter": 17000,
    "densification_interval": 100,
    "densify_grad_threshold": 0.00045,
    "densify_prune_threshold": 0.008,
    "densify_with_depth": 1,
    "prune_range": 3,
    "pts_iter": 2500,
    "pts_rate": 1.1,
    "pts_dist": 2,
    "pts_N_intpl": 4,
    "pts_N_pts": 200000,
    "pts_add_bound": 10,
    "protect_new_points_iters": 1500,
    "dash_enable": True,
    "dash_start_iter": 3000,
    "resolution_mode": "freq",
    "densify_mode": "freq",
    "max_n_gaussian": -1,
    "dash_max_reso_scale": 4,
    "dash_start_significance_factor": 4,
    "dash_max_densify_rate_per_step": 0.12,
    "Grouping": True,
    "grouping_method": "Opacity-weighted",
    "UTR": 0.78,
    "grouping_from_iter": 4500,
    "grouping_until_iter": 20000,
    "grouping_interval": 600,
    "grouping_freeze_around_pts": 1000,
}

INDOOR_DEFOCUS = {
    **INDOOR_MOTION,
    "iterations": 22000,
    "deblur": 2,
    "num_moments": 3,
    "hidden": 2,
    "lambda_s": 0.008,
    "lambda_p": 0.0,
    "max_clamp": 1.06,
    "densify_until_iter": 16000,
    "densify_grad_threshold": 0.00018,
    "densify_prune_threshold": 0.0045,
    "dash_max_densify_rate_per_step": 0.10,
    "UTR": 0.82,
    "grouping_until_iter": 18500,
}

OUTDOOR_MOTION = {
    **INDOOR_MOTION,
    "iterations": 30000,
    "resolution": 4,
    "lambda_p": 0.01,
    "max_clamp": 1.10,
    "densify_from_iter": 1000,
    "densify_until_iter": 22000,
    "densify_grad_threshold": 0.0005,
    "prune_range": 4,
    "pts_iter": 3500,
    "pts_rate": 1.3,
    "pts_dist": 3,
    "pts_add_bound": 20,
    "protect_new_points_iters": 2500,
    "dash_start_iter": 5000,
    "dash_max_densify_rate_per_step": 0.10,
    "UTR": 0.75,
    "grouping_from_iter": 6500,
    "grouping_until_iter": 26000,
    "grouping_interval": 1000,
    "grouping_freeze_around_pts": 1500,
}

OUTDOOR_DEFOCUS = {
    **OUTDOOR_MOTION,
    "iterations": 28000,
    "deblur": 2,
    "num_moments": 3,
    "hidden": 2,
    "lambda_s": 0.008,
    "lambda_p": 0.0,
    "max_clamp": 1.08,
    "densify_until_iter": 21000,
    "densify_grad_threshold": 0.00022,
    "densify_prune_threshold": 0.004,
    "dash_max_densify_rate_per_step": 0.09,
    "UTR": 0.78,
    "grouping_until_iter": 24500,
}

CONFIG_PRESETS = {
    ("indoor", "motion"): INDOOR_MOTION,
    ("indoor", "defocus"): INDOOR_DEFOCUS,
    ("outdoor", "motion"): OUTDOOR_MOTION,
    ("outdoor", "defocus"): OUTDOOR_DEFOCUS,
}

CONFIG_KEY_ORDER = list(INDOOR_MOTION)
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
    "protect_new_points_iters",
    "dash_start_iter",
    "max_n_gaussian",
    "dash_max_reso_scale",
    "dash_start_significance_factor",
    "grouping_from_iter",
    "grouping_until_iter",
    "grouping_interval",
    "grouping_freeze_around_pts",
}
FLOAT_KEYS = {
    "gtnet_lr",
    "lambda_s",
    "lambda_p",
    "max_clamp",
    "densify_grad_threshold",
    "densify_prune_threshold",
    "pts_rate",
    "dash_max_densify_rate_per_step",
    "UTR",
}
BOOL_KEYS = {"white_background", "eval", "dash_enable", "Grouping"}
STRING_KEYS = {"resolution_mode", "densify_mode", "grouping_method"}


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
    deblur_mode = resolve_effective_deblur_mode(options, blur_analysis)
    config = build_training_config(options, blur_analysis=blur_analysis)
    runtime_dir = work_dir / "dash_deblur_group"
    output_dir = runtime_dir / "model"
    config_path = runtime_dir / "train_config.txt"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_training_config(config_path, config)

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
    shutil.copy2(produced_ply, final_ply)
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
        "dash_enable": bool(config["dash_enable"]),
        "dash_start_iter": int(config["dash_start_iter"]),
        "resolution_mode": str(config["resolution_mode"]),
        "densify_mode": str(config["densify_mode"]),
        "Grouping": bool(config["Grouping"]),
        "UTR": float(config["UTR"]),
        "grouping_from_iter": int(config["grouping_from_iter"]),
        "grouping_until_iter": int(config["grouping_until_iter"]),
        "grouping_interval": int(config["grouping_interval"]),
        "iterations": int(config["iterations"]),
        "splat_count": splat_count,
        "final_spz_enabled": resolved_spz is not None,
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
    base_mode = "defocus" if deblur_mode == "defocus" else "motion"
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
    return config


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
    if file_contains(train_py, ("deblur", "Grouping", "--config")):
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
    ]


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
        text = line.rstrip()
        print(line, end="", flush=True)
        tail.append(text)
        tail = tail[-80:]
        parsed_iter = parse_iteration(text)
        if parsed_iter is not None and iterations > 0:
            value = 44 + int(min(1.0, parsed_iter / iterations) * 34)
            if progress and value > last_progress:
                last_progress = value
                progress("fine_training", value, f"{label} iteration {parsed_iter}/{iterations}")
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
    if "fine_deblur_mode" in options:
        value = str(options.get("fine_deblur_mode") or "mix").strip().lower()
        if value in {"mix", "auto", "automatic"}:
            return "mix"
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
    return "mix"


def resolve_effective_deblur_mode(options: dict[str, Any], blur_analysis: Any | None = None) -> EffectiveDeblurMode:
    requested = normalize_deblur_mode(options)
    if requested in {"motion", "defocus", "sharp"}:
        return EffectiveDeblurMode(requested, requested, "explicit", "user_selected", 0, 0, 0, 0)

    counts = count_training_blur_kinds(blur_analysis)
    motion = counts["motion"]
    defocus = counts["defocus"]
    mixed = counts["mixed"]
    sharp = counts["sharp"]
    scene = normalize_scene_type(options)
    fallback = "motion"

    if motion > defocus and motion >= max(2, mixed + 1):
        return EffectiveDeblurMode("mix", "motion", "high", "motion_vote", motion, defocus, mixed, sharp)
    if defocus > motion and defocus >= max(2, mixed + 1):
        return EffectiveDeblurMode("mix", "defocus", "high", "defocus_vote", motion, defocus, mixed, sharp)
    if motion == 1 and defocus == 0 and mixed == 0:
        return EffectiveDeblurMode("mix", "motion", "medium", "single_motion_frame", motion, defocus, mixed, sharp)
    if defocus == 1 and motion == 0 and mixed == 0:
        return EffectiveDeblurMode("mix", "defocus", "medium", "single_defocus_frame", motion, defocus, mixed, sharp)
    return EffectiveDeblurMode("mix", fallback, "low", f"{scene}_conservative_default", motion, defocus, mixed, sharp)


def count_training_blur_kinds(blur_analysis: Any | None) -> dict[str, int]:
    counts = {"motion": 0, "defocus": 0, "mixed": 0, "sharp": 0}
    registry = getattr(blur_analysis, "per_frame_blur", None)
    if not isinstance(registry, dict):
        mode = str(getattr(blur_analysis, "mode", "sharp") or "sharp").lower()
        if mode in counts:
            counts[mode] += int(getattr(blur_analysis, "training_blur_frames", 0) or 0) or 1
        return counts
    for item in registry.values():
        if not isinstance(item, dict) or item.get("rejected"):
            continue
        kind = str(item.get("kind") or "sharp").lower()
        blurred = bool(item.get("blurred"))
        if not blurred:
            counts["sharp"] += 1
        elif kind in {"motion", "defocus", "mixed"}:
            counts[kind] += 1
        else:
            counts["mixed"] += 1
    return counts


def deblur_mode_from_config(config: dict[str, Any]) -> str:
    code = int(config.get("deblur", 1))
    if code == 2:
        return "defocus"
    if code == 0:
        return "sharp"
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
