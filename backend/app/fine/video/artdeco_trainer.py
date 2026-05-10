from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.config import Settings
from app.fine.option_utils import read_float, read_int
from app.fine.types import FineFailure
from app.fine.video.speed3r_pi3 import ensure_video_artdeco_weights
from app.fine.video.types import ArtdecoTrainingResult, CameraIntrinsics, ExtractedVideoFrames


ARTDECO_COMMIT = "bb654395826e50ac9e4671682d901377115a24ce"
SPEED3R_COMMIT = "5460f7309c87e5daac36385ff6611627de7d7267"


def run_artdeco_speed3r_training(
    *,
    frames: ExtractedVideoFrames,
    intrinsics: CameraIntrinsics,
    intrinsics_path: Path,
    output_dir: Path,
    model_cache_dir: Path,
    settings: Settings,
    options: dict[str, Any],
    progress,
) -> ArtdecoTrainingResult:
    weight_paths = ensure_video_artdeco_weights(model_cache_dir)
    artdeco_root = resolve_artdeco_root(settings)
    speed3r_root = resolve_speed3r_root(settings)
    runtime_root = output_dir.parent / "artdeco_runtime"
    models_dir = stage_artdeco_models(runtime_root, weight_paths)

    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_artdeco_command(
        artdeco_root=artdeco_root,
        speed3r_root=speed3r_root,
        models_dir=models_dir,
        frames=frames,
        intrinsics=intrinsics,
        intrinsics_path=intrinsics_path,
        output_dir=output_dir,
        options=options,
    )
    env = os.environ.copy()
    env["ARTDECO_ROOT"] = str(artdeco_root)
    env["SPEED3R_ROOT"] = str(speed3r_root) if speed3r_root else env.get("SPEED3R_ROOT", "")
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[3]), env.get("PYTHONPATH", "")])
    progress("artdeco_training", 34, "running ARTDECO VSLAM + h3dgsv3 mapper with Speed3R-Pi3", {"artdeco_root": str(artdeco_root)})
    process = subprocess.Popen(
        command,
        cwd=runtime_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise FineFailure("ARTDECO_TRAINING_FAILED", f"ARTDECO video trainer exited with code {return_code}")

    gs_ply = output_dir / "point_clouds" / "gs.ply"
    if not gs_ply.exists() or gs_ply.stat().st_size <= 0:
        raise FineFailure("ARTIFACT_NOT_FOUND", f"ARTDECO did not create non-empty point_clouds/gs.ply: {gs_ply}")

    metrics = read_artdeco_metrics(output_dir)
    metrics.update(
        {
            "artdeco_root": str(artdeco_root),
            "speed3r_root": str(speed3r_root) if speed3r_root else None,
            "speed3r_pi3_model_dir": str(models_dir),
            "artdeco_output_dir": str(output_dir),
        }
    )
    return ArtdecoTrainingResult(output_dir=output_dir, gs_ply=gs_ply, metrics=metrics)


def build_artdeco_command(
    *,
    artdeco_root: Path,
    speed3r_root: Path | None,
    models_dir: Path,
    frames: ExtractedVideoFrames,
    intrinsics: CameraIntrinsics,
    intrinsics_path: Path,
    output_dir: Path,
    options: dict[str, Any],
) -> list[str]:
    key_iterations = read_int(options.get("fine_artdeco_key_iterations"), 20, minimum=1, maximum=2_000)
    common_iterations = read_int(options.get("fine_artdeco_common_iterations"), 10, minimum=0, maximum=2_000)
    finetune_iteration = read_int(options.get("fine_artdeco_finetune_iteration"), 10_000, minimum=0, maximum=200_000)
    downsampling = read_float(options.get("fine_artdeco_downsampling"), 2.0, minimum=1.0, maximum=8.0)
    local_feat_dim = read_int(options.get("fine_artdeco_local_feat_dim"), 16, minimum=1, maximum=128)
    global_feat_dim = read_int(options.get("fine_artdeco_global_feat_dim"), 16, minimum=1, maximum=128)
    config_path = artdeco_root / "config" / "base.yaml"
    if not config_path.exists():
        raise FineFailure("ARTDECO_RUNTIME_UNAVAILABLE", f"ARTDECO config missing: {config_path}")
    command = [
        sys.executable,
        "-m",
        "app.fine.video.artdeco_entrypoint",
        "--artdeco-root",
        str(artdeco_root),
        "--speed3r-model-dir",
        str(models_dir),
        "--artdeco-config",
        str(config_path),
        "--metrics-json",
        str(output_dir / "artdeco_entrypoint_metrics.json"),
        "-s",
        str(frames.dataset_root),
        "-i",
        "images",
        "-m",
        str(output_dir),
        "-d",
        "selfCaptured",
        "--calib",
        str(intrinsics_path),
        "--device_frontend",
        str(options.get("fine_artdeco_device_frontend") or "cuda:0"),
        "--device_backend",
        str(options.get("fine_artdeco_device_backend") or "cuda:0"),
        "--device_mapper",
        str(options.get("fine_artdeco_device_mapper") or "cuda:0"),
        "--device_shared",
        str(options.get("fine_artdeco_device_shared") or "cpu"),
        "--downsampling",
        str(downsampling),
        "--test_hold",
        str(read_int(options.get("fine_artdeco_test_hold"), -1, minimum=-1, maximum=10_000)),
        "--use_all_frames",
        "--base_model",
        "h3dgsv3",
        "--num_key_iterations",
        str(key_iterations),
        "--num_common_iterations",
        str(common_iterations),
        "--local_feat_dim",
        str(local_feat_dim),
        "--global_feat_dim",
        str(global_feat_dim),
        "--visible_threshold",
        str(read_float(options.get("fine_artdeco_visible_threshold"), 0.0, minimum=0.0, maximum=1.0)),
        "--gs_add_ratio",
        str(read_float(options.get("fine_artdeco_gs_add_ratio"), 1.0, minimum=0.01, maximum=1.0)),
        "--covariance_filter",
        "--point_fusion_frontend",
        "--accurate_loop_closure",
        "--viewer_mode",
        "none",
    ]
    if speed3r_root:
        command[command.index("--speed3r-model-dir") : command.index("--speed3r-model-dir")] = ["--speed3r-root", str(speed3r_root)]
    if intrinsics.optimize_focal:
        command.append("--optimize_focal")
    if finetune_iteration > 0:
        command.extend(["--save_at_finetune_iteration", str(finetune_iteration)])
    return command


def resolve_artdeco_root(settings: Settings) -> Path:
    candidates = [
        os.getenv("ARTDECO_ROOT"),
        str(Path(settings.repo_cache_dir) / "artdeco" / "source"),
        "/opt/artdeco-runtime",
        "/opt/artdeco",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).resolve()
        if (path / "VSLAM").exists() and (path / "Reconstruct" / "scene" / "scene_models" / "h3dgsv3.py").exists():
            return path
    raise FineFailure("ARTDECO_RUNTIME_UNAVAILABLE", "ARTDECO runtime source is missing; set ARTDECO_ROOT or build the worker image")


def resolve_speed3r_root(settings: Settings) -> Path | None:
    candidates = [
        os.getenv("SPEED3R_ROOT"),
        str(Path(settings.repo_cache_dir) / "speed3r" / "source"),
        "/opt/speed3r-runtime",
        "/opt/speed3r",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).resolve()
        if (path / "pi3" / "models" / "pi3_sparse.py").exists():
            return path
    return None


def stage_artdeco_models(runtime_root: Path, paths: dict[str, Path]) -> Path:
    models_dir = runtime_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    link_or_copy(paths["config"], models_dir / "config.json")
    link_or_copy(paths["model"], models_dir / "model.safetensors")
    link_or_copy(paths["mast3r"], models_dir / paths["mast3r"].name)
    link_or_copy(paths["mast3r_retrieval"], models_dir / paths["mast3r_retrieval"].name)
    link_or_copy(paths["mast3r_codebook"], models_dir / paths["mast3r_codebook"].name)
    return models_dir


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return
    target.unlink(missing_ok=True)
    try:
        os.symlink(source, target)
    except OSError:
        shutil.copy2(source, target)


def read_artdeco_metrics(output_dir: Path) -> dict[str, Any]:
    for name in ("artdeco_entrypoint_metrics.json", "metadata.json"):
        path = output_dir / name
        if path.exists():
            try:
                import json

                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}
