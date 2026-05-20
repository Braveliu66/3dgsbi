from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.fine.preprocess import (
    SceneBuildResult,
    ensure_colmap_sparse_zero,
    select_best_colmap_model,
    validate_colmap_pinhole_scene,
)
from app.fine.sparse_filter import write_filtered_sparse_points_ply
from app.fine.types import FineFailure
from app.preview.utils import image_files


Progress = Callable[[str, int, str], None]
REQUIRED_COLMAP_COMMANDS = {
    "feature_extractor",
    "exhaustive_matcher",
    "mapper",
    "image_undistorter",
}
GLOBAL_COLMAP_COMMAND = "global_mapper"


@dataclass(slots=True)
class ColmapCapabilities:
    executable: str
    help_text: str
    commands: set[str]
    has_aliked_lightglue: bool = False
    has_sift_lightglue: bool = False


def build_colmap_cli_scene(
    input_dir: Path,
    scene_dir: Path,
    *,
    scene_type: str,
    input_type: str,
    quality_mode: str = "auto",
    capture_order: str = "auto",
    matcher_policy: str = "auto",
    prefer_gpu: bool = True,
    gpu_index: str | None = None,
    min_registered_ratio: float | None = None,
    max_num_features: int | None = None,
    max_image_size: int | None = None,
    max_num_matches: int | None = None,
    sequential_overlap: int | None = None,
    progress: Progress,
) -> SceneBuildResult:
    files = image_files(input_dir)
    if len(files) < 3:
        raise FineFailure("INSUFFICIENT_IMAGES", "COLMAP initialization requires at least 3 images")

    capabilities = detect_colmap_capabilities()
    return _run_braveliu_colmap(
        capabilities,
        input_dir,
        scene_dir,
        mapper_command="mapper",
        backend="colmap_cli",
        matcher_policy=matcher_policy,
        use_gpu=prefer_gpu,
        gpu_index=gpu_index,
        min_registered_ratio=min_registered_ratio,
        max_num_features=max_num_features,
        max_image_size=max_image_size,
        progress=progress,
    )


def build_colmap_global_scene(
    input_dir: Path,
    scene_dir: Path,
    *,
    scene_type: str,
    input_type: str,
    quality_mode: str = "auto",
    capture_order: str = "auto",
    matcher_policy: str = "auto",
    prefer_gpu: bool = True,
    gpu_index: str | None = None,
    min_registered_ratio: float | None = None,
    max_num_features: int | None = None,
    max_image_size: int | None = None,
    max_num_matches: int | None = None,
    sequential_overlap: int | None = None,
    progress: Progress,
) -> SceneBuildResult:
    files = image_files(input_dir)
    if len(files) < 3:
        raise FineFailure("INSUFFICIENT_IMAGES", "COLMAP initialization requires at least 3 images")

    capabilities = detect_colmap_capabilities()
    _require_colmap_command(capabilities, GLOBAL_COLMAP_COMMAND, "COLMAP global mapper backend")
    return _run_braveliu_colmap(
        capabilities,
        input_dir,
        scene_dir,
        mapper_command=GLOBAL_COLMAP_COMMAND,
        backend="colmap_global",
        matcher_policy=matcher_policy,
        use_gpu=prefer_gpu,
        gpu_index=gpu_index,
        min_registered_ratio=min_registered_ratio,
        max_num_features=max_num_features,
        max_image_size=max_image_size,
        progress=progress,
    )


def detect_colmap_capabilities() -> ColmapCapabilities:
    executable = os.environ.get("DBULR_COLMAP_BINARY") or shutil.which("colmap")
    if not executable:
        raise FineFailure("COLMAP_CLI_UNAVAILABLE", "colmap executable was not found in PATH")

    completed = subprocess.run(
        [executable, "help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )
    help_text = completed.stdout or ""
    if completed.returncode != 0:
        raise FineFailure("COLMAP_CLI_UNAVAILABLE", help_text.strip() or f"colmap help exited with {completed.returncode}")

    commands = {match.group(1) for match in re.finditer(r"^\s+([a-z_]+)(?:\s|$)", help_text, flags=re.MULTILINE)}
    commands.update(command for command in REQUIRED_COLMAP_COMMANDS if _help_mentions_command(help_text, command))
    missing = sorted(REQUIRED_COLMAP_COMMANDS - commands)
    if missing:
        raise FineFailure("COLMAP_CLI_UNSUPPORTED", f"colmap CLI is missing required commands: {', '.join(missing)}")

    return ColmapCapabilities(
        executable=executable,
        help_text=help_text,
        commands=commands,
        has_aliked_lightglue="ALIKED_LIGHTGLUE" in help_text,
        has_sift_lightglue="SIFT_LIGHTGLUE" in help_text,
    )


def _run_braveliu_colmap(
    capabilities_or_executable: ColmapCapabilities | str,
    input_dir: Path,
    scene_dir: Path,
    *,
    mapper_command: str,
    backend: str,
    matcher_policy: str,
    use_gpu: bool,
    gpu_index: str | None,
    min_registered_ratio: float | None,
    max_num_features: int | None,
    max_image_size: int | None,
    progress: Progress,
) -> SceneBuildResult:
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    if isinstance(capabilities_or_executable, ColmapCapabilities):
        capabilities = capabilities_or_executable
    else:
        capabilities = ColmapCapabilities(executable=capabilities_or_executable, help_text="", commands=set(REQUIRED_COLMAP_COMMANDS))
    executable = capabilities.executable
    workspace = scene_dir / "colmap_workspace"
    images_dir = workspace / "images"
    sparse_dir = workspace / "sparse"
    undistorted_dir = scene_dir / "colmap"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    files = image_files(input_dir)
    for index, path in enumerate(files, start=1):
        shutil.copy2(path, images_dir / f"{index:03d}_{path.stem}.jpg")

    database_path = workspace / "database.db"
    gpu_flag = "1" if use_gpu else "0"
    started = time.monotonic()
    matcher_command = _resolve_matcher_command(matcher_policy, len(files))
    _require_colmap_command(capabilities, matcher_command, "COLMAP matching")

    progress("fine_colmap_features", 24, "running COLMAP feature extraction")
    feature_cmd = [
        executable,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_dir),
        "--ImageReader.single_camera",
        "1",
        "--SiftExtraction.use_gpu",
        gpu_flag,
    ]
    if gpu_index:
        feature_cmd.extend(["--SiftExtraction.gpu_index", gpu_index])
    _run_colmap_with_gpu_fallback(feature_cmd, "--SiftExtraction.use_gpu", progress, "fine_colmap_features", 26)

    progress("fine_colmap_matching", 30, f"running COLMAP {matcher_command}")
    match_cmd = _build_matcher_command(executable, matcher_command, database_path, use_gpu=gpu_flag, gpu_index=gpu_index)
    _run_colmap_with_gpu_fallback(match_cmd, "--SiftMatching.use_gpu", progress, "fine_colmap_matching", 32)

    progress("fine_colmap_mapping", 36, f"running COLMAP {mapper_command}")
    _run_colmap_command(
        _build_mapper_command(capabilities, mapper_command, database_path, images_dir, sparse_dir, use_gpu=use_gpu),
        stage="fine_colmap_mapping",
        progress=progress,
        progress_value=38,
    )

    recon_path = select_best_colmap_model(sparse_dir)
    if recon_path is None:
        raise FineFailure("COLMAP_RECONSTRUCTION_FAILED", "COLMAP did not produce sparse/0")

    progress("fine_colmap_undistort", 40, "running COLMAP image undistorter")
    _run_colmap_command(
        [
            executable,
            "image_undistorter",
            "--image_path",
            str(images_dir),
            "--input_path",
            str(recon_path),
            "--output_path",
            str(undistorted_dir),
            "--output_type",
            "COLMAP",
        ],
        stage="fine_colmap_undistort",
        progress=progress,
        progress_value=40,
    )

    _normalize_colmap_sparse_layout(undistorted_dir)
    sparse_zero = ensure_colmap_sparse_zero(undistorted_dir / "sparse")
    reconstruction = _load_reconstruction(sparse_zero)
    if reconstruction is not None:
        validate_colmap_pinhole_scene(reconstruction)

    registered = _registered_image_count(reconstruction, len(files))
    registered_ratio = registered / max(len(files), 1)
    threshold = _resolve_colmap_min_registered_ratio(min_registered_ratio)
    if threshold is not None and registered_ratio < threshold:
        raise FineFailure(
            "COLMAP_RECONSTRUCTION_INCOMPLETE",
            f"COLMAP registered {registered}/{len(files)} images ({registered_ratio:.1%}), below threshold {threshold:.1%}",
        )

    filtered_point_count = write_filtered_sparse_points_ply(reconstruction, sparse_zero / "points3D.ply") if reconstruction is not None else None
    raw_point_count = len(getattr(reconstruction, "points3D", {}) or {}) if reconstruction is not None else 0
    point_count = int(filtered_point_count) if filtered_point_count is not None else raw_point_count

    return SceneBuildResult(
        scene_dir=undistorted_dir,
        backend=backend,
        image_count=len(files),
        registered_images=registered,
        point_count=point_count,
        metrics={
            "sfm_backend": backend,
            "colmap_backend": backend,
            "sfm_elapsed_seconds": round(time.monotonic() - started, 3),
            "sfm_registered_images": registered,
            "sfm_registered_ratio": registered_ratio,
            "sfm_min_registered_ratio": threshold,
            "sfm_sparse_points": point_count,
            "sfm_sparse_points_raw": raw_point_count,
            "sfm_sparse_points_filtered": filtered_point_count,
            "sfm_undistorted": True,
            "colmap_matcher": matcher_command,
            "colmap_mapper": mapper_command,
            "colmap_use_gpu": use_gpu,
            "colmap_gpu_index": gpu_index,
            "colmap_sift_max_num_features": max_num_features,
            "colmap_max_image_size": max_image_size,
        },
    )


def _resolve_matcher_command(matcher_policy: str | None, image_count: int) -> str:
    matcher = str(matcher_policy or "auto").strip().lower()
    if matcher == "auto":
        matcher = "exhaustive" if image_count <= 80 else "sequential"
    if matcher == "exhaustive":
        return "exhaustive_matcher"
    if matcher == "sequential":
        return "sequential_matcher"
    if matcher in {"vocab_tree", "spatial"}:
        raise FineFailure(
            "COLMAP_MATCHER_UNSUPPORTED",
            f"COLMAP matcher '{matcher}' requires additional configuration and is not enabled for this pipeline",
        )
    raise FineFailure("COLMAP_MATCHER_UNSUPPORTED", f"Unsupported COLMAP matcher: {matcher}")


def _build_matcher_command(
    executable: str,
    matcher_command: str,
    database_path: Path,
    *,
    use_gpu: str,
    gpu_index: str | None,
) -> list[str]:
    command = [
        executable,
        matcher_command,
        "--database_path",
        str(database_path),
        "--SiftMatching.use_gpu",
        use_gpu,
    ]
    if gpu_index:
        command.extend(["--SiftMatching.gpu_index", gpu_index])
    return command


def _build_mapper_command(
    capabilities: ColmapCapabilities,
    mapper_command: str,
    database_path: Path,
    images_dir: Path,
    sparse_dir: Path,
    *,
    use_gpu: bool,
) -> list[str]:
    command = [
        capabilities.executable,
        mapper_command,
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(sparse_dir),
    ]
    if use_gpu and mapper_command == GLOBAL_COLMAP_COMMAND:
        command.extend(_global_mapper_gpu_options(capabilities.help_text))
    return command


def _global_mapper_gpu_options(help_text: str) -> list[str]:
    for option in ("--Mapper.ba_use_gpu", "--GlobalMapper.ba_use_gpu"):
        if option in help_text:
            return [option, "1"]
    return []


def _require_colmap_command(capabilities: ColmapCapabilities, command: str, context: str) -> None:
    if command in capabilities.commands or _help_mentions_command(capabilities.help_text, command):
        return
    raise FineFailure("COLMAP_CLI_UNSUPPORTED", f"{context} requires COLMAP command '{command}'")


def _run_colmap_with_gpu_fallback(command: list[str], gpu_option: str, progress: Progress, stage: str, progress_value: int) -> None:
    try:
        _run_colmap_command(command, stage=stage, progress=progress, progress_value=progress_value)
    except FineFailure as exc:
        compat_command = _colmap_option_family_compat_command(command, exc.message)
        if compat_command != command:
            try:
                print("[colmap-cli] option family unsupported, retrying with COLMAP generic option names", flush=True)
                _run_colmap_command(compat_command, stage=stage, progress=progress, progress_value=progress_value)
                return
            except FineFailure as compat_exc:
                exc = compat_exc
                command = compat_command
                gpu_option = _compat_gpu_option(gpu_option)
        if not _bool_env("DBULR_COLMAP_ALLOW_CPU_FALLBACK", True):
            raise exc
        fallback = _override_option(command, gpu_option, "0")
        if fallback == command:
            raise exc
        print("[colmap-cli] GPU command failed, retrying on CPU", flush=True)
        _run_colmap_command(fallback, stage=stage, progress=progress, progress_value=progress_value)


def _run_colmap_command(command: list[str], *, stage: str, progress: Progress, progress_value: int) -> None:
    tail: deque[str] = deque(maxlen=80)
    print("[colmap-cli] command " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        tail.append(text)
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        message = "\n".join(tail) or f"{command[1]} exited with {return_code}"
        raise FineFailure("COLMAP_COMMAND_FAILED", message)
    progress(stage, progress_value, f"completed COLMAP {command[1]}")


def _normalize_colmap_sparse_layout(colmap_root: Path) -> None:
    sparse_root = colmap_root / "sparse"
    nested_sparse = sparse_root / "0"
    if nested_sparse.exists():
        return
    if not sparse_root.exists():
        raise FineFailure("COLMAP_UNDISTORT_FAILED", f"COLMAP did not create sparse outputs under {sparse_root}")

    has_flat_binary = all((sparse_root / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin"))
    has_flat_text = all((sparse_root / name).exists() for name in ("cameras.txt", "images.txt", "points3D.txt"))
    if not (has_flat_binary or has_flat_text):
        available = sorted(path.name for path in sparse_root.glob("*"))
        raise FineFailure("COLMAP_UNDISTORT_FAILED", f"unexpected COLMAP sparse layout: {available}")

    nested_sparse.mkdir(parents=True, exist_ok=True)
    for path in sparse_root.glob("*"):
        if path.is_file():
            shutil.copy2(path, nested_sparse / path.name)


def _override_option(command: list[str], key: str, value: str) -> list[str]:
    output = list(command)
    for index in range(len(output) - 1):
        if output[index] == key:
            output[index + 1] = value
            return output
    return output


def _colmap_option_family_compat_command(command: list[str], message: str) -> list[str]:
    if "unrecognised option" not in message and "unrecognized option" not in message:
        return command
    replacements = {
        "--SiftExtraction.use_gpu": "--FeatureExtraction.use_gpu",
        "--SiftExtraction.gpu_index": "--FeatureExtraction.gpu_index",
        "--SiftExtraction.max_image_size": "--FeatureExtraction.max_image_size",
        "--SiftMatching.use_gpu": "--FeatureMatching.use_gpu",
        "--SiftMatching.gpu_index": "--FeatureMatching.gpu_index",
    }
    if not any(option in message for option in replacements):
        return command
    return [replacements.get(item, item) for item in command]


def _compat_gpu_option(option: str) -> str:
    return {
        "--SiftExtraction.use_gpu": "--FeatureExtraction.use_gpu",
        "--SiftMatching.use_gpu": "--FeatureMatching.use_gpu",
    }.get(option, option)


def _load_reconstruction(path: Path):
    try:
        import pycolmap

        return pycolmap.Reconstruction(path)
    except Exception:
        return None


def _registered_image_count(reconstruction: Any, fallback: int) -> int:
    if reconstruction is None:
        return fallback
    return len(getattr(reconstruction, "images", {}) or {})


def _resolve_colmap_min_registered_ratio(override: float | None) -> float | None:
    if override is None:
        return None
    return max(0.30, min(0.95, float(override)))


def _help_mentions_command(help_text: str, command: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(command)}(?![A-Za-z0-9_])"
    return re.search(pattern, help_text) is not None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
