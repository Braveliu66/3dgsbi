from __future__ import annotations

import math
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
from app.fine.types import FineFailure
from app.preview.utils import image_files
from app.resources import collect_gpu


Progress = Callable[[str, int, str], None]
REQUIRED_COLMAP_COMMANDS = {"global_mapper", "hierarchical_mapper"}


@dataclass(slots=True)
class ColmapPolicy:
    name: str
    scene_type: str
    input_type: str
    n_images: int
    feature_type: str
    matcher_type: str
    matchers: list[str]
    mapper: str
    max_num_features: int
    max_image_size: int
    max_num_matches: int
    sequential_overlap: int
    vocab_num_images: int
    spatial_neighbors: int
    guided_matching: bool
    estimate_affine_shape: bool
    domain_size_pooling: bool
    use_gpu: bool
    gpu_index: str
    ba_use_gpu: bool


@dataclass(slots=True)
class ColmapCapabilities:
    executable: str
    help_text: str
    commands: set[str]
    has_aliked_lightglue: bool
    has_sift_lightglue: bool


def build_colmap_cli_scene(
    input_dir: Path,
    scene_dir: Path,
    *,
    scene_type: str,
    input_type: str,
    quality_mode: str = "auto",
    capture_order: str = "auto",
    prefer_gpu: bool = True,
    gpu_index: str | None = None,
    min_registered_ratio: float | None = None,
    progress: Progress,
) -> SceneBuildResult:
    files = image_files(input_dir)
    if len(files) < 3:
        raise FineFailure("INSUFFICIENT_IMAGES", "COLMAP initialization requires at least 3 images")

    capabilities = detect_colmap_capabilities()
    free_vram_gb, resolved_gpu_index = current_free_vram_gb()
    if gpu_index:
        resolved_gpu_index = gpu_index
    policies = retry_policies(
        scene_type=scene_type,
        input_type=input_type,
        n_images=len(files),
        free_vram_gb=free_vram_gb,
        quality_mode=quality_mode,
        capture_order=capture_order,
        prefer_gpu=prefer_gpu,
        gpu_index=resolved_gpu_index,
        capabilities=capabilities,
    )

    last_failure: FineFailure | None = None
    for attempt, policy in enumerate(policies, start=1):
        try:
            result = _run_colmap_attempt(
                capabilities.executable,
                input_dir,
                scene_dir,
                policy=policy,
                min_registered_ratio=min_registered_ratio,
                progress=progress,
            )
            result.metrics["colmap_attempts"] = attempt
            result.metrics["colmap_retry_profiles"] = [item.name for item in policies[:attempt]]
            return result
        except FineFailure as exc:
            last_failure = exc
            if exc.code not in {
                "COLMAP_COMMAND_FAILED",
                "COLMAP_RECONSTRUCTION_FAILED",
                "COLMAP_RECONSTRUCTION_INCOMPLETE",
                "COLMAP_UNDISTORT_FAILED",
            }:
                raise
            print(
                "[colmap-cli] attempt failed "
                f"attempt={attempt} policy={policy.name} code={exc.code} message={exc.message}",
                flush=True,
            )
    assert last_failure is not None
    raise last_failure


def retry_policies(
    *,
    scene_type: str,
    input_type: str,
    n_images: int,
    free_vram_gb: float,
    quality_mode: str,
    capture_order: str,
    prefer_gpu: bool,
    gpu_index: str,
    capabilities: ColmapCapabilities,
) -> list[ColmapPolicy]:
    policies = [
        resolve_colmap_policy(
            scene_type=scene_type,
            input_type=input_type,
            n_images=n_images,
            free_vram_gb=free_vram_gb,
            quality_mode=quality_mode,
            capture_order=capture_order,
            prefer_gpu=prefer_gpu,
            gpu_index=gpu_index,
            retry_profile="primary",
            feature_type="SIFT",
            matcher_type="SIFT_BRUTEFORCE",
        ),
        resolve_colmap_policy(
            scene_type=scene_type,
            input_type=input_type,
            n_images=n_images,
            free_vram_gb=free_vram_gb,
            quality_mode="quality",
            capture_order=capture_order,
            prefer_gpu=prefer_gpu,
            gpu_index=gpu_index,
            retry_profile="high_recall",
            feature_type="SIFT",
            matcher_type="SIFT_BRUTEFORCE",
        ),
    ]
    if capabilities.has_aliked_lightglue:
        policies.append(
            resolve_colmap_policy(
                scene_type=scene_type,
                input_type=input_type,
                n_images=n_images,
                free_vram_gb=free_vram_gb,
                quality_mode="quality",
                capture_order=capture_order,
                prefer_gpu=prefer_gpu,
                gpu_index=gpu_index,
                retry_profile="aliked_lightglue",
                feature_type="ALIKED_N16ROT",
                matcher_type="ALIKED_LIGHTGLUE",
            )
        )
    elif capabilities.has_sift_lightglue:
        policies.append(
            resolve_colmap_policy(
                scene_type=scene_type,
                input_type=input_type,
                n_images=n_images,
                free_vram_gb=free_vram_gb,
                quality_mode="quality",
                capture_order=capture_order,
                prefer_gpu=prefer_gpu,
                gpu_index=gpu_index,
                retry_profile="sift_lightglue",
                feature_type="SIFT",
                matcher_type="SIFT_LIGHTGLUE",
            )
        )
    return policies[:3]


def resolve_colmap_policy(
    *,
    scene_type: str,
    input_type: str,
    n_images: int,
    free_vram_gb: float,
    quality_mode: str,
    capture_order: str,
    prefer_gpu: bool,
    gpu_index: str,
    retry_profile: str = "primary",
    feature_type: str = "SIFT",
    matcher_type: str = "SIFT_BRUTEFORCE",
) -> ColmapPolicy:
    scene = "outdoor" if scene_type == "outdoor" else "indoor"
    sequential_input = input_type == "video" or capture_order == "sequential"
    quality = quality_mode if quality_mode in {"quality", "speed"} else ("quality" if scene == "indoor" else "speed")

    if scene == "indoor":
        max_features, max_size, mapper = _indoor_feature_size_mapper(n_images)
        matchers = _indoor_matchers(n_images, sequential_input)
        guided = True
        affine = True
        dsp = True
        overlap = _seq_overlap(scene, n_images, input_type)
    else:
        max_features, max_size, mapper = _outdoor_feature_size_mapper(n_images)
        matchers = _outdoor_matchers(n_images, sequential_input)
        guided = False
        affine = False
        dsp = False
        overlap = _seq_overlap(scene, n_images, input_type)

    if retry_profile != "primary":
        guided = True
        max_features = min(32768, int(max_features * 1.35))
        max_size = min(4096, int(max_size * 1.15))
        if scene == "indoor" and "transitive" not in matchers and n_images <= 1200:
            matchers = [*matchers, "transitive"]
        overlap = int(overlap * 1.35)

    if quality == "quality":
        max_features = min(32768, int(max_features * 1.15))
        max_size = min(4096, int(max_size * 1.10))
        guided = guided or scene == "indoor"
    elif quality == "speed":
        max_features = max(3000, int(max_features * 0.85))
        max_size = max(1200, int(max_size * 0.90))

    max_matches = choose_max_num_matches(free_vram_gb, scene, n_images)
    return ColmapPolicy(
        name=retry_profile,
        scene_type=scene,
        input_type=input_type,
        n_images=n_images,
        feature_type=feature_type,
        matcher_type=matcher_type,
        matchers=matchers,
        mapper=mapper,
        max_num_features=max_features,
        max_image_size=max_size,
        max_num_matches=max_matches,
        sequential_overlap=max(4, overlap),
        vocab_num_images=80 if scene == "indoor" else 100,
        spatial_neighbors=50 if scene == "indoor" else 80,
        guided_matching=guided,
        estimate_affine_shape=affine,
        domain_size_pooling=dsp,
        use_gpu=bool(prefer_gpu),
        gpu_index=gpu_index or "-1",
        ba_use_gpu=False,
    )


def choose_max_num_matches(free_vram_gb: float, scene_type: str, n_images: int) -> int:
    if free_vram_gb <= 0:
        return 10000 if scene_type == "indoor" else 8000
    reserve = 0.45 if free_vram_gb <= 12 else 0.60
    budget = free_vram_gb * reserve * (1024**3)
    matches = int((-1024 + math.sqrt(1024**2 + 16 * budget)) / 8)
    if scene_type == "indoor":
        return max(8000, min(matches, 24000))
    if n_images > 3000:
        return max(6000, min(matches, 14000))
    return max(8000, min(matches, 18000))


def detect_colmap_capabilities() -> ColmapCapabilities:
    executable = shutil.which("colmap")
    if not executable:
        raise FineFailure("COLMAP_CLI_UNAVAILABLE", "colmap executable was not found in PATH")
    try:
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
    except Exception as exc:
        raise FineFailure("COLMAP_CLI_UNAVAILABLE", f"failed to run colmap help: {exc}") from exc
    help_text = completed.stdout or ""
    if completed.returncode != 0:
        raise FineFailure("COLMAP_CLI_UNAVAILABLE", help_text.strip() or f"colmap help exited with {completed.returncode}")
    commands = {match.group(1) for match in re.finditer(r"^\s+([a-z_]+)(?:\s|$)", help_text, flags=re.MULTILINE)}
    commands.update(command for command in REQUIRED_COLMAP_COMMANDS if command in help_text)
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


def current_free_vram_gb() -> tuple[float, str]:
    gpu = collect_gpu()
    gpus = gpu.get("gpus") or []
    if not gpu.get("available") or not gpus:
        return 0.0, "-1"
    best = max(gpus, key=lambda item: float(item.get("memory_total", 0) or 0) - float(item.get("memory_used", 0) or 0))
    total = float(best.get("memory_total", 0) or 0)
    used = float(best.get("memory_used", 0) or 0)
    free_mib = max(0.0, total - used)
    return free_mib / 1024.0, str(best.get("index", 0))


def find_sparse_models(root: Path) -> list[Path]:
    if (root / "images.bin").exists():
        return [root]
    models: list[Path] = []
    for path in sorted(root.rglob("images.bin")):
        parent = path.parent
        if (parent / "cameras.bin").exists() and (parent / "points3D.bin").exists():
            models.append(parent)
    return models


def materialize_chunk_scenes(chunk_models: list[Path], global_scene_dir: Path, output_root: Path) -> list[Path]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    global_images_dir = global_scene_dir / "images"
    chunks: list[Path] = []
    for index, model_dir in enumerate(chunk_models):
        chunk_dir = output_root / f"chunk_{index:03d}"
        sparse_dir = chunk_dir / "sparse" / "0"
        images_dir = chunk_dir / "images"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        for item in model_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, sparse_dir / item.name)
        names = read_reconstruction_image_names(model_dir)
        if not names:
            names = [path.name for path in image_files(global_images_dir)]
        for name in sorted(set(names)):
            source = global_images_dir / name
            if not source.exists():
                continue
            target = images_dir / name
            link_or_copy(source, target)
        chunks.append(chunk_dir)
    return chunks


def read_reconstruction_image_names(model_dir: Path) -> list[str]:
    try:
        import pycolmap

        reconstruction = pycolmap.Reconstruction(model_dir)
        return sorted(str(image.name) for image in reconstruction.images.values())
    except Exception:
        return []


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _run_colmap_attempt(
    executable: str,
    input_dir: Path,
    scene_dir: Path,
    *,
    policy: ColmapPolicy,
    min_registered_ratio: float | None,
    progress: Progress,
) -> SceneBuildResult:
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    workspace = scene_dir / "colmap_workspace"
    workspace_images = workspace / "images"
    sparse_dir = workspace / "sparse"
    workspace_images.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    files = image_files(input_dir)
    for index, path in enumerate(files):
        shutil.copy2(path, workspace_images / f"{index:06d}.jpg")

    database_path = workspace / "database.db"
    started = time.monotonic()
    progress("fine_colmap_features", 24, f"extracting COLMAP CLI features from {len(files)} images")
    feature_command = [
        executable,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(workspace_images),
        "--ImageReader.camera_model",
        "PINHOLE",
        "--ImageReader.single_camera",
        "0",
        "--FeatureExtraction.type",
        policy.feature_type,
        "--FeatureExtraction.use_gpu",
        "1" if policy.use_gpu else "0",
        "--FeatureExtraction.gpu_index",
        policy.gpu_index,
        "--FeatureExtraction.max_image_size",
        str(policy.max_image_size),
    ]
    if policy.feature_type.startswith("ALIKED"):
        feature_command.extend(["--AlikedExtraction.max_num_features", str(policy.max_num_features)])
    else:
        feature_command.extend(
            [
                "--SiftExtraction.max_num_features",
                str(policy.max_num_features),
                "--SiftExtraction.estimate_affine_shape",
                "1" if policy.estimate_affine_shape else "0",
                "--SiftExtraction.domain_size_pooling",
                "1" if policy.domain_size_pooling else "0",
            ]
        )
    _run_colmap_command(feature_command, stage="fine_colmap_features", progress=progress, progress_value=26)

    for matcher in policy.matchers:
        progress("fine_colmap_matching", 30, f"running COLMAP {matcher}_matcher")
        matcher_command = [
            executable,
            f"{matcher}_matcher",
            "--database_path",
            str(database_path),
            "--FeatureMatching.type",
            policy.matcher_type,
            "--FeatureMatching.use_gpu",
            "1" if policy.use_gpu else "0",
            "--FeatureMatching.gpu_index",
            policy.gpu_index,
            "--FeatureMatching.max_num_matches",
            str(policy.max_num_matches),
        ]
        if matcher == "sequential":
            matcher_command.extend(
                [
                    "--SequentialMatching.overlap",
                    str(policy.sequential_overlap),
                    "--SequentialMatching.loop_detection",
                    "1",
                ]
            )
        elif matcher == "vocab_tree":
            matcher_command.extend(["--VocabTreeMatching.num_images", str(policy.vocab_num_images)])
        elif matcher == "spatial":
            matcher_command.extend(["--SpatialMatching.max_num_neighbors", str(policy.spatial_neighbors)])
        if policy.guided_matching and matcher != "transitive":
            matcher_command.extend(["--FeatureMatching.guided_matching", "1"])
        _run_colmap_command(matcher_command, stage="fine_colmap_matching", progress=progress, progress_value=32)

    mapper_database = database_path
    if policy.mapper == "global":
        mapper_database = workspace / "database_global.db"
        shutil.copy2(database_path, mapper_database)
        _run_colmap_command(
            [executable, "view_graph_calibrator", "--database_path", str(mapper_database)],
            stage="fine_colmap_view_graph_calibrator",
            progress=progress,
            progress_value=34,
        )

    progress("fine_colmap_mapping", 36, f"running COLMAP {policy.mapper} mapper")
    if policy.mapper == "hierarchical":
        mapper_command = [
            executable,
            "hierarchical_mapper",
            "--database_path",
            str(mapper_database),
            "--image_path",
            str(workspace_images),
            "--output_path",
            str(sparse_dir),
        ]
    elif policy.mapper == "global":
        mapper_command = [
            executable,
            "global_mapper",
            "--database_path",
            str(mapper_database),
            "--image_path",
            str(workspace_images),
            "--output_path",
            str(sparse_dir),
        ]
    else:
        mapper_command = [
            executable,
            "mapper",
            "--database_path",
            str(mapper_database),
            "--image_path",
            str(workspace_images),
            "--output_path",
            str(sparse_dir),
            "--Mapper.ba_use_gpu",
            "1" if policy.ba_use_gpu else "0",
        ]
        if policy.scene_type == "outdoor" and policy.n_images > 3000:
            mapper_command.extend(["--Mapper.ba_global_ignore_redundant_points3D", "1"])
    _run_colmap_command(mapper_command, stage="fine_colmap_mapping", progress=progress, progress_value=38)

    recon_path = select_best_colmap_model(sparse_dir)
    if recon_path is None:
        raise FineFailure("COLMAP_RECONSTRUCTION_FAILED", "COLMAP CLI did not produce a sparse reconstruction")

    analysis = analyze_model(executable, recon_path, len(files))
    threshold = _resolve_colmap_min_registered_ratio(policy.scene_type, len(files), min_registered_ratio)
    registered = int(analysis["registered_images"])
    registered_ratio = float(analysis["registered_ratio"])
    if registered < max(3, min(10, len(files))) or registered_ratio < threshold:
        raise FineFailure(
            "COLMAP_RECONSTRUCTION_INCOMPLETE",
            f"COLMAP CLI registered {registered}/{len(files)} images ({registered_ratio:.1%}), below threshold {threshold:.1%}",
        )

    progress("fine_colmap_undistort", 40, "undistorting COLMAP CLI images")
    _run_colmap_command(
        [
            executable,
            "image_undistorter",
            "--image_path",
            str(workspace_images),
            "--input_path",
            str(recon_path),
            "--output_path",
            str(scene_dir),
            "--output_type",
            "COLMAP",
        ],
        stage="fine_colmap_undistort",
        progress=progress,
        progress_value=40,
    )
    sparse_zero = ensure_colmap_sparse_zero(scene_dir / "sparse")
    reconstruction = _load_reconstruction(sparse_zero)
    if reconstruction is not None:
        validate_colmap_pinhole_scene(reconstruction)
    elapsed = round(time.monotonic() - started, 3)
    point_count = int(analysis.get("points3D") or 0)
    return SceneBuildResult(
        scene_dir=scene_dir,
        backend="colmap_cli",
        image_count=len(files),
        registered_images=registered,
        point_count=point_count,
        metrics={
            "sfm_backend": "colmap_cli",
            "colmap_backend": "colmap_cli",
            "sfm_elapsed_seconds": elapsed,
            "sfm_registered_images": registered,
            "sfm_registered_ratio": registered_ratio,
            "sfm_min_registered_ratio": threshold,
            "sfm_sparse_points": point_count,
            "sfm_undistorted": True,
            "colmap_policy": policy.name,
            "colmap_matchers": policy.matchers,
            "colmap_mapper": policy.mapper,
            "colmap_registered_ratio": registered_ratio,
            "colmap_feature_type": policy.feature_type,
            "colmap_feature_matching_type": policy.matcher_type,
            "colmap_sift_max_num_features": policy.max_num_features,
            "colmap_max_image_size": policy.max_image_size,
            "colmap_max_num_matches": policy.max_num_matches,
            "colmap_sequential_overlap": policy.sequential_overlap,
            "colmap_guided_matching": policy.guided_matching,
            "colmap_estimate_affine_shape": policy.estimate_affine_shape,
            "colmap_domain_size_pooling": policy.domain_size_pooling,
            "colmap_gpu_index": policy.gpu_index,
            **{f"colmap_analyzer_{key}": value for key, value in analysis.items() if key not in {"registered_images", "registered_ratio", "points3D"}},
        },
    )


def analyze_model(executable: str, model_path: Path, total_images: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    reconstruction = _load_reconstruction(model_path)
    if reconstruction is not None:
        metrics["registered_images"] = len(getattr(reconstruction, "images", {}) or {})
        metrics["registered_ratio"] = metrics["registered_images"] / max(total_images, 1)
        metrics["points3D"] = len(getattr(reconstruction, "points3D", {}) or {})
    completed = subprocess.run(
        [executable, "model_analyzer", "--path", str(model_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    print(completed.stdout or "", flush=True)
    parsed = parse_model_analyzer_output(completed.stdout or "")
    metrics.update(parsed)
    metrics.setdefault("registered_images", int(parsed.get("registered_images", 0) or 0))
    metrics.setdefault("registered_ratio", metrics["registered_images"] / max(total_images, 1))
    metrics.setdefault("points3D", int(parsed.get("points3D", 0) or 0))
    return metrics


def parse_model_analyzer_output(output: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    patterns = {
        "registered_images": r"(?:Registered images|Images)\s*:\s*([0-9]+)",
        "points3D": r"(?:Points|3D points|Number of points)\s*:\s*([0-9]+)",
        "mean_track_length": r"Mean track length\s*:\s*([0-9.]+)",
        "mean_reprojection_error": r"Mean reprojection error\s*:\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            value = match.group(1)
            metrics[key] = float(value) if "." in value else int(value)
    return metrics


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
        tail.append(line.rstrip())
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        message = "\n".join(tail) or f"{command[1]} exited with {return_code}"
        raise FineFailure("COLMAP_COMMAND_FAILED", message)
    progress(stage, progress_value, f"completed COLMAP {command[1]}")


def _load_reconstruction(path: Path):
    try:
        import pycolmap

        return pycolmap.Reconstruction(path)
    except Exception:
        return None


def _resolve_colmap_min_registered_ratio(scene_type: str, image_count: int, override: float | None) -> float:
    if override is not None:
        return max(0.30, min(0.95, float(override)))
    if scene_type == "indoor":
        if image_count <= 80:
            return 0.95
        if image_count <= 300:
            return 0.90
        if image_count <= 1000:
            return 0.85
        return 0.80
    if image_count <= 150:
        return 0.90
    if image_count <= 3000:
        return 0.80
    return 0.75


def _indoor_feature_size_mapper(n_images: int) -> tuple[int, int, str]:
    if n_images <= 80:
        return 16000, 4000, "mapper"
    if n_images <= 300:
        return 12000, 3600, "mapper"
    if n_images <= 1000:
        return 10000, 3200, "mapper"
    if n_images <= 3000:
        return 8192, 2800, "mapper"
    return 6000, 2400, "hierarchical"


def _outdoor_feature_size_mapper(n_images: int) -> tuple[int, int, str]:
    if n_images <= 150:
        return 8192, 3200, "mapper"
    if n_images <= 800:
        return 8192, 3000, "mapper"
    if n_images <= 3000:
        return 6000, 2600, "global"
    if n_images <= 8000:
        return 4096, 2400, "hierarchical"
    return 4096, 2200, "hierarchical"


def _indoor_matchers(n_images: int, sequential_input: bool) -> list[str]:
    if n_images <= 300 and not sequential_input:
        return ["exhaustive"]
    if n_images <= 1000:
        return ["sequential"]
    return ["sequential", "vocab_tree"]


def _outdoor_matchers(n_images: int, sequential_input: bool) -> list[str]:
    if n_images <= 150 and not sequential_input:
        return ["exhaustive"]
    if sequential_input:
        return ["sequential"]
    if n_images > 800:
        return ["vocab_tree"]
    return ["sequential"]


def _seq_overlap(scene_type: str, n_images: int, input_type: str) -> int:
    if scene_type == "indoor":
        if n_images <= 300:
            return 55 if input_type == "video" else 45
        if n_images <= 1000:
            return 35
        return 25
    if n_images <= 800:
        return 25 if input_type == "video" else 20
    if n_images <= 3000:
        return 18
    return 12
