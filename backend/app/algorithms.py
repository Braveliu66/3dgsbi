from __future__ import annotations

import os
import subprocess
import shutil
import importlib
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings, resolve_local_path
from app.models import AlgorithmRegistry, WorkerHeartbeat
from app.preview.io.spz import locate_spark_cli
from app.preview.utils import prepend_sys_path
from app.preview.weights import weight_file_status
from app.resources import collect_gpu, python_info, torch_info


VENDOR_ROOT = Path(__file__).resolve().parent / "preview" / "vendor"
FINE_ROOT = Path(__file__).resolve().parent / "fine"


ALGORITHM_DEPENDENCIES: dict[str, dict[str, Any]] = {
    "LiteVGGT": {
        "python_modules": [
            "torch",
            "torchvision",
            "transformer_engine.pytorch",
            "einops",
            "cv2",
            "PIL",
            "numpy",
            "huggingface_hub",
        ],
        "executables": ["node"],
        "requires_cuda": True,
    },
    "Spark SPZ": {
        "python_modules": [],
        "executables": ["node"],
        "requires_cuda": False,
    },
    "DashDeblurGroupGS Fine": {
        "python_modules": [
            "torch",
            "torchvision",
            "pycolmap",
            "cv2",
            "PIL",
            "numpy",
            "plyfile",
            "scipy",
            "tqdm",
            "lpips",
            "pytorch_msssim",
            "sklearn",
            "diff_gaussian_rasterization",
            "simple_knn._C",
        ],
        "executables": ["colmap", "ffmpeg", "node", "git"],
        "requires_cuda": True,
    },
}


ALGORITHMS: list[dict[str, Any]] = [
    {
        "name": "LiteVGGT",
        "repo_url": "https://github.com/GarlicBa/LiteVGGT-repo",
        "license": "MIT",
        "commit_hash_setting": "litevggt_repo_commit",
        "local_path": VENDOR_ROOT / "litevggt",
        "enabled": True,
        "weight_paths": ["litevggt/te_dict.pt"],
        "commands": {},
        "source_type": "bundled",
        "license_notice": "MIT; bundled key preview code from fixed upstream commit.",
        "notes": "Bundled LiteVGGT direct point cloud preview.",
    },
    {
        "name": "Spark SPZ",
        "repo_url": "https://github.com/sparkjsdev/spark",
        "license": "MIT",
        "commit_hash_setting": "spark_repo_commit",
        "local_path": VENDOR_ROOT / "SPARK_LICENSE",
        "enabled": True,
        "weight_paths": [],
        "commands": {},
        "source_type": "bundled",
        "license_notice": "MIT; worker uses @sparkjsdev/spark transcodeSpz for SPZ conversion.",
        "notes": "Spark-readable SPZ conversion/validation.",
    },
    {
        "name": "DashDeblurGroupGS Fine",
        "repo_url": "https://github.com/benhenryL/Deblurring-3D-Gaussian-Splatting",
        "license": "Research",
        "commit_hash_setting": None,
        "local_path": FINE_ROOT / "runner.py",
        "enabled": True,
        "weight_paths": [],
        "commands": {},
        "source_type": "bundled",
        "license_notice": "Fine reconstruction uses system COLMAP/pycolmap plus the embedded DashDeblurGroupGS worker trainer.",
        "notes": "Default fine reconstruction pipeline: image or video-frame normalization, existing COLMAP scene construction, embedded DashDeblurGroupGS training, final PLY/SPZ export.",
    },
]


def seed_algorithm_registry(db: Session, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    model_root = Path(settings.model_cache_dir)
    for entry in ALGORITHMS:
        existing = db.scalar(select(AlgorithmRegistry).where(AlgorithmRegistry.name == entry["name"]))
        commit_hash = getattr(settings, entry["commit_hash_setting"]) if entry["commit_hash_setting"] else None
        local_path = str(Path(entry["local_path"]).resolve()) if entry["local_path"] else None
        weight_paths = [str(model_root / path) for path in entry["weight_paths"]]
        commands = entry["commands"]
        if existing:
            existing.repo_url = entry["repo_url"]
            existing.license = entry["license"]
            existing.commit_hash = commit_hash
            existing.local_path = local_path
            existing.weight_paths = weight_paths
            existing.commands = commands
            existing.notes = entry["notes"]
            existing.enabled = bool(entry["enabled"])
            existing.source_type = entry.get("source_type", "bundled")
        else:
            db.add(
                AlgorithmRegistry(
                    name=entry["name"],
                    repo_url=entry["repo_url"],
                    license=entry["license"],
                    commit_hash=commit_hash,
                    local_path=local_path,
                    weight_paths=weight_paths,
                    commands=commands,
                    notes=entry["notes"],
                    enabled=bool(entry["enabled"]),
                    source_type=entry.get("source_type", "bundled"),
                )
            )
    known = {entry["name"] for entry in ALGORITHMS}
    for stale in db.scalars(select(AlgorithmRegistry).where(AlgorithmRegistry.name.not_in(known))).all():
        db.delete(stale)
    db.commit()


def normalize_preview_pipeline(value: str | None, default: str = "litevggt_spz") -> str:
    normalized = (value or default).strip().lower()
    aliases = {
        "litevggt_spark": "litevggt_spz",
        "litevggt_spz": "litevggt_spz",
        "direct": "litevggt_spz",
    }
    return aliases.get(normalized, normalized)


def runtime_preflight(db: Session, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    algorithms = []
    errors: list[str] = []
    warnings: list[str] = []
    fine_workers = fine_worker_status(db)
    preview_workers = preview_worker_status(db)
    for item in db.scalars(select(AlgorithmRegistry).order_by(AlgorithmRegistry.name)).all():
        issues = []
        module_status = bundled_module_status(item.name)
        dependency_status = algorithm_dependency_status(item.name)
        worker_runtime_available = worker_runtime_for_algorithm(item.name, preview_workers, fine_workers)
        weights_ready = True
        weight_statuses = []
        extensions_ready = True
        spz_converter_ready = spark_converter_status()
        if item.enabled:
            if item.local_path and not Path(item.local_path).exists():
                issues.append(f"bundled source missing: {item.local_path}")
            for weight_path in item.weight_paths or []:
                status = weight_file_status(Path(weight_path))
                weight_statuses.append(status)
                if not status["exists"]:
                    issues.append(f"weight missing: {weight_path}")
                    weights_ready = False
            if not worker_runtime_available:
                for missing in dependency_status["missing"]:
                    issues.append(f"dependency missing: {missing}")
            if not module_status.get("available", True):
                issues.append(f"bundled module import failed: {module_status.get('error')}")
            if item.name == "DashDeblurGroupGS Fine":
                fine_runtime = fine_runtime_status()
                extensions_ready = bool(fine_runtime["available"] or fine_workers["available"])
                if not extensions_ready:
                    issues.append("worker-fine heartbeat missing and backend fine runtime unavailable")
            if item.name == "Spark SPZ":
                if not spz_converter_ready["available"]:
                    issues.append(f"Spark SPZ converter unavailable: {spz_converter_ready['error']}")
        ready = item.enabled and not issues
        if item.enabled and issues:
            warnings.extend([f"{item.name}: {issue}" for issue in issues])
        algorithms.append(
            {
                "name": item.name,
                "enabled": item.enabled,
                "ready": ready,
                "repo_url": item.repo_url,
                "license": item.license,
                "commit_hash": item.commit_hash,
                "local_path": item.local_path,
                "weight_paths": item.weight_paths or [],
                "weights": weight_statuses,
                "commands": item.commands or {},
                "source_type": item.source_type,
                "bundled": item.source_type == "bundled",
                "license_notice": license_notice_for(item.name),
                "weights_ready": weights_ready,
                "extensions_ready": extensions_ready,
                "spz_converter_ready": spz_converter_ready["available"],
                "module": module_status,
                "dependencies": dependency_status,
                "issues": issues,
            }
        )
    gpu = collect_gpu()
    if not gpu.get("available"):
        warnings.append(str(gpu.get("message") or "GPU unavailable"))
    torch = torch_info()
    if torch.get("available") and not torch.get("cuda_available"):
        warnings.append("torch CUDA unavailable")
    return {
        "python": python_info(),
        "gpu": gpu,
        "torch": torch,
        "transformer_engine": import_check("transformer_engine"),
        "preview_runtime": {"litevggt": litevggt_runtime_status(), "worker_preview": preview_workers},
        "fine_runtime": {**fine_runtime_status(), "worker_fine": fine_workers},
        "spz_converter": spark_converter_status(),
        "algorithms": algorithms,
        "errors": errors,
        "warnings": warnings,
    }


def preview_worker_status(db: Session) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    workers = db.scalars(select(WorkerHeartbeat).where(WorkerHeartbeat.worker_id.like("preview-%"))).all()
    active = []
    for item in workers:
        last_seen = item.last_seen_at
        if last_seen.tzinfo is None:
            last_seen = last_seen.replace(tzinfo=timezone.utc)
        age_seconds = (now - last_seen.astimezone(timezone.utc)).total_seconds()
        if age_seconds <= 90:
            active.append(
                {
                    "worker_id": item.worker_id,
                    "hostname": item.hostname,
                    "gpu_name": item.gpu_name,
                    "gpu_memory_total": item.gpu_memory_total,
                    "gpu_memory_used": item.gpu_memory_used,
                    "gpu_utilization": item.gpu_utilization,
                    "age_seconds": round(age_seconds, 1),
                }
            )
    return {
        "available": bool(active),
        "active_count": len(active),
        "stale_count": max(0, len(workers) - len(active)),
        "workers": active,
        "note": "worker-preview runs LiteVGGT direct Spark SPZ preview tasks.",
    }


def fine_worker_status(db: Session) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    workers = db.scalars(select(WorkerHeartbeat).where(WorkerHeartbeat.worker_id.like("fine-%"))).all()
    active = []
    for item in workers:
        last_seen = item.last_seen_at
        if last_seen.tzinfo is None:
            last_seen = last_seen.replace(tzinfo=timezone.utc)
        age_seconds = (now - last_seen.astimezone(timezone.utc)).total_seconds()
        if age_seconds <= 90:
            active.append(
                {
                    "worker_id": item.worker_id,
                    "hostname": item.hostname,
                    "gpu_name": item.gpu_name,
                    "gpu_memory_total": item.gpu_memory_total,
                    "gpu_memory_used": item.gpu_memory_used,
                    "gpu_utilization": item.gpu_utilization,
                    "age_seconds": round(age_seconds, 1),
                }
            )
    return {
        "available": bool(active),
        "active_count": len(active),
        "stale_count": max(0, len(workers) - len(active)),
        "workers": active,
        "note": "worker-fine runs COLMAP preprocessing and DashDeblurGroupGS training from the configured worker image.",
    }


def import_check(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        root = module_name.split(".", 1)[0]
        root_module = importlib.import_module(root) if root != module_name else module
        return {"available": True, "version": getattr(module, "__version__", None) or getattr(root_module, "__version__", None)}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def algorithm_dependency_status(name: str) -> dict[str, Any]:
    spec = ALGORITHM_DEPENDENCIES.get(name, {})
    module_status = {module: import_check(module) for module in spec.get("python_modules", [])}
    executable_statuses = {
        executable: executable_status(executable, executable_version_args(executable))
        for executable in spec.get("executables", [])
    }
    torch_status = torch_info() if spec.get("requires_cuda") else None
    missing = [
        f"python:{module}"
        for module, status in module_status.items()
        if not status.get("available")
    ]
    missing.extend(
        f"command:{command}"
        for command, status in executable_statuses.items()
        if not status.get("available")
    )
    if spec.get("requires_cuda") and not (torch_status or {}).get("cuda_available"):
        missing.append("cuda:torch")
    return {
        "python_modules": module_status,
        "executables": executable_statuses,
        "requires_cuda": bool(spec.get("requires_cuda")),
        "torch_cuda": torch_status,
        "missing": missing,
        "available": not missing,
    }


def worker_runtime_for_algorithm(name: str, preview_workers: dict[str, Any], fine_workers: dict[str, Any]) -> bool:
    if name == "DashDeblurGroupGS Fine":
        return bool(fine_workers.get("available"))
    if name == "Spark SPZ":
        return bool(preview_workers.get("available") or fine_workers.get("available"))
    if name == "LiteVGGT":
        return bool(preview_workers.get("available"))
    return False


def executable_version_args(command: str) -> list[str]:
    if command == "node":
        return ["--version"]
    if command == "git":
        return ["--version"]
    if command == "ffmpeg":
        return ["-version"]
    if command == "colmap":
        return ["help"]
    return ["--version"]


def bundled_module_status(name: str) -> dict[str, Any]:
    modules = {
        "LiteVGGT": "app.preview.vendor.litevggt_runtime",
        "DashDeblurGroupGS Fine": "app.fine.runner",
        "Spark SPZ": "app.preview.io.spz",
    }
    module = modules.get(name)
    return import_check(module) if module else {"available": True}


def fine_runtime_status() -> dict[str, Any]:
    torch_status = torch_info()
    spark_status = spark_converter_status()
    colmap_status = colmap_cli_status()
    ffmpeg_status = executable_status("ffmpeg", ["-version"])
    trainer_status = dash_deblur_group_status()
    dependencies = algorithm_dependency_status("DashDeblurGroupGS Fine")
    modules = {
        "pycolmap": import_check("pycolmap"),
    }
    available = bool(
        colmap_status.get("available")
        and ffmpeg_status.get("available")
        and trainer_status.get("available")
        and dependencies.get("available")
        and all(item.get("available") for item in modules.values())
    )
    return {
        "available": available,
        "torch": torch_status,
        "spark_spz": spark_status,
        "colmap_cli": colmap_status,
        "ffmpeg": ffmpeg_status,
        "dash_deblur_group": trainer_status,
        "dependencies": dependencies,
        **modules,
        "error": None if available else "COLMAP CLI/ffmpeg/pycolmap/DashDeblurGroupGS check failed",
    }


def dash_deblur_group_status() -> dict[str, Any]:
    from app.fine.dash_deblur_group import default_embedded_trainer_dir, detect_trainer_flavor

    repo_value = os.getenv("DASH_DEBLUR_GROUP_REPO")
    repo = Path(repo_value).expanduser().resolve() if repo_value else default_embedded_trainer_dir(resolve_local_path(get_settings().repo_cache_dir))
    train_py = repo / "train.py"
    flavor = detect_trainer_flavor(repo) if repo.exists() else "unknown"
    submodules = {
        "diff_gaussian_rasterization": repo / "submodules" / "diff-gaussian-rasterization",
        "simple_knn": repo / "submodules" / "simple-knn",
        "fused_ssim": repo / "submodules" / "fused-ssim",
    }
    submodule_status = {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in submodules.items()
    }
    return {
        "available": train_py.exists() and train_py.is_file(),
        "repo": str(repo),
        "entrypoint": str(train_py),
        "flavor": flavor,
        "submodules": submodule_status,
        "error": None if train_py.exists() else "embedded fine trainer train.py not found; set DASH_DEBLUR_GROUP_REPO only for an explicit override",
    }


def executable_status(command: str, args: list[str]) -> dict[str, Any]:
    executable = shutil.which(command)
    if not executable:
        return {"available": False, "error": f"{command} executable was not found in PATH"}
    try:
        completed = subprocess.run(
            [executable, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {"available": False, "path": executable, "error": str(exc)}
    return {
        "available": completed.returncode == 0,
        "path": executable,
        "returncode": completed.returncode,
        "version": (completed.stdout or "").splitlines()[0] if completed.stdout else None,
        "error": None if completed.returncode == 0 else (completed.stdout or "").strip()[:500],
    }


def colmap_cli_status() -> dict[str, Any]:
    status = executable_status("colmap", ["help"])
    if not status.get("available"):
        return status
    executable = status["path"]
    try:
        completed = subprocess.run(
            [executable, "help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {"available": False, "path": executable, "error": str(exc)}
    help_text = completed.stdout or ""
    commands = {match.group(1) for match in re.finditer(r"^\s+([a-z_]+)(?:\s|$)", help_text, flags=re.MULTILINE)}
    required = {"feature_extractor", "mapper", "image_undistorter", "model_analyzer"}
    optional = {"global_mapper", "hierarchical_mapper", "view_graph_calibrator", "model_clusterer", "model_splitter"}
    missing = sorted(command for command in required if command not in commands and not colmap_help_contains_command(help_text, command))
    return {
        "available": completed.returncode == 0 and not missing,
        "path": executable,
        "returncode": completed.returncode,
        "required_commands": sorted(required),
        "optional_commands": {command: command in commands or colmap_help_contains_command(help_text, command) for command in sorted(optional)},
        "missing_commands": missing,
        "error": None if completed.returncode == 0 and not missing else f"missing COLMAP CLI commands: {', '.join(missing)}",
    }


def colmap_help_contains_command(help_text: str, command: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(command)}(?![A-Za-z0-9_])"
    return re.search(pattern, help_text) is not None


def litevggt_runtime_status() -> dict[str, Any]:
    try:
        with prepend_sys_path(VENDOR_ROOT / "litevggt"):
            from vggt.models.vggt import VGGT

        return {"available": True, "symbol": VGGT.__name__}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def spark_converter_status() -> dict[str, Any]:
    try:
        cli = locate_spark_cli()
        node = subprocess.run(["node", "--version"], capture_output=True, text=True, check=False)
        if node.returncode != 0:
            return {"available": True, "node_available": False, "node_error": "node --version failed", "cli": str(cli)}
        return {"available": True, "node_available": True, "node": node.stdout.strip(), "cli": str(cli)}
    except Exception as exc:
        if "Spark SPZ converter CLI is missing" in str(exc):
            return {"available": False, "error": str(exc)}
        return {"available": True, "node_available": False, "node_error": str(exc)}


def license_notice_for(name: str) -> str | None:
    for entry in ALGORITHMS:
        if entry["name"] == name:
            return entry.get("license_notice")
    return None
