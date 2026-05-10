from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import AlgorithmRegistry, WorkerHeartbeat
from app.preview.io.spz import locate_spark_cli
from app.preview.utils import prepend_sys_path
from app.preview.weights import weight_file_status
from app.resources import collect_gpu, python_info, torch_info


VENDOR_ROOT = Path(__file__).resolve().parent / "preview" / "vendor"
FINE_ROOT = Path(__file__).resolve().parent / "fine"


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
        "name": "LingBot-Map Video Preview",
        "repo_url": "https://github.com/Robbyant/lingbot-map",
        "license": "Apache-2.0",
        "commit_hash_setting": "lingbot_map_repo_commit",
        "local_path": Path(__file__).resolve().parent / "preview" / "adapters" / "lingbot.py",
        "enabled": True,
        "weight_paths": ["lingbot/lingbot-map-long.pt"],
        "commands": {},
        "source_type": "pinned_runtime_package",
        "license_notice": "Apache-2.0; worker installs the LingBot-Map core package at a pinned commit and excludes optional rendering/visualization dependencies.",
        "notes": "Video fast preview pipeline: sampled video frames -> LingBot-Map streaming/windowed RGB-D reconstruction -> lightweight Gaussian PLY -> Spark SPZ. It does not use LingBot offline rendering, Kaolin, Open3D, viser, sky segmentation, or custom render CUDA extensions.",
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
        "name": "MobileGS Fine (AMB3R-SfM + DeblurMLP + LM-RS)",
        "repo_url": "https://github.com/HengyiWang/amb3r",
        "license": "Mixed upstream terms; AMB3R/VGGT/PTv3 + Deblur/LM-RS research use",
        "commit_hash_setting": "amb3r_repo_commit",
        "local_path": FINE_ROOT / "runner.py",
        "enabled": True,
        "weight_paths": ["amb3r/amb3r.pt"],
        "commands": {},
        "source_type": "system",
        "license_notice": "Fine pipeline integrates the minimal AMB3R-SfM runtime from commit 7aae7fbb77a750651ffa236bb9c3212290c6fc78, Deblurring-3DGS GTnet ideas, and LM-RS matrix-free optimization. Preview still uses LiteVGGT.",
        "notes": "Default image fine reconstruction pipeline: JPG/PNG normalization, AMB3R-SfM COLMAP-compatible no-EXIF sparse/0 initialization, DeblurMLP-MobileGS training, FastGS-style densification/pruning, patched LM-RS Compact Box rasterizer, optional LM-RS matrix-free Phase 2, and Spark SPZ conversion. pycolmap is explicit diagnostics only.",
    },
    {
        "name": "Video Fine ARTDECO + Speed3R-Pi3",
        "repo_url": "https://github.com/InternRobotics/ARTDECO",
        "license": "ARTDECO/MASt3R/Speed3R-Pi3 research and non-commercial restrictions; verify upstream terms before commercial use",
        "commit_hash_setting": "artdeco_repo_commit",
        "local_path": FINE_ROOT / "video",
        "enabled": True,
        "weight_paths": [
            "speed3r_pi3/config.json",
            "speed3r_pi3/model.safetensors",
            "mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
            "mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
            "mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl",
        ],
        "commands": {},
        "source_type": "adapted_module",
        "license_notice": "Video fine integrates ARTDECO VSLAM/Reconstruct at commit bb654395826e50ac9e4671682d901377115a24ce and replaces Pi3 loop-closure inference with Speed3R-Pi3 from commit 5460f7309c87e5daac36385ff6611627de7d7267. Speed3R-Pi3 weights are CC BY-NC 4.0.",
        "notes": "Canonical video pipeline video_artdeco_speed3r. It accepts exactly one video, extracts frames, writes pinhole intrinsics when absent, runs ARTDECO h3dgsv3 training, validates point_clouds/gs.ply, and only then converts final.ply to Spark SPZ. It does not call AMB3R, MobileGS, LM-RS, or DeblurMLP.",
    },
    {
        "name": "Deblurring-3DGS GTnet",
        "repo_url": "https://github.com/benhenryL/Deblurring-3D-Gaussian-Splatting",
        "license": "Research/non-commercial risk; verify upstream terms before commercial use",
        "commit_hash_setting": "deblurring_3dgs_repo_commit",
        "local_path": FINE_ROOT / "deblur_mlp.py",
        "enabled": True,
        "weight_paths": [],
        "commands": {},
        "source_type": "adapted_module",
        "license_notice": "Only the GTnet/Fourier embedding training-time blur model is adapted locally; the upstream environment and full repository are not vendored.",
        "notes": "DeblurMLP models blurred observations during training. It does not export deblurred 2D images and final output remains a standard sharp Gaussian PLY.",
    },
    {
        "name": "FastGS Reference",
        "repo_url": "https://github.com/MrNeRF/FastGS",
        "license": "Research license, see upstream",
        "commit_hash_setting": "fastgs_repo_commit",
        "local_path": None,
        "enabled": False,
        "weight_paths": [],
        "commands": {},
        "source_type": "optional_reference",
        "license_notice": "Registered as an optional algorithm reference only; official FastGS repo is not vendored or required by the default fine path.",
        "notes": "Reference for future selective CUDA/kernel optimization. It must be compiled inside the existing worker CUDA/Python environment if enabled later.",
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


def normalize_preview_pipeline(value: str | None, input_type: str) -> str:
    default = "lingbot_map_spz" if input_type == "video" else "litevggt_spz"
    normalized = (value or default).strip().lower()
    aliases = {
        "litevggt_spark": "litevggt_spz",
        "litevggt_spz": "litevggt_spz",
        "direct": "litevggt_spz",
        "lingbot": "lingbot_map_spz",
        "lingbot_map": "lingbot_map_spz",
        "lingbot_map_spz": "lingbot_map_spz",
        "video_lingbot": "lingbot_map_spz",
    }
    return aliases.get(normalized, normalized)


def runtime_preflight(db: Session, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    algorithms = []
    errors: list[str] = []
    warnings: list[str] = []
    fine_workers = fine_worker_status(db)
    for item in db.scalars(select(AlgorithmRegistry).order_by(AlgorithmRegistry.name)).all():
        issues = []
        module_status = bundled_module_status(item.name)
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
            if not module_status.get("available", True):
                issues.append(f"bundled module import failed: {module_status.get('error')}")
            if item.name == "MobileGS Fine (AMB3R-SfM + DeblurMLP + LM-RS)":
                fine_runtime = fine_runtime_status()
                extensions_ready = bool(fine_runtime["available"] or fine_workers["available"])
                if not extensions_ready:
                    issues.append("worker-fine heartbeat missing and backend fine CUDA runtime unavailable")
            if item.name == "LingBot-Map Video Preview":
                lingbot_runtime = lingbot_preview_runtime_status()
                extensions_ready = bool(lingbot_runtime["available"] or preview_worker_status(db)["available"])
                if not extensions_ready:
                    issues.append("worker-preview heartbeat missing and backend LingBot-Map runtime unavailable")
            if item.name == "Video Fine ARTDECO + Speed3R-Pi3":
                video_runtime = artdeco_video_runtime_status()
                extensions_ready = bool(video_runtime["available"] or fine_workers["available"])
                if not extensions_ready:
                    issues.append("worker-fine heartbeat missing and backend ARTDECO video runtime unavailable")
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
        "preview_runtime": {"lingbot_map": lingbot_preview_runtime_status(), "worker_preview": preview_worker_status(db)},
        "fine_runtime": {**fine_runtime_status(), "worker_fine": fine_workers},
        "video_fine_runtime": {**artdeco_video_runtime_status(), "worker_fine": fine_workers},
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
        "note": "worker-fine runs from the same CUDA/PyTorch worker image and model-cache as preview.",
    }


def import_check(module_name: str) -> dict[str, Any]:
    try:
        module = __import__(module_name)
        return {"available": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def bundled_module_status(name: str) -> dict[str, Any]:
    modules = {
        "LiteVGGT": "app.preview.vendor.litevggt_runtime",
        "LingBot-Map Video Preview": "app.preview.adapters.lingbot",
        "MobileGS Fine (AMB3R-SfM + DeblurMLP + LM-RS)": "app.fine.runner",
        "Video Fine ARTDECO + Speed3R-Pi3": "app.fine.video.pipeline",
        "Deblurring-3DGS GTnet": "app.fine.deblur_mlp",
        "Spark SPZ": "app.preview.io.spz",
    }
    module = modules.get(name)
    return import_check(module) if module else {"available": True}


def extension_pair_status() -> dict[str, Any]:
    raster = import_check("diff_gaussian_rasterization")
    knn = import_check("simple_knn")
    fused = import_check("fused_ssim")
    available = bool(raster.get("available") and knn.get("available") and fused.get("available"))
    return {
        "available": available,
        "diff_gaussian_rasterization": raster,
        "simple_knn": knn,
        "fused_ssim": fused,
        "error": None if available else "diff_gaussian_rasterization/simple_knn/fused_ssim import failed",
    }


def fine_runtime_status() -> dict[str, Any]:
    torch_status = torch_info()
    spark_status = spark_converter_status()
    modules = {
        "amb3r": import_check("app.fine.amb3r_sfm"),
        "omegaconf": import_check("omegaconf"),
        "spconv": import_check("spconv.pytorch"),
        "torch_scatter": import_check("torch_scatter"),
        "timm": import_check("timm"),
        "deblur_mlp": import_check("app.fine.deblur_mlp"),
        "diff_gaussian_rasterization": import_check("diff_gaussian_rasterization"),
        "simple_knn": import_check("simple_knn"),
        "fused_ssim": import_check("fused_ssim"),
    }
    optional_modules = {
        "pycolmap_diagnostic": import_check("pycolmap"),
    }
    modules["lmrs_matrix_free"] = lmrs_matrix_free_status(modules["diff_gaussian_rasterization"])
    modules["compact_box"] = compact_box_status(modules["diff_gaussian_rasterization"])
    available = bool(
        torch_status.get("available")
        and torch_status.get("cuda_available")
        and spark_status.get("available")
        and all(item.get("available") for item in modules.values())
    )
    return {
        "available": available,
        "torch_cuda": torch_status,
        "spark_spz": spark_status,
        **modules,
        **optional_modules,
        "error": None if available else "CUDA torch/Spark SPZ/AMB3R-SfM/DeblurMLP/LM-RS rasterizer/Compact Box/simple_knn/fused_ssim check failed",
    }


def lingbot_preview_runtime_status() -> dict[str, Any]:
    torch_status = torch_info()
    spark_status = spark_converter_status()
    modules = {
        "lingbot_adapter": import_check("app.preview.adapters.lingbot"),
        "lingbot_map": import_check("lingbot_map.models.gct_stream"),
        "opencv": import_check("cv2"),
        "einops": import_check("einops"),
        "safetensors": import_check("safetensors"),
        "flashinfer": import_check("flashinfer"),
    }
    unavailable_required = [
        name
        for name, status in modules.items()
        if name != "flashinfer" and not status.get("available")
    ]
    available = bool(
        torch_status.get("available")
        and torch_status.get("cuda_available")
        and spark_status.get("available")
        and not unavailable_required
    )
    return {
        "available": available,
        "torch_cuda": torch_status,
        "spark_spz": spark_status,
        **modules,
        "sdpa_fallback": not modules["flashinfer"].get("available"),
        "error": None if available else "CUDA torch/Spark SPZ/LingBot-Map core runtime check failed",
    }


def artdeco_video_runtime_status() -> dict[str, Any]:
    torch_status = torch_info()
    spark_status = spark_converter_status()
    modules = {
        "video_pipeline": import_check("app.fine.video.pipeline"),
        "speed3r_pi3": import_check("app.fine.video.speed3r_pi3"),
        "gsplat": import_check("gsplat"),
        "safetensors": import_check("safetensors"),
        "pypose": import_check("pypose"),
        "natsort": import_check("natsort"),
        "mast3r_slam_backends": import_check("mast3r_slam_backends"),
        "diff_gaussian_rasterization": import_check("diff_gaussian_rasterization"),
        "simple_knn": import_check("simple_knn"),
        "fused_ssim": import_check("fused_ssim"),
    }
    modules["artdeco_adam"] = artdeco_adam_status(modules["diff_gaussian_rasterization"])
    available = bool(
        torch_status.get("available")
        and torch_status.get("cuda_available")
        and spark_status.get("available")
        and all(item.get("available") for item in modules.values())
    )
    return {
        "available": available,
        "torch_cuda": torch_status,
        "spark_spz": spark_status,
        **modules,
        "error": None if available else "CUDA torch/Spark SPZ/ARTDECO VSLAM backend/Speed3R-Pi3/MASt3R runtime check failed",
    }


def artdeco_adam_status(raster_status: dict[str, Any]) -> dict[str, Any]:
    if not raster_status.get("available"):
        return {"available": False, "error": "diff_gaussian_rasterization import failed"}
    try:
        from app.fine.video.artdeco_optimizer_compat import install_artdeco_adam_compat

        install_artdeco_adam_compat()
        module = __import__("diff_gaussian_rasterization")
        missing = [name for name in ("adamUpdate", "adamUpdateBasic") if not hasattr(module, name)]
        if missing:
            return {"available": False, "error": f"missing ARTDECO Adam symbols: {', '.join(missing)}"}
        return {"available": True, "symbols": ["adamUpdate", "adamUpdateBasic"]}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def litevggt_runtime_status() -> dict[str, Any]:
    try:
        with prepend_sys_path(VENDOR_ROOT / "litevggt"):
            from vggt.models.vggt import VGGT

        return {"available": True, "symbol": VGGT.__name__}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def lmrs_matrix_free_status(raster_status: dict[str, Any]) -> dict[str, Any]:
    if not raster_status.get("available"):
        return {"available": False, "error": "diff_gaussian_rasterization import failed"}
    try:
        module = __import__("diff_gaussian_rasterization")
        extension = getattr(module, "_C", None)
        missing = [name for name in ("get_JTv", "get_Diag", "get_JTJv") if not hasattr(extension, name)]
        if missing:
            return {"available": False, "error": f"missing LM-RS matrix-free symbols: {', '.join(missing)}"}
        return {"available": True, "symbols": ["get_JTv", "get_Diag", "get_JTJv"]}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def compact_box_status(raster_status: dict[str, Any]) -> dict[str, Any]:
    if not raster_status.get("available"):
        return {"available": False, "error": "diff_gaussian_rasterization import failed"}
    try:
        module = __import__("diff_gaussian_rasterization")
        if bool(getattr(module, "MOBILEGS_COMPACT_BOX", False)):
            return {"available": True, "backend": "fastgs_compact_box_on_lmrs"}
        return {"available": False, "error": "missing MOBILEGS_COMPACT_BOX marker"}
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
