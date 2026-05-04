from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import AlgorithmRegistry
from app.preview.io.spz import locate_spark_cli
from app.resources import collect_gpu, python_info, torch_info


VENDOR_ROOT = Path(__file__).resolve().parent / "preview" / "vendor"


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
        "name": "EDGS",
        "repo_url": "https://github.com/CompVis/EDGS",
        "license": "Non-commercial academic/personal use",
        "commit_hash_setting": "edgs_repo_commit",
        "local_path": VENDOR_ROOT / "edgs",
        "enabled": True,
        "weight_paths": ["roma/roma_indoor.pth", "roma/dinov2_vitl14_pretrain.pth"],
        "commands": {},
        "source_type": "bundled",
        "license_notice": "EDGS is limited to non-commercial academic/personal use.",
        "notes": "Bundled EDGS preview Gaussian optimizer; CUDA extensions are compiled in worker-preview.",
    },
    {
        "name": "LingBot-Map",
        "repo_url": "https://github.com/robbyant/lingbot-map",
        "license": "Apache-2.0",
        "commit_hash_setting": "lingbot_repo_commit",
        "local_path": VENDOR_ROOT / "lingbot",
        "enabled": True,
        "weight_paths": ["lingbot-map/lingbot-map-long.pt"],
        "commands": {},
        "source_type": "bundled",
        "license_notice": "Apache-2.0; bundled key preview code from fixed upstream commit.",
        "notes": "Bundled LingBot-Map video/streaming preview path.",
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
        "name": "Fine Reconstruction Stack",
        "repo_url": None,
        "license": None,
        "commit_hash_setting": None,
        "local_path": None,
        "enabled": False,
        "weight_paths": [],
        "commands": {},
        "source_type": "reserved",
        "license_notice": None,
        "notes": "Reserved for Faster-GS/FastGS/Deblurring-3DGS/3DGS-LM in worker-fine.",
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
    if input_type == "video":
        return "lingbot_spz"
    normalized = (value or "litevggt_edgs").strip().lower()
    aliases = {
        "edgs": "litevggt_edgs",
        "litevggt_edgs": "litevggt_edgs",
        "litevggt+edgs": "litevggt_edgs",
        "litevggt_spark": "litevggt_spz",
        "litevggt_spz": "litevggt_spz",
        "direct": "litevggt_spz",
    }
    return aliases.get(normalized, "litevggt_edgs")


def runtime_preflight(db: Session, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    algorithms = []
    errors: list[str] = []
    warnings: list[str] = []
    for item in db.scalars(select(AlgorithmRegistry).order_by(AlgorithmRegistry.name)).all():
        issues = []
        module_status = bundled_module_status(item.name)
        weights_ready = True
        extensions_ready = True
        spz_converter_ready = spark_converter_status()
        if item.enabled:
            if item.local_path and not Path(item.local_path).exists():
                issues.append(f"bundled source missing: {item.local_path}")
            for weight_path in item.weight_paths or []:
                if not Path(weight_path).exists():
                    issues.append(f"weight missing: {weight_path}")
                    weights_ready = False
            if not module_status.get("available", True):
                issues.append(f"bundled module import failed: {module_status.get('error')}")
            if item.name == "EDGS":
                edgs_ext = extension_pair_status()
                extensions_ready = bool(edgs_ext["available"])
                if not extensions_ready:
                    issues.append(f"EDGS CUDA extensions missing: {edgs_ext['error']}")
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
        "edgs_cuda_extensions": extension_pair_status(),
        "lingbot_runtime": {"flashinfer": import_check("flashinfer"), "sdpa_fallback": True},
        "spz_converter": spark_converter_status(),
        "algorithms": algorithms,
        "errors": errors,
        "warnings": warnings,
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
        "EDGS": "app.preview.vendor.edgs_runtime",
        "LingBot-Map": "app.preview.vendor.lingbot_runtime",
        "Spark SPZ": "app.preview.io.spz",
    }
    module = modules.get(name)
    return import_check(module) if module else {"available": True}


def extension_pair_status() -> dict[str, Any]:
    raster = import_check("diff_gaussian_rasterization")
    knn = import_check("simple_knn")
    available = bool(raster.get("available") and knn.get("available"))
    return {
        "available": available,
        "diff_gaussian_rasterization": raster,
        "simple_knn": knn,
        "error": None if available else "diff_gaussian_rasterization/simple_knn import failed",
    }


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
