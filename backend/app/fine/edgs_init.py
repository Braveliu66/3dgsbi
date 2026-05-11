from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.fine.types import FineFailure


class EDGSDenseInit:
    """Small EDGS/RoMA adapter for dense-correspondence Gaussian initialization."""

    def __init__(self, device: str = "cuda", roma_model_name: str = "outdoor") -> None:
        self.device = device
        self.roma_model_name = roma_model_name
        self.last_metrics: dict[str, Any] = {}

    def initialize(
        self,
        gaussian_model: Any,
        scene: Any,
        cfg_init_wC: Any,
        roma_model: Any | None = None,
    ) -> Any:
        try:
            from app.fine.edgs_runtime.corr_init import init_gaussians_with_corr
        except Exception as exc:
            raise FineFailure("EDGS_RUNTIME_UNAVAILABLE", f"EDGS runtime import failed: {exc}") from exc

        try:
            result = init_gaussians_with_corr(
                gaussians=gaussian_model,
                scene=scene,
                cfg=cfg_init_wC,
                device=self.device,
                roma_model=roma_model,
            )
        except FineFailure:
            raise
        except Exception as exc:
            raise FineFailure("EDGS_RUNTIME_UNAVAILABLE", f"EDGS dense initialization failed: {exc}") from exc

        self.last_metrics = {
            "edgs_enabled": True,
            "edgs_matches_per_ref": int(cfg_init_wC.matches_per_ref),
            "edgs_nns_per_ref": int(cfg_init_wC.nns_per_ref),
            "edgs_num_refs": int(cfg_init_wC.num_refs),
            "edgs_roma_model": str(getattr(cfg_init_wC, "roma_model", self.roma_model_name)),
            "edgs_roma_coarse_res": str(getattr(cfg_init_wC, "roma_coarse_res", 560)),
            "edgs_roma_upsample_res": str(getattr(cfg_init_wC, "roma_upsample_res", 864)),
            "edgs_roma_sample_thresh": float(getattr(cfg_init_wC, "roma_sample_thresh", 0.05)),
            **result.metrics(),
        }
        return gaussian_model


def make_edgs_cfg(
    matches_per_ref: int = 15_000,
    nns_per_ref: int = 3,
    num_refs: int | None = None,
    scene: Any | None = None,
    roma_model: str = "outdoor",
    roma_coarse_res: int | tuple[int, int] = 560,
    roma_upsample_res: int | tuple[int, int] = 864,
    roma_sample_thresh: float = 0.05,
    roma_sample_mode: str = "threshold_balanced",
    roma_symmetric: bool = True,
    roma_use_custom_corr: bool = True,
    roma_upsample_preds: bool = True,
    roma_with_padding: bool = False,
    max_points: int = 500_000,
    reprojection_error: float = 4.0,
) -> SimpleNamespace:
    if num_refs is None and scene is not None:
        num_refs = len(scene.getTrainCameras())
    elif num_refs is None:
        num_refs = 180

    return SimpleNamespace(
        matches_per_ref=int(matches_per_ref),
        nns_per_ref=int(nns_per_ref),
        num_refs=int(num_refs),
        roma_model=str(roma_model),
        roma_coarse_res=roma_coarse_res,
        roma_upsample_res=roma_upsample_res,
        roma_sample_thresh=float(roma_sample_thresh),
        roma_sample_mode=str(roma_sample_mode),
        roma_symmetric=bool(roma_symmetric),
        roma_use_custom_corr=bool(roma_use_custom_corr),
        roma_upsample_preds=bool(roma_upsample_preds),
        roma_with_padding=bool(roma_with_padding),
        max_points=int(max_points),
        reprojection_error=float(reprojection_error),
    )


def roma_weight_paths(model_cache_dir: Path) -> tuple[Path, Path, Path]:
    root = Path(model_cache_dir) / "roma"
    return (
        root / "roma_outdoor.pth",
        root / "roma_indoor.pth",
        root / "dinov2_vitl14_pretrain.pth",
    )


def ensure_roma_weights(model_cache_dir: Path) -> None:
    missing = [str(path) for path in roma_weight_paths(model_cache_dir) if not path.exists() or not path.is_file()]
    if missing:
        raise FineFailure("ROMA_WEIGHT_MISSING", f"RoMA/EDGS weights not found: {', '.join(missing)}")
