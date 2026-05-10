from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from app.fine.types import FineFailure
from app.preview.weights import safe_weight_path


SPEED3R_PI3_CONFIG_REL = "speed3r_pi3/config.json"
SPEED3R_PI3_WEIGHT_REL = "speed3r_pi3/model.safetensors"
MAST3R_WEIGHT_REL = "mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
MAST3R_RETRIEVAL_WEIGHT_REL = "mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
MAST3R_RETRIEVAL_CODEBOOK_REL = "mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"


def speed3r_pi3_paths(model_cache_dir: Path) -> dict[str, Path]:
    return {
        "config": safe_weight_path(model_cache_dir, SPEED3R_PI3_CONFIG_REL),
        "model": safe_weight_path(model_cache_dir, SPEED3R_PI3_WEIGHT_REL),
        "mast3r": safe_weight_path(model_cache_dir, MAST3R_WEIGHT_REL),
        "mast3r_retrieval": safe_weight_path(model_cache_dir, MAST3R_RETRIEVAL_WEIGHT_REL),
        "mast3r_codebook": safe_weight_path(model_cache_dir, MAST3R_RETRIEVAL_CODEBOOK_REL),
    }


def ensure_video_artdeco_weights(model_cache_dir: Path) -> dict[str, Path]:
    paths = speed3r_pi3_paths(model_cache_dir)
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists() or path.stat().st_size <= 0]
    if missing:
        raise FineFailure("MODEL_WEIGHT_MISSING", "Video ARTDECO weights missing: " + "; ".join(missing))
    return paths


class Speed3RPi3Adapter:
    """Thin replacement for ARTDECO's Pi3 loop-closure inference."""

    def __init__(self, model_dir: Path, *, speed3r_root: Path | None = None, device: str = "cuda:0") -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.speed3r_root = Path(speed3r_root).resolve() if speed3r_root else None
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model
        if self.speed3r_root and str(self.speed3r_root) not in sys.path:
            sys.path.insert(0, str(self.speed3r_root))
        try:
            from pi3.models.pi3_sparse import Pi3_Sparse
        except Exception as exc:
            raise FineFailure("SPEED3R_RUNTIME_UNAVAILABLE", f"Speed3R Pi3 package import failed: {exc}") from exc
        try:
            model = Pi3_Sparse.from_pretrained(str(self.model_dir)).to(self.device).eval()
        except Exception as exc:
            raise FineFailure("SPEED3R_WEIGHT_LOAD_FAILED", f"Could not load Speed3R-Pi3 from {self.model_dir}: {exc}") from exc
        self._model = model
        return model

    def infer_tensor(self, images) -> dict[str, Any]:
        try:
            import torch
        except Exception as exc:
            raise FineFailure("TORCH_UNAVAILABLE", f"PyTorch import failed: {exc}") from exc

        model = self.load()
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=str(self.device).startswith("cuda")):
            result = model(images.to(self.device, non_blocking=True))
        return {
            "points": result["points"],
            "local_points": result.get("local_points"),
            "conf": result["conf"],
            "camera_poses": result["camera_poses"],
        }
