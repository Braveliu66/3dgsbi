from __future__ import annotations

from typing import Any

from app.fine.types import FineFailure


def ensure_lmrs_matrix_free_symbols() -> None:
    try:
        from diff_gaussian_rasterization import _RasterizeGaussians
    except Exception as exc:
        raise FineFailure("LMRS_MATRIX_FREE_UNAVAILABLE", f"LM-RS rasterizer wrapper unavailable: {exc}") from exc
    for name in ("get_JTv", "get_Diag", "get_JTJv"):
        if not hasattr(_RasterizeGaussians, name):
            raise FineFailure("LMRS_MATRIX_FREE_UNAVAILABLE", f"LM-RS rasterizer symbol missing: {name}")


def compact_box_status() -> str:
    try:
        module = __import__("diff_gaussian_rasterization")
        if bool(getattr(module, "MOBILEGS_COMPACT_BOX", False)):
            return "mobilegs_lmrs_fastgs_compact_box"
    except Exception:
        pass
    return "compact_box_marker_unavailable"


def resolve_lm_status(lm_start_iter: int, iterations: int) -> dict[str, Any]:
    if lm_start_iter >= iterations:
        return {"active": False, "start_iter": lm_start_iter, "reason": "LM phase disabled because start_iter >= iterations"}
    return {
        "active": False,
        "start_iter": lm_start_iter,
        "reason": "LM-RS temporarily isolated due to unstable local backend",
    }
