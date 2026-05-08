from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from app.fine.types import FineFailure
from app.preview.utils import VENDOR_ROOT, prepend_sys_path


GS_ROOT = VENDOR_ROOT / "edgs" / "gaussian_splatting"


@dataclass(slots=True)
class Local3DGSRuntime:
    root: Path
    render: object
    GaussianModel: type
    Scene: type
    l1_loss: object
    ssim: object


@contextmanager
def local_3dgs_runtime() -> Iterator[Local3DGSRuntime]:
    if not GS_ROOT.exists():
        raise FineFailure("LOCAL_3DGS_RUNTIME_UNAVAILABLE", f"Bundled 3DGS runtime not found: {GS_ROOT}")
    with prepend_sys_path(GS_ROOT):
        from gaussian_renderer import render
        from scene import GaussianModel, Scene
        from utils.loss_utils import l1_loss, ssim

        yield Local3DGSRuntime(
            root=GS_ROOT,
            render=render,
            GaussianModel=GaussianModel,
            Scene=Scene,
            l1_loss=l1_loss,
            ssim=ssim,
        )


def normalize_visibility_filter(visibility_filter: torch.Tensor, gaussian_count: int) -> torch.Tensor:
    if visibility_filter.dtype == torch.bool and visibility_filter.numel() == gaussian_count:
        return visibility_filter.reshape(-1)
    visible = torch.zeros((gaussian_count,), dtype=torch.bool, device=visibility_filter.device)
    if visibility_filter.numel() == 0:
        return visible
    indices = visibility_filter.reshape(-1).long()
    indices = indices[(indices >= 0) & (indices < gaussian_count)]
    visible[indices] = True
    return visible


def normalize_render_pkg(render_pkg: dict[str, torch.Tensor], gaussian_count: int) -> dict[str, torch.Tensor]:
    render_pkg["visibility_filter"] = normalize_visibility_filter(render_pkg["visibility_filter"], gaussian_count)
    return render_pkg

