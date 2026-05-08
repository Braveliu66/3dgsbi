from __future__ import annotations

import torch
import torch.nn.functional as F


def scale_invariant_alignment(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    mask: torch.Tensor | None,
    align_resolution: int = 64,
    beta: float = 0.0,
    trunc: float | None = 1.0,
    sparsity_aware: bool = False,
    detach: bool = False,
):
    del beta, sparsity_aware
    pred_lr, gt_lr, mask_lr = _resize_for_alignment(pred_points, gt_points, mask, align_resolution)
    scale = _least_squares_scale(pred_lr.flatten(1, -2), gt_lr.flatten(1, -2), mask_lr.flatten(1))
    if detach:
        scale = scale.detach()

    if trunc is not None:
        pred_scaled_lr = scale[:, None, None, None] * pred_lr
        rel_err = (pred_scaled_lr - gt_lr).norm(dim=-1) / gt_lr.norm(dim=-1).clamp_min(1e-6)
        robust_mask = mask_lr & (rel_err <= trunc)
        robust_scale = _least_squares_scale(pred_lr.flatten(1, -2), gt_lr.flatten(1, -2), robust_mask.flatten(1))
        scale = torch.where(robust_scale > 0, robust_scale, scale)

    return scale[:, None, None, None] * pred_points, scale


def _resize_for_alignment(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    mask: torch.Tensor | None,
    align_resolution: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = pred_points.shape[0]
    height, width = pred_points.shape[-3:-1]
    if mask is None:
        mask = torch.ones((batch, height, width), dtype=torch.bool, device=pred_points.device)
    else:
        mask = mask.to(device=pred_points.device, dtype=torch.bool)

    if min(height, width) <= align_resolution:
        return pred_points, gt_points, mask

    target = (align_resolution, align_resolution)
    pred_lr = F.interpolate(pred_points.permute(0, 3, 1, 2), size=target, mode="nearest").permute(0, 2, 3, 1)
    gt_lr = F.interpolate(gt_points.permute(0, 3, 1, 2), size=target, mode="nearest").permute(0, 2, 3, 1)
    mask_lr = F.interpolate(mask[:, None].float(), size=target, mode="nearest").squeeze(1) > 0.5
    return pred_lr, gt_lr, mask_lr


def _least_squares_scale(pred_points: torch.Tensor, gt_points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask & torch.isfinite(pred_points).all(dim=-1) & torch.isfinite(gt_points).all(dim=-1)
    weights = valid.float()
    numerator = (weights[..., None] * pred_points * gt_points).sum(dim=(1, 2))
    denominator = (weights[..., None] * pred_points.square()).sum(dim=(1, 2)).clamp_min(1e-8)
    scale = numerator / denominator
    return torch.where(torch.isfinite(scale) & (scale > 0), scale, torch.ones_like(scale))
