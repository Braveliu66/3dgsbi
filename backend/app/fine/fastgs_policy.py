from __future__ import annotations

import random
from typing import Any, Callable

import torch

from app.fine.local_3dgs.runtime import normalize_render_pkg, normalize_visibility_filter
from app.fine.option_utils import read_float, read_int


RenderFn = Callable[[Any, Any, Any, torch.Tensor], dict[str, torch.Tensor]]


class FastGSPolicy:
    """Local FastGS-style multi-view densification/pruning policy.

    This policy computes FastGS' multi-view decision signal from sampled views
    and uses CUDA metric accumulation when the rebuilt rasterizer exposes it.
    """

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        self.sample_cameras = read_int(options.get("fine_fastgs_sample_cameras"), 10, minimum=1, maximum=32)
        self.loss_thresh = read_float(options.get("fine_fastgs_loss_thresh"), 0.08, minimum=0.0, maximum=1.0)
        self.percentile = read_float(options.get("fine_vcd_percentile"), 0.60, minimum=0.05, maximum=0.95)
        self.lambda_dssim = read_float(options.get("fine_lambda_dssim"), 0.2, minimum=0.0, maximum=1.0)
        self.score_sum: torch.Tensor | None = None
        self.metric_counts: torch.Tensor | None = None
        self.pruning_score: torch.Tensor | None = None
        self.last_candidate_ratio: float | None = None
        self.multiview_evaluations = 0
        self.densify_events = 0
        self.prune_events = 0
        self.final_prune_events = 0
        self.gaussian_count_curve: list[int] = []
        self.cuda_metric_available = cuda_metric_available()
        self.compiled_features = compiled_features()
        self.compact_box_available = "compact_box" in self.compiled_features
        self.cuda_metric_calls = 0

    def update_multiview_scores(
        self,
        *,
        cameras: list[Any],
        gaussians: Any,
        pipe: Any,
        background: torch.Tensor,
        render_fn: RenderFn,
        ssim_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        if not cameras:
            return
        selected = random.sample(cameras, k=min(self.sample_cameras, len(cameras)))
        gaussian_count = int(gaussians.get_xyz.shape[0])
        score_sum = torch.zeros((gaussian_count, 1), dtype=torch.float32, device=background.device)
        metric_counts = torch.zeros((gaussian_count, 1), dtype=torch.float32, device=background.device)

        for viewpoint in selected:
            with torch.no_grad():
                pkg = normalize_render_pkg(render_fn(viewpoint, gaussians, pipe, background), gaussian_count)
                image = pkg["render"]
                gt = viewpoint.original_image.to(background.device)
                pixel_l1 = torch.abs(image - gt).mean(dim=0)
                denom = (pixel_l1.max() - pixel_l1.min()).clamp_min(1e-6)
                pixel_l1_norm = (pixel_l1 - pixel_l1.min()) / denom
                ssim_gap = (1.0 - ssim_fn(image, gt)).detach().clamp(0.0, 1.0)
                loss_map = (1.0 - self.lambda_dssim) * pixel_l1_norm + self.lambda_dssim * ssim_gap
                high_error_ratio = (loss_map > self.loss_thresh).float().mean()
                visible = pkg["visibility_filter"].reshape(-1)
            if visible.any():
                view_score = float(loss_map.mean().item())
                metric_weight = float(high_error_ratio.item() * len(selected))
                if self.cuda_metric_available and cuda_accumulate_metrics(visible, view_score, metric_weight, score_sum, metric_counts):
                    self.cuda_metric_calls += 1
                else:
                    score_sum[visible] += view_score
                    metric_counts[visible] += metric_weight

        valid = metric_counts > 0
        pruning_score = torch.zeros_like(score_sum)
        if valid.any():
            raw = score_sum / metric_counts.clamp_min(1.0)
            values = raw[valid]
            pruning_score[valid] = (values - values.min()) / (values.max() - values.min()).clamp_min(1e-6)
        self.score_sum = score_sum
        self.metric_counts = metric_counts
        self.pruning_score = pruning_score
        self.multiview_evaluations += 1

    def apply_densification_gate(self, gaussians: Any) -> torch.Tensor | None:
        if self.metric_counts is None or self.pruning_score is None:
            return None
        if self.metric_counts.shape[0] != gaussians.get_xyz.shape[0]:
            return None
        eligible = self.metric_counts.squeeze(-1) >= 1.0
        if int(eligible.sum().item()) < 32:
            return None
        threshold = torch.quantile(self.pruning_score[eligible].squeeze(-1), self.percentile)
        candidates = eligible & (self.pruning_score.squeeze(-1) >= threshold)
        self.last_candidate_ratio = float(candidates.float().mean().item())
        original = gaussians.xyz_gradient_accum.clone()
        gaussians.xyz_gradient_accum[~candidates] *= 0.10
        self.densify_events += 1
        return original

    def apply_final_prune(self, gaussians: Any, *, min_opacity: float = 0.1) -> None:
        if self.pruning_score is None or self.pruning_score.shape[0] != gaussians.get_xyz.shape[0]:
            return
        score_mask = self.pruning_score.squeeze(-1) > 0.9
        opacity_mask = gaussians.get_opacity.squeeze(-1) < min_opacity
        prune_mask = torch.logical_or(score_mask, opacity_mask)
        if bool(prune_mask.any().item()):
            gaussians.prune_points(prune_mask)
            self.final_prune_events += 1
            self.prune_events += 1

    def reset_after_topology_change(self) -> None:
        self.score_sum = None
        self.metric_counts = None
        self.pruning_score = None

    def observe_gaussian_count(self, gaussians: Any) -> None:
        count = int(gaussians.get_xyz.shape[0])
        if not self.gaussian_count_curve or self.gaussian_count_curve[-1] != count:
            self.gaussian_count_curve.append(count)

    def metrics(self) -> dict[str, Any]:
        return {
            "fastgs_multiview_score_enabled": True,
            "fastgs_sample_cameras": self.sample_cameras,
            "fastgs_loss_thresh": self.loss_thresh,
            "fastgs_multiview_evaluations": self.multiview_evaluations,
            "fastgs_densify_events": self.densify_events,
            "fastgs_prune_events": self.prune_events,
            "fastgs_final_prune_events": self.final_prune_events,
            "fastgs_vcd_last_candidate_ratio": self.last_candidate_ratio,
            "fastgs_gaussian_count_curve": self.gaussian_count_curve[-32:],
            "fastgs_vcp_mode": "local_multiview_loss_map_visibility_assignment",
            "fastgs_cuda_metric_enabled": self.cuda_metric_available and self.cuda_metric_calls > 0,
            "fastgs_cuda_metric_calls": self.cuda_metric_calls,
            "fastgs_metric_count_nonzero": int(self.metric_counts.gt(0).sum().item()) if self.metric_counts is not None else 0,
            "fastergs_compiled_features": self.compiled_features,
        }


def vcp_min_opacity(policy: FastGSPolicy) -> float:
    if policy.last_candidate_ratio is not None and policy.last_candidate_ratio > 0.15:
        return 0.003
    return 0.005


def visible_mean_loss_score(visibility_filter: torch.Tensor, gaussian_count: int, view_error: float) -> torch.Tensor:
    visible = normalize_visibility_filter(visibility_filter, gaussian_count)
    scores = torch.zeros((gaussian_count, 1), dtype=torch.float32, device=visibility_filter.device)
    scores[visible] = float(view_error)
    return scores


def cuda_metric_available() -> bool:
    try:
        from diff_gaussian_rasterization import _RasterizeGaussians

        return hasattr(_RasterizeGaussians, "fastgs_accumulate_metrics")
    except Exception:
        return False


def compiled_features() -> list[str]:
    try:
        import diff_gaussian_rasterization
    except Exception:
        return []
    features: list[str] = []
    if getattr(diff_gaussian_rasterization, "MOBILEGS_COMPACT_BOX", False):
        features.append("compact_box")
    if getattr(diff_gaussian_rasterization, "MOBILEGS_FASTGS_METRIC", False):
        features.append("cuda_metric_accumulation")
    return features


def cuda_accumulate_metrics(
    visible: torch.Tensor,
    view_score: float,
    metric_weight: float,
    score_sum: torch.Tensor,
    metric_counts: torch.Tensor,
) -> bool:
    try:
        from diff_gaussian_rasterization import _RasterizeGaussians

        _RasterizeGaussians.fastgs_accumulate_metrics(visible.reshape(-1), float(view_score), float(metric_weight), score_sum, metric_counts)
        return True
    except Exception:
        return False
