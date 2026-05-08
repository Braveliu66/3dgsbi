from __future__ import annotations

from typing import Any

import torch


class FastGSPolicy:
    def __init__(self, *, percentile: float = 0.60, window: int = 5) -> None:
        self.percentile = max(0.05, min(0.95, percentile))
        self.window = max(1, window)
        self.score_sum: torch.Tensor | None = None
        self.score_count: torch.Tensor | None = None
        self.observations = 0
        self.last_candidate_ratio: float | None = None

    def observe(self, gaussian_count: int, visibility_filter: torch.Tensor, view_error: float) -> None:
        indices = torch.nonzero(visibility_filter.reshape(-1), as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            return
        if self.score_sum is None or self.score_sum.shape[0] != gaussian_count:
            self.score_sum = torch.zeros((gaussian_count, 1), device="cuda")
            self.score_count = torch.zeros((gaussian_count, 1), device="cuda")
        self.score_sum[indices] += float(view_error)
        self.score_count[indices] += 1.0
        self.observations += 1

    def apply_vcd_gate(self, gaussians: Any) -> torch.Tensor | None:
        if self.score_sum is None or self.score_count is None or self.score_sum.shape[0] != gaussians.get_xyz.shape[0]:
            return None
        eligible = self.score_count.squeeze(-1) >= min(2, self.window)
        if int(eligible.sum().item()) < 32:
            return None
        scores = (self.score_sum / self.score_count.clamp_min(1.0)).squeeze(-1)
        threshold = torch.quantile(scores[eligible], self.percentile)
        candidates = eligible & (scores >= threshold)
        self.last_candidate_ratio = float(candidates.float().mean().item())
        original = gaussians.xyz_gradient_accum.clone()
        gaussians.xyz_gradient_accum[~candidates] *= 0.15
        return original

    def reset_after_topology_change(self) -> None:
        self.score_sum = None
        self.score_count = None

    def metrics(self) -> dict[str, Any]:
        return {
            "fastgs_vcd_observations": self.observations,
            "fastgs_vcd_last_candidate_ratio": self.last_candidate_ratio,
            "fastgs_vcp_mode": "opacity_prune_with_view_consistency_gate",
        }


def vcp_min_opacity(policy: FastGSPolicy) -> float:
    if policy.last_candidate_ratio is not None and policy.last_candidate_ratio > 0.15:
        return 0.003
    return 0.005
