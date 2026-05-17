from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class SceneProfile:
    name: str
    size_prune_world_scale_ratio: float
    late_prune_world_scale_ratio: float
    final_prune_world_scale_ratio: float
    max_prune_fraction_per_step: float
    densify_grad_threshold: float
    densify_abs_grad_threshold: float
    deblur_loss_weight_init: float
    deblur_loss_weight_final: float
    knn_extra_points_enabled: bool
    knn_k: int
    knn_depth_limit: float
    late_prune_from: int
    late_prune_until: int
    late_prune_interval: int
    late_prune_min_opacity: float
    late_prune_score_thresh: float
    final_prune_min_opacity: float
    final_prune_score_thresh: float
    sharp_refine_from: int
    sharp_refine_clear_only: str
    densify_until_iter: int
    deblur_warmup_iters: int


INDOOR_PROFILE = SceneProfile(
    name="indoor",
    size_prune_world_scale_ratio=0.15,
    late_prune_world_scale_ratio=0.12,
    final_prune_world_scale_ratio=0.10,
    max_prune_fraction_per_step=0.02,
    densify_grad_threshold=0.00015,
    densify_abs_grad_threshold=0.00035,
    deblur_loss_weight_init=1.0,
    deblur_loss_weight_final=0.2,
    knn_extra_points_enabled=True,
    knn_k=6,
    knn_depth_limit=0.8,
    late_prune_from=28_000,
    late_prune_until=30_000,
    late_prune_interval=4_000,
    late_prune_min_opacity=0.003,
    late_prune_score_thresh=0.97,
    final_prune_min_opacity=0.003,
    final_prune_score_thresh=0.95,
    sharp_refine_from=28_000,
    sharp_refine_clear_only="false",
    densify_until_iter=26_000,
    deblur_warmup_iters=5_000,
)


OUTDOOR_PROFILE = SceneProfile(
    name="outdoor",
    size_prune_world_scale_ratio=0.06,
    late_prune_world_scale_ratio=0.06,
    final_prune_world_scale_ratio=0.05,
    max_prune_fraction_per_step=0.12,
    densify_grad_threshold=0.0002,
    densify_abs_grad_threshold=0.0005,
    deblur_loss_weight_init=1.0,
    deblur_loss_weight_final=0.3,
    knn_extra_points_enabled=True,
    knn_k=8,
    knn_depth_limit=0.6,
    late_prune_from=20_000,
    late_prune_until=30_000,
    late_prune_interval=2_500,
    late_prune_min_opacity=0.02,
    late_prune_score_thresh=0.97,
    final_prune_min_opacity=0.02,
    final_prune_score_thresh=0.95,
    sharp_refine_from=24_000,
    sharp_refine_clear_only="false",
    densify_until_iter=20_000,
    deblur_warmup_iters=5_000,
)


PROFILES: dict[str, SceneProfile] = {
    "auto": INDOOR_PROFILE,
    "mixed_balanced": INDOOR_PROFILE,
    "indoor": INDOOR_PROFILE,
    "indoor_full": INDOOR_PROFILE,
    "outdoor": OUTDOOR_PROFILE,
    "outdoor_full": OUTDOOR_PROFILE,
    "outdoor_fast_clean": OUTDOOR_PROFILE,
}


class DeblurFastGSSchedule:
    def __init__(
        self,
        profile: SceneProfile,
        total_iterations: int,
        warmup_end: int,
        deblur_phase_end: int,
        consolidate_end: int,
        sharp_refine_start: int,
        num_blurred_frames: int,
        num_total_frames: int,
    ) -> None:
        self.profile = profile
        self.total_iterations = int(total_iterations)
        self.warmup_end = int(warmup_end)
        self.deblur_phase_end = int(deblur_phase_end)
        self.consolidate_end = int(consolidate_end)
        self.sharp_refine_start = int(sharp_refine_start)
        self.blurred_ratio = num_blurred_frames / max(num_total_frames, 1)
        self._loss_history: list[float] = []
        self._loss_ema = 0.05
        self._loss_ema_alpha = 0.05
        self._prune_paused_until = 0

    def step(self, iteration: int, current_loss: float) -> dict[str, Any]:
        current_loss = float(current_loss)
        self._loss_ema = self._loss_ema_alpha * current_loss + (1.0 - self._loss_ema_alpha) * self._loss_ema
        self._loss_history.append(current_loss)
        phase = self._get_phase(iteration)
        return {
            "phase": phase,
            "deblur_loss_weight": self._get_deblur_loss_weight(iteration, phase),
            "mlp_active": self._get_mlp_active(iteration, phase),
            "allow_densify": phase in (1, 2, 3),
            "prune_mode": self._get_prune_mode(phase),
            "prune_paused": iteration < self._prune_paused_until,
            "loss_ema": self._loss_ema,
        }

    def safe_prune_mask(
        self,
        iteration: int,
        gaussians: Any,
        raw_prune_mask: "torch.Tensor",
        prune_mode: str,
    ) -> "torch.Tensor":
        import torch

        if self._loss_spiked(iteration):
            return torch.zeros_like(raw_prune_mask)
        if iteration < self._prune_paused_until:
            return torch.zeros_like(raw_prune_mask)

        current_count = int(raw_prune_mask.shape[0])
        max_remove = int(current_count * self.profile.max_prune_fraction_per_step)
        if max_remove <= 0:
            return torch.zeros_like(raw_prune_mask)

        prune_indices = torch.where(raw_prune_mask)[0]
        if int(prune_indices.shape[0]) <= max_remove:
            return raw_prune_mask

        try:
            opacity = gaussians.get_opacity[prune_indices].squeeze(-1)
            selected = prune_indices[torch.argsort(opacity)[:max_remove]]
        except Exception:
            selected = prune_indices[torch.randperm(prune_indices.shape[0], device=prune_indices.device)[:max_remove]]

        safe_mask = torch.zeros_like(raw_prune_mask)
        safe_mask[selected] = True
        print(
            f"[SAFE_PRUNE] iter {iteration}: capped {int(prune_indices.shape[0])} -> "
            f"{max_remove} ({int(prune_indices.shape[0]) - max_remove} deferred)"
        )
        return safe_mask

    def _loss_spiked(self, iteration: int) -> bool:
        if len(self._loss_history) < 10:
            return False
        recent = sorted(self._loss_history[-10:])
        median_val = recent[len(recent) // 2]
        if median_val <= 0:
            return False
        if self._loss_ema > median_val * 2.5:
            self._prune_paused_until = iteration + 2_000
            print(
                f"[SAFE_PRUNE] loss spike detected (ema={self._loss_ema:.4f}, "
                f"median={median_val:.4f}); pausing prune until iter {self._prune_paused_until}"
            )
            return True
        return False

    def _get_phase(self, iteration: int) -> int:
        if iteration < self.warmup_end:
            return 1
        if iteration < self.deblur_phase_end:
            return 2
        if iteration < self.consolidate_end:
            return 3
        return 4

    def _get_deblur_loss_weight(self, iteration: int, phase: int) -> float:
        if phase == 1:
            return 0.0
        if phase == 2:
            t = (iteration - self.warmup_end) / max(self.deblur_phase_end - self.warmup_end, 1)
            return min(1.0, t * 2.0) * self.profile.deblur_loss_weight_init
        if phase == 3:
            t = (iteration - self.deblur_phase_end) / max(self.consolidate_end - self.deblur_phase_end, 1)
            return self.profile.deblur_loss_weight_init + (
                self.profile.deblur_loss_weight_final - self.profile.deblur_loss_weight_init
            ) * t
        return 0.0

    def _get_mlp_active(self, iteration: int, phase: int) -> bool:
        return phase > 1 and iteration < self.sharp_refine_start

    @staticmethod
    def _get_prune_mode(phase: int) -> str:
        if phase in (1, 2):
            return "conservative"
        if phase == 3:
            return "adaptive"
        return "fine"


def profile_for_hint(frontend_hint: str = "") -> SceneProfile:
    profile_name = auto_detect_scene_profile(0, 0, frontend_hint=frontend_hint)
    return PROFILES.get(profile_name, INDOOR_PROFILE)


def auto_detect_scene_profile(
    sfm_sparse_points: int,
    sfm_registered_images: int,
    cameras_xyz: Any = None,
    scene_extent: float = 0.0,
    frontend_hint: str = "",
) -> str:
    hint = str(frontend_hint or "").strip().lower()
    if hint in {"indoor", "indoor_full", "mixed_balanced"}:
        return "indoor_full"
    if hint in {"outdoor", "outdoor_full", "outdoor_fast_clean"}:
        return "outdoor_full"

    if cameras_xyz is not None and cameras_xyz.shape[0] > 3 and scene_extent > 0:
        camera_heights = cameras_xyz[:, 1]
        height_range = float(camera_heights.max() - camera_heights.min())
        if height_range > scene_extent * 0.3:
            return "outdoor_full"

    if scene_extent > 0 and sfm_sparse_points > 0:
        density = sfm_sparse_points / (scene_extent**2 + 1e-6)
        if density < 0.5 and sfm_registered_images > 0:
            return "outdoor_full"

    return "indoor_full"
