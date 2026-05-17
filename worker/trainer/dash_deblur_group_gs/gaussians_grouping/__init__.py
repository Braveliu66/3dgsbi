from arguments import ParamGroup
import torch

from .grouping_method import get_under_training_mask


class GroupingParams(ParamGroup):
    def __init__(self, parser):
        self.Grouping = True
        self.grouping_method = "Opacity-weighted"
        self.UTR = 0.78
        self.grouping_from_iter = 4500
        self.grouping_until_iter = 20000
        self.grouping_interval = 600
        self.grouping_freeze_around_pts = 1000
        super().__init__(parser, "Grouping Parameters")

    def extract(self, args):
        group = super().extract(args)
        interval = max(1, int(group.grouping_interval))
        group.grouping_iteration = set(range(int(group.grouping_from_iter), int(group.grouping_until_iter) + 1, interval))
        group.active_count = None
        group.cached_count = None
        return group


def gaussians_grouping_and_caching(iteration, gaussian_model, group_training, _points_caching=None):
    if _points_caching is not None:
        gaussian_model.densification_postfix(**_points_caching)
        _points_caching = None

    utr = 1.0 if iteration == int(group_training.grouping_until_iter) else float(group_training.UTR)
    if utr >= 0.999:
        group_training.active_count = int(gaussian_model.get_xyz.shape[0])
        group_training.cached_count = 0
        return None
    if utr <= 0:
        raise ValueError(f"Under-training ratio {utr} is invalid; expected (0, 1].")

    mask_active = get_under_training_mask(gaussian_model, utr, group_training.grouping_method)
    mask_cache = ~mask_active
    group_training.active_count = int(mask_active.sum().item())
    group_training.cached_count = int(mask_cache.sum().item())

    with torch.no_grad():
        point_caching = {
            "new_xyz": gaussian_model._xyz[mask_cache],
            "new_features_dc": gaussian_model._features_dc[mask_cache],
            "new_features_rest": gaussian_model._features_rest[mask_cache],
            "new_opacities": gaussian_model._opacity[mask_cache],
            "new_scaling": gaussian_model._scaling[mask_cache],
            "new_rotation": gaussian_model._rotation[mask_cache],
        }
    gaussian_model.prune_points(mask_cache)
    return point_caching
