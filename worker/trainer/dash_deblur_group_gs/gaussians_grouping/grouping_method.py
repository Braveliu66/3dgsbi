import torch


def get_under_training_mask(gaussian_model, under_training_ratio, grouping_method="Opacity-weighted"):
    num_gaussian = int(gaussian_model._xyz.shape[0])
    device = gaussian_model._xyz.device
    keep_count = max(1, min(num_gaussian, int(num_gaussian * float(under_training_ratio))))

    if grouping_method == "Random":
        indices = torch.randperm(num_gaussian, device=device)[:keep_count]
    elif grouping_method == "Volume-weighted":
        weights = torch.prod(gaussian_model.get_scaling, dim=-1).clamp_min(1e-12)
        indices = torch.multinomial(weights / weights.sum(), keep_count, replacement=False)
    elif grouping_method == "Opacity-weighted":
        weights = gaussian_model.get_opacity.squeeze(-1).clamp_min(1e-12)
        indices = torch.multinomial(weights / weights.sum(), keep_count, replacement=False)
    elif grouping_method == "Opacity-Volume-weighted":
        weights = gaussian_model.get_opacity.squeeze(-1).clamp_min(1e-12)
        weights = weights * torch.prod(gaussian_model.get_scaling, dim=-1).clamp_min(1e-12)
        indices = torch.multinomial(weights / weights.sum(), keep_count, replacement=False)
    else:
        raise NotImplementedError(f"Grouping method '{grouping_method}' is not implemented.")

    mask = torch.zeros(num_gaussian, dtype=torch.bool, device=device)
    mask[indices] = True
    return mask
