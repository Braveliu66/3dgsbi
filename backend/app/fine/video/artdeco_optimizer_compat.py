from __future__ import annotations

from typing import Any


def _tensor_like(value: Any) -> bool:
    return hasattr(value, "numel") and hasattr(value, "to")


def _shape_of(value: Any) -> tuple[int, ...] | str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return type(value).__name__
    return tuple(int(item) for item in shape)


def _lr_for_param(lr: Any, param: Any) -> Any:
    if not _tensor_like(lr):
        return float(lr)
    lr_tensor = lr.to(device=param.device, dtype=param.dtype)
    if lr_tensor.numel() == 1:
        return float(lr_tensor.reshape(-1)[0].item())
    if lr_tensor.numel() == param.numel():
        return lr_tensor.reshape_as(param)
    if getattr(param, "ndim", 0) > 1 and lr_tensor.numel() == param.shape[0]:
        return lr_tensor.reshape(param.shape[0], *([1] * (param.ndim - 1)))
    raise RuntimeError(
        "unsupported ARTDECO Adam learning-rate shape "
        f"lr={_shape_of(lr_tensor)} param={_shape_of(param)}"
    )


def _lr_for_visible_rows(lr: Any, view: Any, visible: Any) -> Any:
    if not _tensor_like(lr):
        return float(lr)
    lr_tensor = lr.to(device=view.device, dtype=view.dtype)
    if lr_tensor.numel() == 1:
        return float(lr_tensor.reshape(-1)[0].item())
    if lr_tensor.numel() == view.shape[0]:
        return lr_tensor.reshape(-1, 1)[visible]
    if lr_tensor.numel() == view.numel():
        return lr_tensor.reshape_as(view)[visible]
    raise RuntimeError(
        "unsupported ARTDECO sparse Adam learning-rate shape "
        f"lr={_shape_of(lr_tensor)} view={_shape_of(view)} visible={_shape_of(visible)}"
    )


def install_artdeco_adam_compat() -> None:
    module = __import__("diff_gaussian_rasterization")
    if not hasattr(module, "adamUpdateBasic"):
        setattr(module, "adamUpdateBasic", adam_update_basic)
    if not hasattr(module, "adamUpdate"):
        setattr(module, "adamUpdate", adam_update_sparse)


def adam_update_basic(param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, *args: Any) -> None:
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    denom = exp_avg_sq.sqrt().add_(eps)
    step = _lr_for_param(lr, param)
    if isinstance(step, float):
        param.addcdiv_(exp_avg, denom, value=-step)
    else:
        param.add_(-step * exp_avg / denom)


def adam_update_sparse(param, grad, exp_avg, exp_avg_sq, visibility, lr, beta1, beta2, eps, n, m, *args: Any) -> None:
    if visibility is None:
        adam_update_basic(param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps)
        return
    visible = visibility.to(dtype=bool, device=param.device)
    view = param.reshape(int(n), int(m))
    grad_view = grad.reshape(int(n), int(m))
    avg_view = exp_avg.reshape(int(n), int(m))
    avg_sq_view = exp_avg_sq.reshape(int(n), int(m))
    if visible.ndim != 1:
        visible = visible.reshape(-1)
    if visible.numel() != view.shape[0]:
        adam_update_basic(param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps)
        return

    avg_view[visible].mul_(beta1).add_(grad_view[visible], alpha=1.0 - beta1)
    avg_sq_view[visible].mul_(beta2).addcmul_(grad_view[visible], grad_view[visible], value=1.0 - beta2)
    denom = avg_sq_view[visible].sqrt().add_(eps)
    step = _lr_for_visible_rows(lr, view, visible)
    if isinstance(step, float):
        view[visible] = view[visible] - step * avg_view[visible] / denom
    else:
        view[visible] = view[visible] - step * avg_view[visible] / denom
