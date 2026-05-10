from __future__ import annotations

from typing import Any


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
    param.addcdiv_(exp_avg, denom, value=-float(lr))


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
    if hasattr(lr, "ndim") and lr.ndim > 0 and lr.numel() == view.shape[0]:
        step = lr[visible].reshape(-1, 1)
        view[visible] = view[visible] - step * avg_view[visible] / denom
    else:
        view[visible] = view[visible] - float(lr) * avg_view[visible] / denom
