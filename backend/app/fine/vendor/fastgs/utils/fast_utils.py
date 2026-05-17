import torch
from PIL import ImageFilter
from gaussian_renderer import render_fastgs
from gaussian_renderer.deblur import render_fastgs_deblur
from .loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
import torchvision.transforms as transforms
import random
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[5]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.fine.fastgs_defaults import FASTGS_LAMBDA_DSSIM, FASTGS_SAMPLE_CAMERAS


def sampling_cameras(my_viewpoint_stack, sample_count=FASTGS_SAMPLE_CAMERAS):
    ''' Randomly sample a given number of cameras from the viewpoint stack'''

    num_cams = min(int(sample_count), len(my_viewpoint_stack))
    camlist = []
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))
    
    return camlist

def get_loss(reconstructed_image, original_image):
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))

    return l1_loss_norm

def compute_photometric_loss(viewpoint_cam, image, lambda_dssim=FASTGS_LAMBDA_DSSIM):
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
    return loss

def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

    return ret_value

def _render_score_image(viewpoint_cam, gaussians, pipe, bg, args, score_renderer, deblur_state, get_flag=None, metric_map=None):
    if score_renderer == "sharp":
        return render_fastgs(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            args.mult,
            get_flag=get_flag,
            metric_map=metric_map,
        )
    if score_renderer == "deblur":
        if deblur_state is None or not getattr(deblur_state, "enabled", False):
            raise RuntimeError("deblur score renderer requires an enabled DeblurState")
        return render_fastgs_deblur(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            args.mult,
            deblur_state,
            get_flag=get_flag,
            metric_map=metric_map,
        )
    raise ValueError(f"Unsupported FastGS score renderer: {score_renderer}")


def compute_gaussian_score_fastgs(
    camlist,
    gaussians,
    pipe,
    bg,
    args,
    DENSIFY = False,
    score_renderer = "sharp",
    score_purpose = None,
    deblur_state = None,
):
    """Compute multi-view consistency scores for Gaussians to guide densification.

    For each camera in `camlist` the function renders the scene and computes a
    photometric loss and a binary metric map of high-error pixels. It accumulates
    per-Gaussian counts of views that flagged the Gaussian and a weighted
    photometric score across views.

    Args:
        camlist (list): list of viewpoint camera objects to render from.
        gaussians: current Gaussian representation (model/state) used for rendering.
        pipe: rendering pipeline/context required by `render`.
        bg: background used for rendering.
        args: runtime config containing thresholds (e.g. `loss_thresh`).
        DENSIFY (bool): whether to compute and return the importance score
            used for densification. If False, only the pruning score is computed.
        score_renderer (str): "sharp" for normal FastGS scoring or "deblur"
            for GTnet-transformed scoring during VCD.
        score_purpose (str): optional caller label ("vcd" or "vcp") for explicit
            routing at call sites.
        deblur_state: DeblurState required when `score_renderer` is "deblur".

    Returns:
        importance_score (Tensor): per-Gaussian integer counts of how many views
            marked the Gaussian as high-error (floor-averaged across views).
            This output is only returned if `DENSIFY` is True.
        pruning_score (Tensor): normalized (0..1) per-Gaussian score used to
            prioritize densification (higher means worse reconstruction consistency).
    """

    _ = score_purpose
    full_metric_counts = None
    full_metric_score = None

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        render_image = _render_score_image(
            my_viewpoint_cam,
            gaussians,
            pipe,
            bg,
            args,
            score_renderer,
            deblur_state,
        )["render"]
        photometric_loss = compute_photometric_loss(my_viewpoint_cam, render_image, getattr(args, "lambda_dssim", FASTGS_LAMBDA_DSSIM))

        gt_image = my_viewpoint_cam.original_image.cuda()
        get_flag = True
        l1_loss_norm = get_loss(render_image, gt_image)
        
        metric_map = (l1_loss_norm > args.loss_thresh).int()

        render_pkg = _render_score_image(
            my_viewpoint_cam,
            gaussians,
            pipe,
            bg,
            args,
            score_renderer,
            deblur_state,
            get_flag=get_flag,
            metric_map=metric_map,
        )

        accum_loss_counts = render_pkg["accum_metric_counts"]

        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts += accum_loss_counts

        if full_metric_score is None:
            full_metric_score = photometric_loss * accum_loss_counts.clone()
        else:
            full_metric_score += photometric_loss * accum_loss_counts

    score_min = torch.min(full_metric_score)
    score_range = torch.max(full_metric_score) - score_min
    if float(score_range.detach().item()) <= 1e-8:
        pruning_score = torch.zeros_like(full_metric_score)
    else:
        pruning_score = (full_metric_score - score_min) / score_range
    
    if DENSIFY:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
    else:
        importance_score = None
    return importance_score, pruning_score
