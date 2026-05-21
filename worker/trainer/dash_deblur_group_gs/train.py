#
# 版权所有 (C) 2023, Inria
# GRAPHDECO 研究组, https://team.inria.fr/graphdeco
# 初始化输出目录
#
# 本软件仅可在 LICENSE.md 文件条款下用于
# 非商业、研究和评估用途。
#
# 咨询请联系：george.drettakis@inria.fr
#

import os
import json
import csv
import subprocess
import torch
import time
from random import randint, choice
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import configargparse
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import imageio
import numpy as np
from metrics import compute_img_metric
import torch.nn.functional as F
from scene.blur_types import BLUR_SHARP, BLUR_MOTION, BLUR_DEFOCUS, normalize_blur_type
from scene.gdags import SOURCE_EAP, SOURCE_SFM
from scene.luminance_model import PerImageExposureModel

EXPERIMENT_METRICS_FILE = "experiment_metrics.csv"
FINAL_METRICS_FILE = "final_metrics.txt"
EXPERIMENT_METRIC_FIELDS = [
    "iteration",
    "split",
    "loss_total",
    "loss_l1",
    "loss_photo_raw",
    "loss_photo_weighted",
    "loss_reg",
    "loss_code_reg",
    "loss_delta_reg",
    "loss_lum_reg",
    "psnr",
    "ssim",
    "lpips",
    "num_gaussians",
    "vram_allocated_mb",
    "vram_reserved_mb",
    "iter_time_ms",
    "fps",
    "wall_seconds",
    "blur_type",
    "blur_weight",
    "warmup",
]


def append_experiment_metric(savedir, row):
    path = os.path.join(savedir, EXPERIMENT_METRICS_FILE)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in EXPERIMENT_METRIC_FIELDS})


def cuda_memory_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / (1024.0 * 1024.0), torch.cuda.memory_reserved() / (1024.0 * 1024.0)


def scalar_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return float(value)


def format_metric_value(value, digits=6):
    if value == "" or value is None:
        return ""
    try:
        numeric_value = scalar_value(value)
    except Exception:
        return str(value)
    return f"{numeric_value:.{digits}f}"


def write_final_metrics_document(savedir, iteration, rows):
    if not rows:
        return

    lines = [
        "# Final Training Metrics",
        "",
        f"Iteration: {iteration}",
        "",
        "| Split | PSNR | SSIM | LPIPS | Gaussian Count | Training Seconds | FPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {split} | {psnr} | {ssim} | {lpips} | {num_gaussians} | {train_seconds} | {fps} |".format(
                **row
            )
        )

    lines.extend(["", "Parse-compatible records:", ""])
    for row in rows:
        lines.extend(
            [
                f"FINAL ITERATION {iteration} - {row['split']}",
                f"PSNR: {row['psnr']}",
                f"SSIM: {row['ssim']}",
                f"LPIPS: {row['lpips']}",
                f"NUM_GAUSSIAN: {row['num_gaussians']}",
                f"FPS: {row['fps']}",
                f"TRAIN_SECONDS: {row['train_seconds']}",
                "",
            ]
        )

    path = os.path.join(savedir, FINAL_METRICS_FILE)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[FinalMetrics] wrote {path}")


def find_repo_script(script_name):
    current = os.path.abspath(os.path.dirname(__file__))
    for _ in range(8):
        candidate = os.path.join(current, "scripts", script_name)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def find_project_root():
    current = os.path.abspath(os.path.dirname(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(current, ".git")) or os.path.isdir(os.path.join(current, "scripts")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.getcwd()


def default_thesis_assets_path(model_path):
    if model_path:
        model_dir = os.path.abspath(model_path)
        parent = os.path.dirname(model_dir)
        if os.path.basename(model_dir) == "model" and os.path.basename(parent) == "dash_deblur_group":
            return os.path.join(os.path.dirname(parent), "thesis_assets")
    return os.path.join(find_project_root(), "thesis_assets")


def export_thesis_assets(model_path, output_path=None):
    script = find_repo_script("export_thesis_figures.py")
    if script is None:
        print("[ThesisExport] scripts/export_thesis_figures.py not found; skipped")
        return
    output_path = output_path or default_thesis_assets_path(model_path)
    command = [sys.executable, script, os.path.abspath(model_path), "-o", os.path.abspath(output_path)]
    try:
        subprocess.run(command, check=True)
    except Exception as exc:
        print(f"[ThesisExport] failed: {exc}")
    else:
        print(f"[ThesisExport] wrote assets to {output_path}")


def auto_point_addition_iter(total_iterations: int) -> int:
    """根据总训练轮次自适应计算加点时机。

    经验策略：
    - 目标比例约为总轮次的 12%
    - 下限 800：避免过早加点导致几何尚未稳定
    - 上限 6000：避免在超长训练中过晚加点
    """
    if total_iterations <= 0:
        return 2500
    raw = int(total_iterations * 0.12)
    return max(800, min(6000, raw))


def auto_densify_until_iter(total_iterations: int, densify_from_iter: int, densification_interval: int) -> int:
    if total_iterations <= 0:
        return 0
    interval = max(1, int(densification_interval))
    from_iter = max(0, int(densify_from_iter))
    target = max(int(total_iterations * 0.6), from_iter + interval)
    return min(int(total_iterations), target)


def uses_legacy_random_point_defaults(opt) -> bool:
    return (
        int(opt.pts_iter) == 2500
        and abs(float(opt.pts_rate) - 1.1) < 1e-9
        and int(opt.pts_N_pts) == 200000
    )


def random_point_addition_enabled(opt) -> bool:
    return (
        int(opt.pts_iter) <= int(opt.iterations)
        and (float(opt.pts_rate) > 0.0 or int(opt.pts_N_pts) > 0)
    )


def apply_blur_labels(cameras, blur_label_path):
    with open(blur_label_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    labels = payload.get("labels", payload)
    if not isinstance(labels, dict) or len(labels) == 0:
        raise RuntimeError(f"Blur label file is empty or invalid: {blur_label_path}")

    label_map = {}
    for name, value in labels.items():
        label_record = value if isinstance(value, dict) else {"blur_type": value}
        label_map[str(name)] = label_record
        label_map[os.path.splitext(os.path.basename(str(name)))[0]] = label_record

    for image_id, camera in enumerate(sorted(cameras, key=lambda cam: cam.image_name)):
        camera.image_id = image_id
        label_record = label_map.get(camera.image_name)
        if label_record is None:
            raise RuntimeError(f"Missing blur label for training image: {camera.image_name}")
        label_value = label_record.get("blur_type")
        camera.blur_type = normalize_blur_type(label_value)
        blur_weight = label_record.get("deblur_weight", label_record.get("deblurweight", label_record.get("blur_weight")))
        camera.blur_weight = float(blur_weight) if blur_weight is not None else 1.0

    counts = {0: 0, 1: 0, 2: 0}
    for camera in cameras:
        counts[camera.blur_type] += 1
    print(f"[BlurLabel] sharp={counts[0]} motion={counts[1]} defocus={counts[2]}")
    return len(cameras)


def ensure_camera_image_ids(cameras):
    for image_id, camera in enumerate(sorted(cameras, key=lambda cam: cam.image_name)):
        if not hasattr(camera, "image_id"):
            camera.image_id = image_id
        if not hasattr(camera, "blur_type"):
            camera.blur_type = BLUR_SHARP
    return len(cameras)


def sharp_camera_subset(cameras):
    return [camera for camera in cameras if getattr(camera, "blur_type", BLUR_SHARP) == BLUR_SHARP]


def include_sharp_test_cameras_in_train(scene):
    train_cameras = scene.getTrainCameras()
    train_names = {camera.image_name for camera in train_cameras}
    added = 0
    for camera in scene.getTestCameras():
        if getattr(camera, "blur_type", BLUR_SHARP) == BLUR_SHARP and camera.image_name not in train_names:
            train_cameras.append(camera)
            train_names.add(camera.image_name)
            added += 1
    if added:
        print(f"[BlurLabel] added {added} sharp test cameras to training set")
    return added


def sharp_or_low_blur_weight_subset(cameras, fallback_limit=None):
    sharp_cameras = sharp_camera_subset(cameras)
    if sharp_cameras:
        return sharp_cameras
    sorted_cameras = sorted(cameras, key=lambda camera: getattr(camera, "blur_weight", 1.0))
    if fallback_limit is None:
        return sorted_cameras
    return sorted_cameras[:fallback_limit]


def compute_photo_loss(pred, gt, opt):
    Ll1 = l1_loss(pred, gt)
    return (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(pred, gt)), Ll1


def blur_type_weight(blur_type, opt):
    blur_type = normalize_blur_type(blur_type)
    if blur_type == BLUR_SHARP:
        return float(opt.sharp_weight)
    if blur_type == BLUR_DEFOCUS:
        return float(opt.defocus_weight)
    return float(opt.motion_weight)


def compute_regularization_loss(render_pkg, opt, luminance_model=None, luminance_active=False):
    ref = render_pkg["render"]
    code_reg = torch.zeros((), device=ref.device)
    delta_reg = torch.zeros((), device=ref.device)
    lum_reg = torch.zeros((), device=ref.device)
    if "blur_code" in render_pkg and render_pkg["blur_code"].numel() > 0:
        code_reg = opt.lambda_code * (render_pkg["blur_code"] ** 2).mean()
    if "delta_reg" in render_pkg:
        delta_reg = opt.lambda_delta * render_pkg["delta_reg"]
    if luminance_active and luminance_model is not None:
        lum_reg = luminance_model.regularization_loss(
            lambda_gain=opt.luminance_lambda_gain,
            lambda_bias=opt.luminance_lambda_bias,
        )
    return code_reg, delta_reg, lum_reg, code_reg + delta_reg + lum_reg


def run_gdags_canonical_probe(gaussians, sharp_cameras, pipe, background, opt, optimizer, luminance_optimizer=None):
    if gaussians.gdags is None:
        return
    if not sharp_cameras:
        if not getattr(gaussians, "_gdags_no_sharp_warned", False):
            print("[GDAGS] canonical probe skipped: no sharp cameras")
            gaussians._gdags_no_sharp_warned = True
        return

    probe_cam = choice(sharp_cameras)
    optimizer.zero_grad(set_to_none=True)
    if luminance_optimizer is not None:
        luminance_optimizer.zero_grad(set_to_none=True)

    probe_pkg = render(probe_cam, gaussians, pipe, background, deblur=0, force_original_backend=True)
    proxy_loss, _ = compute_photo_loss(probe_pkg["render"], probe_cam.original_image.cuda(), opt)
    vsp = probe_pkg["gdags_stats_viewspace_points"]
    vis = probe_pkg["gdags_stats_visibility"]
    grad_vsp = torch.autograd.grad(
        outputs=proxy_loss,
        inputs=vsp,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )[0]
    gaussians.gdags.update_canonical_stats(vsp, vis, grad=grad_vsp)

    optimizer.zero_grad(set_to_none=True)
    if luminance_optimizer is not None:
        luminance_optimizer.zero_grad(set_to_none=True)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, deblur=0):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, deblur)
    scene = Scene(dataset, gaussians)
    bbox = gaussians._xyz.amax(0) - gaussians._xyz.amin(0)
    all_cameras = scene.getTrainCameras() + scene.getTestCameras()

    per_image_blur = bool(opt.per_image_blur) and bool(opt.blur_label_path)
    if deblur and per_image_blur:
        num_images = apply_blur_labels(all_cameras, opt.blur_label_path)
        include_sharp_test_cameras_in_train(scene)
        gaussians.create_conditional_GTnets(
            num_images=num_images,
            hidden=opt.hidden,
            width=opt.width,
            code_dim=opt.blur_code_dim,
            num_moments=opt.num_moments,
        )
        print(f"[BlurCode] code_dim={opt.blur_code_dim} images={num_images}")
    else:
        per_image_blur = False
        num_images = ensure_camera_image_ids(all_cameras)
        gaussians.create_GTnet(hidden=opt.hidden, width=opt.width, pos_delta=opt.use_pos, num_moments=opt.num_moments)

    if bool(opt.luminance_enable):
        if (
            opt.luminance_mode != "exposure_gain_bias"
            or bool(opt.luminance_per_channel)
            or bool(opt.luminance_matrix_enable)
            or bool(opt.luminance_curve_enable)
        ):
            raise RuntimeError("First luminance implementation only supports scalar exposure_gain_bias")
        luminance_model = PerImageExposureModel(num_images).cuda()
        luminance_optimizer = torch.optim.Adam(luminance_model.parameters(), lr=opt.luminance_lr, eps=1e-15)
        print(f"[Luminance] exposure_gain_bias images={num_images} start_iter={opt.luminance_start_iter}")
    else:
        luminance_model = None
        luminance_optimizer = None

    if bool(opt.gdags_stats_enable) or bool(opt.gdags_enable):
        source_type = SOURCE_EAP if getattr(dataset, "pc_name", "points3D") == "points3D_eap" else SOURCE_SFM
        gaussians.create_gdags(source_type=source_type)
        print(f"[GDAGS] stats_enable={bool(opt.gdags_stats_enable)} enable={bool(opt.gdags_enable)}")
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if first_iter == opt.iterations:
            first_iter -= 1
        gaussians.restore(model_params, opt)
        if bool(opt.gdags_stats_enable) or bool(opt.gdags_enable):
            source_type = SOURCE_EAP if getattr(dataset, "pc_name", "points3D") == "points3D_eap" else SOURCE_SFM
            gaussians.create_gdags(source_type=source_type)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    viewpoint_stack = scene.getTrainCameras().copy()

    pts_max = gaussians._xyz.amax(0)
    pts_min = gaussians._xyz.amin(0)

    if uses_legacy_random_point_defaults(opt):
        print("[AutoSchedule] Disable legacy random add_points default (pts_iter=2500 pts_rate=1.1 pts_N_pts=200000)")
        opt.pts_iter = int(opt.iterations) + 1
        opt.pts_rate = 0.0
        opt.pts_N_pts = 0

    if random_point_addition_enabled(opt):
        auto_pts_iter = auto_point_addition_iter(opt.iterations)
    else:
        auto_pts_iter = int(opt.pts_iter)
    if random_point_addition_enabled(opt) and auto_pts_iter != opt.pts_iter:
        print(f"[AutoSchedule] Adjust pts_iter: {opt.pts_iter} -> {auto_pts_iter} (iterations={opt.iterations})")
        opt.pts_iter = auto_pts_iter

    auto_densify_until = auto_densify_until_iter(opt.iterations, opt.densify_from_iter, opt.densification_interval)
    if opt.densify_until_iter > auto_densify_until:
        print(f"[AutoSchedule] Adjust densify_until_iter: {opt.densify_until_iter} -> {auto_densify_until} (iterations={opt.iterations})")
        opt.densify_until_iter = auto_densify_until

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # 每 1000 次迭代提升一次 SH 阶数，直到最大阶。
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 非商业、研究和评估用途。
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        img_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(img_idx)
        
        # 初始化输出目录
        if (iteration - 1) == debug_from:
            pipe.debug = True

        warmup_active = (
            bool(opt.pre_deblur_warmup_enable)
            and int(opt.pre_deblur_warmup_iters) > 0
            and iteration <= int(opt.pre_deblur_warmup_iters)
        )
        render_kwargs = {}
        if per_image_blur and not warmup_active:
            render_kwargs["blur_type"] = viewpoint_cam.blur_type
            render_kwargs["image_id"] = viewpoint_cam.image_id
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, deblur=0 if warmup_active else deblur, use_pos=opt.use_pos,
                            lambda_s=opt.lambda_s, lambda_p=opt.lambda_p, max_clamp=opt.max_clamp,
                            force_original_backend=(iteration < opt.densify_until_iter),
                            **render_kwargs)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        denom = 1 / len(visibility_filter) if type(radii) == list else 1.0
        # 初始化输出目录
        gt_image = viewpoint_cam.original_image.cuda()
        image_for_loss = image
        luminance_active = (
            not warmup_active
            and
            luminance_model is not None
            and iteration >= int(opt.luminance_start_iter)
            and hasattr(viewpoint_cam, "image_id")
        )
        if luminance_active:
            image_for_loss = luminance_model(image, viewpoint_cam.image_id)

        photo_loss_raw, Ll1 = compute_photo_loss(image_for_loss, gt_image, opt)
        current_blur_type = BLUR_SHARP if warmup_active else getattr(viewpoint_cam, "blur_type", BLUR_SHARP)
        weight = blur_type_weight(current_blur_type, opt) if per_image_blur and not warmup_active else 1.0
        photo_loss_weighted = weight * photo_loss_raw
        code_reg, delta_reg, lum_reg, reg_loss = compute_regularization_loss(
            render_pkg,
            opt,
            luminance_model=luminance_model,
            luminance_active=luminance_active,
        )
        loss = photo_loss_weighted + reg_loss
        loss.backward()
        iter_end.record()

        if gaussians.gdags is not None and bool(opt.gdags_stats_enable):
            gdags_points = render_pkg.get("gdags_stats_viewspace_points", viewspace_point_tensor)
            gdags_visibility = render_pkg.get("gdags_stats_visibility", visibility_filter)
            if warmup_active:
                gaussians.gdags.update_canonical_stats(gdags_points, gdags_visibility)
            else:
                gaussians.gdags.update_blur_stats(gdags_points, gdags_visibility, current_blur_type)

        with torch.no_grad():
            # 全部完成
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            train_psnr = ""
            if iteration % 100 == 0:
                Ll2 = l2_loss(image, gt_image)
                train_psnr = (-10.0 * np.log(Ll2.cpu()) / np.log(10.0)).item()
                progress_bar.set_postfix({"PSNR": f"{train_psnr:.{2}f}", "raw": f"{photo_loss_raw.item():.4f}", "reg": f"{reg_loss.item():.4f}"})
                progress_bar.update(100)
                print(
                    f"\n[Loss] iter={iteration} warmup={int(warmup_active)} blur_type={int(current_blur_type)} "
                    f"weight={weight:.3f} photo_raw={photo_loss_raw.item():.6f} "
                    f"photo_weighted={photo_loss_weighted.item():.6f} reg={reg_loss.item():.6f} "
                    f"code={code_reg.item():.6f} delta={delta_reg.item():.6f} lum={lum_reg.item():.6f}"
                )
            if iteration == opt.iterations:
                progress_bar.close()

            # 非商业、研究和评估用途。
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                photo_loss_raw,
                photo_loss_weighted,
                code_reg,
                delta_reg,
                lum_reg,
                reg_loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                dataset.model_path,
                per_image_blur=per_image_blur,
                deblur=deblur,
                use_pos=opt.use_pos,
                lambda_s=opt.lambda_s,
                lambda_p=opt.lambda_p,
                max_clamp=opt.max_clamp,
                final_iteration=(iteration == opt.iterations),
                current_blur_type=current_blur_type,
                blur_weight=weight,
                warmup_active=warmup_active,
                train_psnr=train_psnr,
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 初始化输出目录
            if iteration < opt.densify_until_iter:
                # 非商业、研究和评估用途。
                if type(visibility_filter) == list:
                    gaussians.max_radii2D[visibility_filter[0]] = torch.max(gaussians.max_radii2D[visibility_filter[0]], radii[0][visibility_filter[0]])
                else:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, denom)

                if (
                    warmup_active
                    and bool(opt.pre_deblur_warmup_densify)
                    and iteration >= int(opt.pre_deblur_warmup_densify_from_iter)
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_warmup_clone_split(
                        opt.densify_grad_threshold,
                        scene.cameras_extent,
                        current_iter=iteration,
                        gdags_protect_iters=opt.gdags_newborn_protect_iters,
                    )
                elif (not warmup_active) and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.densify_prune_threshold,
                        scene.cameras_extent,
                        size_threshold,
                        opt.densify_with_depth,
                        opt.prune_range,
                        current_iter=iteration,
                        gdags_protect_iters=opt.gdags_newborn_protect_iters,
                    )

                # 全部完成
                if random_point_addition_enabled(opt) and iteration == opt.pts_iter:
                    bbox = pts_max - pts_min
                    volume = bbox[0] * bbox[1] * bbox[2]
                    if opt.pts_rate > 0.0:
                        pts_N_pts = int(min(volume / (opt.pts_rate ** 3), 200000))
                    else:
                        pts_N_pts = opt.pts_N_pts
                    print(f"Allocate {pts_N_pts} points\n")

                    if pts_N_pts > 0:
                        gaussians.add_points(training_args=opt, dist=opt.pts_dist, N=opt.pts_N_intpl, num_pts=pts_N_pts, bound=opt.pts_add_bound)

            # 非商业、研究和评估用途。
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                if luminance_optimizer is not None and luminance_active:
                    luminance_optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if luminance_optimizer is not None:
                    luminance_optimizer.zero_grad(set_to_none=True)

                if (
                    gaussians.gdags is not None
                    and bool(opt.gdags_stats_enable)
                    and iteration >= int(opt.gdags_start_iter)
                    and int(opt.gdags_probe_interval) > 0
                    and iteration % int(opt.gdags_probe_interval) == 0
                ):
                    with torch.enable_grad():
                        run_gdags_canonical_probe(
                            gaussians,
                            sharp_camera_subset(scene.getTrainCameras()),
                            pipe,
                            background,
                            opt,
                            gaussians.optimizer,
                            luminance_optimizer=luminance_optimizer,
                        )
                if gaussians.gdags is not None:
                    gaussians.gdags.step_age()
                    gaussians.gdags.assert_shape(gaussians.get_xyz.shape[0])

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        tag = args.expname if args.expname != None else unique_str[0:10]
        args.model_path = os.path.join("./output/", tag)
        
    # 初始化输出目录
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    os.makedirs(args.model_path+"/TEST", exist_ok = True)
    os.makedirs(args.model_path+"/TRAIN", exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建 TensorBoard 记录器
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    # 记录训练开始时间（module 级变量），用于最终汇总
    try:
        global TRAIN_START_TIME
        TRAIN_START_TIME = time.time()
    except Exception:
        TRAIN_START_TIME = None
    return tb_writer

def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    photo_loss_raw,
    photo_loss_weighted,
    code_reg,
    delta_reg,
    lum_reg,
    reg_loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene : Scene,
    renderFunc,
    renderArgs,
    savedir,
    per_image_blur=False,
    deblur=0,
    use_pos=False,
    lambda_s=0.01,
    lambda_p=0.01,
    max_clamp=1.1,
    final_iteration: bool = False,
    current_blur_type=BLUR_SHARP,
    blur_weight=1.0,
    warmup_active=False,
    train_psnr="",
):
    train_start = globals().get('TRAIN_START_TIME')
    wall_seconds = time.time() - train_start if train_start else ""
    vram_allocated_mb, vram_reserved_mb = cuda_memory_mb()
    fps = 1000.0 / elapsed if elapsed and elapsed > 0 else ""
    append_experiment_metric(
        savedir,
        {
            "iteration": iteration,
            "split": "train",
            "loss_total": loss.item(),
            "loss_l1": Ll1.item(),
            "loss_photo_raw": photo_loss_raw.item(),
            "loss_photo_weighted": photo_loss_weighted.item(),
            "loss_reg": reg_loss.item(),
            "loss_code_reg": code_reg.item(),
            "loss_delta_reg": delta_reg.item(),
            "loss_lum_reg": lum_reg.item(),
            "psnr": train_psnr,
            "num_gaussians": int(scene.gaussians.get_xyz.shape[0]),
            "vram_allocated_mb": vram_allocated_mb,
            "vram_reserved_mb": vram_reserved_mb,
            "iter_time_ms": elapsed,
            "fps": fps,
            "wall_seconds": wall_seconds,
            "blur_type": int(current_blur_type),
            "blur_weight": float(blur_weight),
            "warmup": int(bool(warmup_active)),
        },
    )
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 非商业、研究和评估用途。
    # 在测试迭代或最后一次迭代也执行评估，确保训练结束时有最终指标。
    if iteration in testing_iterations or final_iteration:
        torch.cuda.empty_cache()
        if per_image_blur:
            validation_configs = (
                {'name': 'test', 'cameras' : sharp_or_low_blur_weight_subset(scene.getTestCameras(), fallback_limit=5)},
                {'name': 'train', 'cameras' : sharp_or_low_blur_weight_subset(scene.getTrainCameras(), fallback_limit=5)[:5]},
            )
        else:
            validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                                  {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        final_metric_rows = []
        for config in validation_configs:
            _type = config["name"].upper()
            os.makedirs(f"{savedir}/{_type}", exist_ok=True)
            if _type == "TEST":
                with open(f"{savedir}/psnr.txt", "a") as f:
                    f.write("[ITER {}] NUM GAUSSIAN: {} \n".format(iteration, scene.gaussians.get_xyz.shape[0]))
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_kwargs = {}
                    if per_image_blur:
                        render_kwargs = {
                            "deblur": deblur,
                            "use_pos": use_pos,
                            "blur_type": viewpoint.blur_type,
                            "image_id": viewpoint.image_id,
                            "lambda_s": lambda_s,
                            "lambda_p": lambda_p,
                            "max_clamp": max_clamp,
                        }
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, **render_kwargs)["render"], 0.0, 1.0)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    image_metric = image.permute(1,2,0)
                    gt_image_metic = gt_image.permute(1,2,0)
                    ssim_test += compute_img_metric(image_metric, gt_image_metic, 'ssim')
                    lpips = compute_img_metric(image_metric, gt_image_metic, 'lpips')
                    if isinstance(lpips, torch.Tensor):
                        lpips = lpips.item()
                    lpips_test += lpips
                        
                    imageio.imwrite(f"{savedir}/{_type}/img_{iteration}_{idx:03d}.png", (image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
                    if iteration == testing_iterations[0]:
                        imageio.imwrite(f"{savedir}/{_type}/GT_{idx:03d}.png", (gt_image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])     
                lpips_test /= len(config['cameras'])    

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                print("[ITER {}] Evaluating {}: SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], ssim_test, lpips_test))
                with open(f"{savedir}/psnr.txt", "a") as f:
                    f.write("[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, config['name'], l1_test, psnr_test))
                    f.write("[ITER {}] Evaluating {}: SSIM {:.4f} LPIPS {:.4f}\n".format(iteration, config['name'], ssim_test, lpips_test))
                # 如果这是最终迭代，写入一份 final_metrics.txt，包含训练耗时与高斯点数
                append_experiment_metric(
                    savedir,
                    {
                        "iteration": iteration,
                        "split": config["name"],
                        "loss_l1": scalar_value(l1_test),
                        "psnr": scalar_value(psnr_test),
                        "ssim": scalar_value(ssim_test),
                        "lpips": scalar_value(lpips_test),
                        "num_gaussians": int(scene.gaussians.get_xyz.shape[0]),
                        "vram_allocated_mb": vram_allocated_mb,
                        "vram_reserved_mb": vram_reserved_mb,
                        "fps": fps,
                        "wall_seconds": wall_seconds,
                    },
                )
                if final_iteration:
                    total_points = int(scene.gaussians.get_xyz.shape[0])
                    final_metric_rows.append(
                        {
                            "split": config["name"],
                            "psnr": format_metric_value(psnr_test),
                            "ssim": format_metric_value(ssim_test),
                            "lpips": format_metric_value(lpips_test),
                            "num_gaussians": str(total_points),
                            "fps": format_metric_value(fps, digits=3) if fps != "" else "",
                            "train_seconds": format_metric_value(wall_seconds, digits=3) if wall_seconds != "" else "unknown",
                        }
                    )
                    
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)


        if final_iteration:
            write_final_metrics_document(savedir, iteration, final_metric_rows)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # 初始化命令行参数解析器
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--deblur', type=int, default=1)
    parser.add_argument("--skip_thesis_export", action="store_true")
    parser.add_argument("--thesis_assets_path", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    # 初始化系统随机状态（RNG）
    safe_state(args.quiet)

    # 启动 GUI 服务并开始训练（当前默认关闭）
    # network_gui.init(args.ip, args.port)  # 如需 GUI 交互可取消注释
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    model_params = lp.extract(args)
    training(model_params, op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.deblur)
    if not args.skip_thesis_export:
        export_thesis_assets(model_params.model_path, args.thesis_assets_path)

    # 初始化命令行参数解析器
    print("\nTraining complete.")



