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
import torch
from random import randint
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
from scene.blur_types import BLUR_SHARP, normalize_blur_type

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


def apply_blur_labels(cameras, blur_label_path):
    with open(blur_label_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    labels = payload.get("labels", payload)
    if not isinstance(labels, dict) or len(labels) == 0:
        raise RuntimeError(f"Blur label file is empty or invalid: {blur_label_path}")

    label_map = {}
    for name, value in labels.items():
        label_value = value.get("blur_type") if isinstance(value, dict) else value
        label_map[str(name)] = label_value
        label_map[os.path.splitext(os.path.basename(str(name)))[0]] = label_value

    for image_id, camera in enumerate(sorted(cameras, key=lambda cam: cam.image_name)):
        camera.image_id = image_id
        label_value = label_map.get(camera.image_name)
        if label_value is None:
            raise RuntimeError(f"Missing blur label for training image: {camera.image_name}")
        camera.blur_type = normalize_blur_type(label_value)

    counts = {0: 0, 1: 0, 2: 0}
    for camera in cameras:
        counts[camera.blur_type] += 1
    print(f"[BlurLabel] sharp={counts[0]} motion={counts[1]} defocus={counts[2]}")
    return len(cameras)


def sharp_camera_subset(cameras):
    return [camera for camera in cameras if getattr(camera, "blur_type", BLUR_SHARP) == BLUR_SHARP]


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, deblur=0):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, deblur)
    scene = Scene(dataset, gaussians)
    bbox = gaussians._xyz.amax(0) - gaussians._xyz.amin(0)

    per_image_blur = bool(opt.per_image_blur) and bool(opt.blur_label_path)
    if deblur and per_image_blur:
        num_images = apply_blur_labels(scene.getTrainCameras() + scene.getTestCameras(), opt.blur_label_path)
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
        gaussians.create_GTnet(hidden=opt.hidden, width=opt.width, pos_delta=opt.use_pos, num_moments=opt.num_moments)
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if first_iter == opt.iterations:
            first_iter -= 1
        gaussians.restore(model_params, opt)

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

    # 按训练轮次自适应优化加点时机（覆盖固定配置）。
    # 这样在短轮次训练中不会加点过晚，在长轮次训练中也不会拖得太后。
    auto_pts_iter = auto_point_addition_iter(opt.iterations)
    if auto_pts_iter != opt.pts_iter:
        print(f"[AutoSchedule] Adjust pts_iter: {opt.pts_iter} -> {auto_pts_iter} (iterations={opt.iterations})")
        opt.pts_iter = auto_pts_iter

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

        render_kwargs = {}
        if per_image_blur:
            render_kwargs["blur_type"] = viewpoint_cam.blur_type
            render_kwargs["image_id"] = viewpoint_cam.image_id
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, deblur=deblur, use_pos=opt.use_pos,
                            lambda_s=opt.lambda_s, lambda_p=opt.lambda_p, max_clamp=opt.max_clamp,
                            **render_kwargs)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        denom = 1 / len(visibility_filter) if type(radii) == list else 1.0
        # 初始化输出目录
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        photo_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if per_image_blur:
            loss = opt.sharp_weight * photo_loss if viewpoint_cam.blur_type == BLUR_SHARP else photo_loss
            if "blur_code" in render_pkg:
                loss = loss + opt.lambda_code * (render_pkg["blur_code"] ** 2).mean()
            if "delta_reg" in render_pkg:
                loss = loss + opt.lambda_delta * render_pkg["delta_reg"]
        else:
            loss = photo_loss
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # 全部完成
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 100 == 0:
                Ll2 = l2_loss(image, gt_image)
                psnr = (-10.0 * np.log(Ll2.cpu()) / np.log(10.0)).item()
                progress_bar.set_postfix({"PSNR": f"{psnr:.{2}f}"})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # 非商业、研究和评估用途。
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
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

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_prune_threshold, scene.cameras_extent, size_threshold, opt.densify_with_depth, opt.prune_range)

                # 全部完成
                if iteration == opt.pts_iter:
                    bbox = pts_max - pts_min
                    volume = bbox[0] * bbox[1] * bbox[2]
                    if opt.pts_rate > 0.0:
                        pts_N_pts = int(min(volume / (opt.pts_rate ** 3), 200000))
                    else:
                        pts_N_pts = opt.pts_N_pts
                    print(f"Allocate {pts_N_pts} points\n")

                    gaussians.add_points(training_args=opt, dist=opt.pts_dist, N=opt.pts_N_intpl, num_pts=pts_N_pts, bound=opt.pts_add_bound)

            # 非商业、研究和评估用途。
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

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
    return tb_writer

def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
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
):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 非商业、研究和评估用途。
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        if per_image_blur:
            validation_configs = (
                {'name': 'test_sharp', 'cameras' : sharp_camera_subset(scene.getTestCameras())},
                {'name': 'train_sharp', 'cameras' : sharp_camera_subset(scene.getTrainCameras())[:5]},
            )
        else:
            validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                                  {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

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
                    
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)


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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[20_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--deblur', type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    # 初始化系统随机状态（RNG）
    safe_state(args.quiet)

    # 启动 GUI 服务并开始训练（当前默认关闭）
    # network_gui.init(args.ip, args.port)  # 如需 GUI 交互可取消注释
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.deblur)

    # 初始化命令行参数解析器
    print("\nTraining complete.")



