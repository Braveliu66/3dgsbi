from __future__ import annotations

# 本文件为 EDGS 预览训练入口，按 CompVis/EDGS 的 gradio_demo.py 与
# source/trainer.py 关键流程改写：图像选择/COLMAP -> RoMA correspondence init
# -> EDGS Gaussian 训练 -> 输出 Gaussian PLY。
# 上游仓库: https://github.com/CompVis/EDGS
# 固定提交: 9a897645eb47c1b24d4f9e4428cd745927bf1ee1
# 许可证: 非商业学术/个人用途

import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.preview.types import PreviewFailure
from app.preview.utils import VENDOR_ROOT, image_files, prepend_sys_path


Progress = Callable[[str, int, str], None]


def run_edgs_preview(
    *,
    input_dir: Path,
    scene_dir: Path,
    output_dir: Path,
    num_ref_views: int,
    num_corrs_per_view: int,
    num_steps: int,
    max_size: int,
    roma_weight: Path,
    dinov2_weight: Path,
    progress: Progress,
    colmap_max_num_features: int = 4096,
    colmap_max_image_size: int = 1024,
    colmap_min_model_size: int = 10,
    colmap_num_threads: int = 8,
    nns_per_ref: int = 3,
    scaling_factor: float = 0.001,
    fine_profile: bool = False,
) -> dict[str, object]:
    """执行 EDGS 预览训练，返回最终 Gaussian PLY 路径和训练指标。"""

    import torch

    if not torch.cuda.is_available():
        raise PreviewFailure("GPU_RESOURCE_UNAVAILABLE", "EDGS preview requires CUDA")

    edgs_root = VENDOR_ROOT / "edgs"
    gs_root = edgs_root / "gaussian_splatting"
    roma_root = edgs_root / "roma"
    with prepend_sys_path(edgs_root, gs_root, roma_root):
        ensure_edgs_imports()
        import diff_gaussian_rasterization as rasterizer_module
        from hydra import compose, initialize_config_dir
        import hydra
        from romatch import roma_indoor
        from source.trainer import EDGSTrainer
        from source.utils_aux import set_seed
        from source.utils_preprocess import preprocess_frames, run_colmap_on_scene, save_frames_to_scene_dir, select_optimal_frames

        frames = load_edgs_frames(input_dir, max_size=max_size)
        if len(frames) < 3:
            raise PreviewFailure("EDGS_NOT_ENOUGH_IMAGES", "EDGS preview requires at least 3 images")

        selected_count = min(num_ref_views, len(frames))
        resolved_nns_per_ref = max(1, min(int(nns_per_ref), selected_count - 1))
        progress("edgs_select_frames", 26, f"selecting {selected_count} sharp frames for COLMAP")
        scores = preprocess_frames(frames, verbose=False)
        selected = select_optimal_frames(scores, selected_count)
        selected_frames = [frames[index] for index in selected]
        save_frames_to_scene_dir(selected_frames, str(scene_dir))

        progress("edgs_colmap", 34, "running pycolmap feature extraction/mapping")
        run_colmap_on_scene(
            str(scene_dir),
            max_num_features=colmap_max_num_features,
            max_image_size=colmap_max_image_size,
            min_model_size=colmap_min_model_size,
            num_threads=colmap_num_threads,
        )

        progress("edgs_config", 44, f"initializing EDGS trainer ({num_steps} iterations, {resolved_nns_per_ref} NN/ref)")
        with initialize_config_dir(config_dir=str((edgs_root / "configs").resolve()), version_base="1.1"):
            cfg = compose(config_name="train")

        cfg.wandb.mode = "disabled"
        cfg.gs.dataset.model_path = str(output_dir)
        cfg.gs.dataset.source_path = str(scene_dir)
        cfg.gs.dataset.images = "images"
        cfg.gs.opt.TEST_CAM_IDX_TO_LOG = min(12, max(0, selected_count - 1))
        cfg.train.gs_epochs = 30000
        cfg.gs.opt.opacity_reset_interval = 3000 if fine_profile else 1_000_000
        cfg.train.reduce_opacity = True
        cfg.train.no_densify = not fine_profile
        cfg.train.max_lr = not fine_profile
        cfg.init_wC.use = True
        cfg.init_wC.matches_per_ref = num_corrs_per_view
        cfg.init_wC.nns_per_ref = resolved_nns_per_ref
        cfg.init_wC.num_refs = selected_count
        cfg.init_wC.add_SfM_init = False
        cfg.init_wC.scaling_factor = float(scaling_factor)

        set_seed(cfg.seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        generator3dgs = hydra.utils.instantiate(cfg.gs, do_train_test_split=False)
        trainer = EDGSTrainer(
            GS=generator3dgs,
            training_config=cfg.gs.opt,
            device=cfg.device,
            log_wandb=False,
        )
        trainer.saving_iterations = []
        trainer.evaluate_iterations = []

        progress("edgs_corr_init", 52, f"initializing Gaussians with local RoMA correspondences ({selected_count} refs)")
        torch.set_float32_matmul_precision("highest")
        roma_state = torch.load(roma_weight, map_location="cuda:0")
        dinov2_state = torch.load(dinov2_weight, map_location="cuda:0")
        roma_model = roma_indoor(device="cuda:0", weights=roma_state, dinov2_weights=dinov2_state, use_custom_corr=False)
        roma_model.upsample_preds = False
        roma_model.symmetric = False
        trainer.timer.start()
        trainer.init_with_corr(cfg.init_wC, roma_model=roma_model)

        chunk = 10
        completed = 0
        while completed < num_steps:
            current = min(chunk, num_steps - completed)
            cfg.train.gs_epochs = current
            trainer.train(cfg.train)
            completed += current
            progress("edgs_train", 56 + int((completed / max(1, num_steps)) * 26), f"trained {completed}/{num_steps} EDGS iterations")

        progress("edgs_save", 84, "saving EDGS Gaussian PLY")
        trainer.save_model()
        ply_path = output_dir / "point_cloud" / f"iteration_{trainer.gs_step}" / "point_cloud.ply"
        if not ply_path.exists():
            raise PreviewFailure("EDGS_PLY_NOT_FOUND", f"EDGS did not produce point_cloud.ply: {ply_path}")

        peak_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0
        return {
            "ply_path": ply_path,
            "input_frame_count": len(frames),
            "selected_frame_count": len(selected_frames),
            "edgs_num_ref_views_requested": num_ref_views,
            "edgs_selected_ref_views": selected_count,
            "edgs_matches_per_ref": num_corrs_per_view,
            "edgs_nns_per_ref": resolved_nns_per_ref,
            "edgs_preview_steps": num_steps,
            "edgs_colmap_max_num_features": colmap_max_num_features,
            "edgs_colmap_max_image_size": colmap_max_image_size,
            "edgs_colmap_min_model_size": colmap_min_model_size,
            "edgs_scaling_factor": float(scaling_factor),
            "edgs_rasterizer_module": getattr(rasterizer_module, "__file__", None),
            "edgs_rasterizer_fields": list(getattr(rasterizer_module.GaussianRasterizationSettings, "_fields", ())),
            "edgs_iterations": trainer.gs_step,
            "cuda_memory_peak_mb": peak_mb,
        }


def load_edgs_frames(input_dir: Path, *, max_size: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for path in image_files(input_dir):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        scale = max(h, w) / max_size
        if scale > 1:
            frame = cv2.resize(frame, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    return frames


def ensure_edgs_imports() -> None:
    try:
        __import__("pycolmap")
        __import__("diff_gaussian_rasterization")
        __import__("simple_knn")
        __import__("romatch")
    except Exception as exc:  # pragma: no cover - depends on CUDA worker image
        raise PreviewFailure("EDGS_RUNTIME_UNAVAILABLE", f"EDGS runtime dependency missing: {exc}") from exc
    os.environ.setdefault("WANDB_MODE", "disabled")
