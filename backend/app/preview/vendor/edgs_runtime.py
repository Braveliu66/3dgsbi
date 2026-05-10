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

from app.preview.types import PreviewFailure
from app.preview.utils import VENDOR_ROOT, image_files, prepend_sys_path


Progress = Callable[[str, int, str], None]


def run_edgs_preview(
    *,
    scene_dir: Path,
    output_dir: Path,
    num_ref_views: int,
    num_corrs_per_view: int,
    num_steps: int,
    roma_weight: Path,
    dinov2_weight: Path,
    progress: Progress,
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

        image_count = _prepared_scene_image_count(scene_dir)
        if image_count < 3:
            raise PreviewFailure("EDGS_NOT_ENOUGH_IMAGES", "EDGS preview requires at least 3 registered images")

        selected_count = min(num_ref_views, image_count)
        resolved_nns_per_ref = max(1, min(int(nns_per_ref), selected_count - 1))
        progress("edgs_config", 68, f"loading LiteVGGT scene for EDGS ({num_steps} iterations, {resolved_nns_per_ref} NN/ref)")
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

        progress("edgs_corr_init", 70, f"initializing Gaussians with local RoMA correspondences ({selected_count} refs)")
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
            progress("edgs_train", 72 + int((completed / max(1, num_steps)) * 12), f"trained {completed}/{num_steps} EDGS iterations")

        progress("edgs_save", 84, "saving EDGS Gaussian PLY")
        trainer.save_model()
        ply_path = output_dir / "point_cloud" / f"iteration_{trainer.gs_step}" / "point_cloud.ply"
        if not ply_path.exists():
            raise PreviewFailure("EDGS_PLY_NOT_FOUND", f"EDGS did not produce point_cloud.ply: {ply_path}")

        peak_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0
        return {
            "ply_path": ply_path,
            "input_frame_count": image_count,
            "selected_frame_count": selected_count,
            "edgs_num_ref_views_requested": num_ref_views,
            "edgs_selected_ref_views": selected_count,
            "edgs_matches_per_ref": num_corrs_per_view,
            "edgs_nns_per_ref": resolved_nns_per_ref,
            "edgs_preview_steps": num_steps,
            "edgs_scaling_factor": float(scaling_factor),
            "pycolmap_used": False,
            "edgs_rasterizer_module": getattr(rasterizer_module, "__file__", None),
            "edgs_rasterizer_fields": list(getattr(rasterizer_module.GaussianRasterizationSettings, "_fields", ())),
            "edgs_iterations": trainer.gs_step,
            "cuda_memory_peak_mb": peak_mb,
        }


def _prepared_scene_image_count(scene_dir: Path) -> int:
    sparse_dir = scene_dir / "sparse" / "0"
    required = [scene_dir / "images", sparse_dir / "cameras.bin", sparse_dir / "images.bin", sparse_dir / "points3D.bin", sparse_dir / "points3D.ply"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise PreviewFailure("EDGS_SCENE_NOT_PREPARED", f"LiteVGGT EDGS scene is missing: {', '.join(missing)}")
    return len(image_files(scene_dir / "images"))


def ensure_edgs_imports() -> None:
    try:
        __import__("diff_gaussian_rasterization")
        __import__("simple_knn")
        __import__("romatch")
    except Exception as exc:  # pragma: no cover - depends on CUDA worker image
        raise PreviewFailure("EDGS_RUNTIME_UNAVAILABLE", f"EDGS runtime dependency missing: {exc}") from exc
    os.environ.setdefault("WANDB_MODE", "disabled")
