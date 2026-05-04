from __future__ import annotations

# 本文件为 LingBot-Map 视频/长序列预览入口，按 Robbyant/lingbot-map demo.py
# 的关键流程改写：抽帧/预处理 -> GCTStream 推理 -> world_points/conf 后处理 -> PLY。
# 上游仓库: https://github.com/Robbyant/lingbot-map
# 固定提交: f720b421c6c50af3adc63272033226aa4811ef42
# 许可证: Apache-2.0

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import cv2
import numpy as np

from app.preview.io.ply import write_point_cloud_ply
from app.preview.types import PreviewFailure
from app.preview.utils import VENDOR_ROOT, image_files, prepend_sys_path


Progress = Callable[[str, int, str], None]


def run_lingbot_pointcloud(
    *,
    input_dir: Path,
    input_video: Path | None,
    checkpoint_path: Path,
    output_ply: Path,
    fps: int,
    max_frames: int,
    confidence_quantile: float,
    max_points: int,
    progress: Progress,
) -> dict[str, int | float | str]:
    import torch

    if not torch.cuda.is_available():
        raise PreviewFailure("GPU_RESOURCE_UNAVAILABLE", "LingBot-Map preview requires CUDA")

    with prepend_sys_path(VENDOR_ROOT / "lingbot"):
        from lingbot_map.models.gct_stream import GCTStream
        from lingbot_map.models.gct_stream_window import GCTStream as GCTStreamWindow
        from lingbot_map.utils.load_fn import load_and_preprocess_images

        frame_dir = input_dir
        if input_video:
            progress("lingbot_extract_frames", 18, f"extracting video frames at {fps} fps")
            frame_dir = Path(tempfile.mkdtemp(prefix="lingbot_frames_", dir=str(output_ply.parent)))
            extracted = extract_frames(input_video, frame_dir, fps=fps, max_frames=max_frames)
            if extracted == 0:
                raise PreviewFailure("VIDEO_FRAME_EXTRACTION_FAILED", f"no frames extracted from {input_video}")

        files = image_files(frame_dir)[:max_frames]
        if len(files) < 2:
            raise PreviewFailure("LINGBOT_NOT_ENOUGH_FRAMES", "LingBot-Map preview requires at least 2 frames")

        progress("lingbot_preprocess", 28, f"loading {len(files)} frames")
        images = load_and_preprocess_images([str(path) for path in files], mode="crop", image_size=518, patch_size=14)
        device = torch.device("cuda")
        images = images.to(device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        use_windowed = len(files) > 500
        model_cls = GCTStreamWindow if use_windowed else GCTStream
        args = SimpleNamespace(
            image_size=518,
            patch_size=14,
            enable_3d_rope=True,
            max_frame_num=1024,
            kv_cache_sliding_window=64,
            num_scale_frames=min(8, max(1, len(files) - 1)),
            use_sdpa=True,
            camera_num_iterations=4,
        )
        progress("lingbot_loading_model", 36, f"loading LingBot checkpoint: {checkpoint_path.name}")
        model = model_cls(
            img_size=args.image_size,
            patch_size=args.patch_size,
            enable_point=True,
            enable_3d_rope=args.enable_3d_rope,
            max_frame_num=args.max_frame_num,
            kv_cache_sliding_window=args.kv_cache_sliding_window,
            kv_cache_scale_frames=args.num_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=args.use_sdpa,
            camera_num_iterations=args.camera_num_iterations,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device).eval()

        progress("lingbot_inference", 48, f"running {'windowed' if use_windowed else 'streaming'} LingBot inference")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            if use_windowed:
                predictions = model.inference_windowed(
                    images,
                    window_size=64,
                    overlap_size=16,
                    overlap_keyframes=None,
                    num_scale_frames=args.num_scale_frames,
                    keyframe_interval=1,
                    output_device=torch.device("cpu"),
                )
            else:
                predictions = model.inference_streaming(
                    images,
                    num_scale_frames=args.num_scale_frames,
                    keyframe_interval=1,
                    output_device=torch.device("cpu"),
                )

        progress("lingbot_export_ply", 72, "filtering world points and writing PLY")
        world_points = squeeze_batch(predictions["world_points"]).detach().cpu().numpy()
        conf = squeeze_batch(predictions.get("world_points_conf")).detach().cpu().numpy()
        image_cpu = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        colors = np.clip(image_cpu * 255.0, 0, 255).astype(np.uint8)

        threshold = np.quantile(conf.reshape(-1), confidence_quantile)
        mask = conf >= threshold
        point_count = write_point_cloud_ply(
            world_points[mask],
            colors[mask],
            output_ply,
            confidence=conf[mask],
            max_points=max_points,
        )
        peak_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0
        return {
            "input_frame_count": len(files),
            "point_count": point_count,
            "confidence_quantile": confidence_quantile,
            "cuda_memory_peak_mb": peak_mb,
            "lingbot_mode": "windowed" if use_windowed else "streaming",
        }


def extract_frames(video_path: Path, frame_dir: Path, *, fps: int, max_frames: int) -> int:
    frame_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PreviewFailure("VIDEO_OPEN_FAILED", f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, round(src_fps / max(1, fps)))
    saved = 0
    index = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if index % interval == 0:
            cv2.imwrite(str(frame_dir / f"{saved:06d}.jpg"), frame)
            saved += 1
        index += 1
    cap.release()
    return saved


def squeeze_batch(value):
    if value is None:
        raise PreviewFailure("LINGBOT_OUTPUT_MISSING", "LingBot output is missing world point confidence")
    if getattr(value, "ndim", 0) >= 5 and value.shape[0] == 1:
        return value[0]
    return value
