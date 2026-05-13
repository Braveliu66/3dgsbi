from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Open LingBot-Map official PointCloudViewer from saved predictions.")
    parser.add_argument("predictions_npz", type=Path, help="Path to official_predictions.npz")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf-threshold", type=float, default=1.5)
    parser.add_argument("--downsample-factor", type=int, default=10)
    parser.add_argument("--point-size", type=float, default=0.001)
    parser.add_argument("--use-point-map", action="store_true", help="Use world_points instead of depth reprojection")
    parser.add_argument("--depth-reprojection", action="store_true", help="Use official depth reprojection path instead of world_points")
    parser.add_argument("--mask-sky", action="store_true")
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--sky-mask-dir", type=str, default=None)
    parser.add_argument("--sky-mask-visualization-dir", type=str, default=None)
    parser.add_argument("--depth-stride", type=int, default=1)
    args = parser.parse_args()

    predictions_npz = args.predictions_npz
    if not predictions_npz.exists() or predictions_npz.stat().st_size <= 0:
        raise SystemExit(f"Missing non-empty predictions NPZ: {predictions_npz}")

    try:
        from lingbot_map.vis import PointCloudViewer
    except ImportError as exc:
        install_command = (
            'python -m pip install "lingbot-map[vis] @ '
            'git+https://github.com/Robbyant/lingbot-map.git@4cd986009b9adeded8a4e740919221940dedeffe"'
        )
        raise SystemExit(
            "Official LingBot viewer is unavailable in this Python environment.\n"
            f"Import error: {exc}\n"
            f"Install the pinned official package with:\n  {install_command}"
        ) from exc

    with np.load(predictions_npz, allow_pickle=False) as data:
        pred_dict = {key: data[key] for key in data.files}

    if args.use_point_map and args.depth_reprojection:
        raise SystemExit("Choose either --use-point-map or --depth-reprojection, not both.")
    use_point_map = (args.use_point_map or not args.depth_reprojection) and "world_points" in pred_dict
    point_source = "world_points" if use_point_map else "depth reprojection"
    if shutil.which("ffmpeg") is None:
        print("Warning: ffmpeg was not found on PATH. The viewer works, but MP4 video export will fail until ffmpeg is installed.")
    viewer = PointCloudViewer(
        pred_dict=pred_dict,
        port=args.port,
        vis_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor,
        point_size=args.point_size,
        use_point_map=use_point_map,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
        sky_mask_dir=args.sky_mask_dir,
        sky_mask_visualization_dir=args.sky_mask_visualization_dir,
        depth_stride=max(1, args.depth_stride),
    )
    print(f"Point source: {point_source}")
    print(f"3D viewer at http://localhost:{args.port}")
    viewer.run()


if __name__ == "__main__":
    main()
