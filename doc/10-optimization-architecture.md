# Optimization And Reconstruction Architecture

## Fine Pipeline Split

`mobilegs_lmrs` is the image fine pipeline. It owns AMB3R-SfM, DeblurMLP-MobileGS training, optional LM-RS refinement, final PLY validation, and Spark SPZ conversion.

`video_artdeco_speed3r` is the video fine pipeline. It owns video frame extraction, ARTDECO `selfCaptured` calibration, ARTDECO VSLAM state/frontend/backend, ARTDECO Reconstruct h3dgsv3 SceneModel training, `point_clouds/gs.ply` validation, and Spark SPZ conversion.

The video path must not call image fine modules such as `train_mobile_3dgs`, `build_scene`, AMB3R, MobileGS, LM-RS, or DeblurMLP.

## Speed3R-Pi3 Boundary

Speed3R-Pi3 replaces ARTDECO's Pi3 inference object for accurate loop closure. The adapter loads from `model-cache/speed3r_pi3/` and returns the ARTDECO backend's expected pose/point/conf tensors. It does not replace ARTDECO training, mapping, SceneModel, optimizer, COLMAP/PLY output, or task orchestration.

## Runtime Constraints

The worker keeps one CUDA stack: the PyTorch/CUDA/cuDNN baseline already used by image fine. ARTDECO `mast3r_slam_backends` is compiled into that environment. The existing `diff_gaussian_rasterization` remains the only rasterizer; missing ARTDECO Adam symbols are installed as compatibility symbols instead of installing a second package with the same import name.

Model files are cached under `model-cache` and use the same downloader semantics as preview and image fine: skip existing files, write `.part`, hold `.lock`, and resume when possible.
