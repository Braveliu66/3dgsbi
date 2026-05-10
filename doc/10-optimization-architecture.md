# Optimization And Reconstruction Architecture

## Fine Pipeline Split

`mobilegs_lmrs` is the image fine pipeline. It owns AMB3R-SfM, minimal EDGS/RoMA dense-correspondence Gaussian initialization, DeblurMLP-MobileGS training with warmup, optional LM-RS refinement, final PLY validation, and Spark SPZ conversion.

`video_artdeco_speed3r` is the video fine pipeline. It owns video frame extraction, ARTDECO `selfCaptured` calibration, ARTDECO VSLAM state/frontend/backend, ARTDECO Reconstruct h3dgsv3 SceneModel training, `point_clouds/gs.ply` validation, and Spark SPZ conversion.

The video path must not call image fine modules such as `train_mobile_3dgs`, `build_scene`, AMB3R, MobileGS, LM-RS, or DeblurMLP.

## Speed3R-Pi3 Boundary

Speed3R-Pi3 replaces ARTDECO's Pi3 inference object for accurate loop closure. The adapter loads from `model-cache/speed3r_pi3/` and returns the ARTDECO backend's expected pose/point/conf tensors. It does not replace ARTDECO training, mapping, SceneModel, optimizer, COLMAP/PLY output, or task orchestration.

## Runtime Constraints

The worker keeps one CUDA stack: the PyTorch/CUDA/cuDNN baseline already used by image fine. ARTDECO `mast3r_slam_backends` is compiled into that environment. The existing `diff_gaussian_rasterization` remains the only rasterizer; missing ARTDECO Adam symbols are installed as compatibility symbols instead of installing a second package with the same import name.

Model files are cached under `model-cache` and use the same downloader semantics as preview and image fine: skip existing files, write `.part`, hold `.lock`, and resume when possible.

## EDGS/RoMA Image Fine Boundary

The project does not vendor or clone the full EDGS repository. Image fine only keeps the small runtime needed for dense-correspondence initialization under `backend/app/fine/edgs_runtime/`, wrapped by `backend/app/fine/edgs_init.py`.

The image fine order is:

1. AMB3R-SfM writes the COLMAP-compatible `sparse/0` scene.
2. `EDGSDenseInit` runs after `Scene(...)` and before `gaussians.training_setup(opt)`.
3. EDGS/RoMA creates the initial Gaussian set from dense correspondences.
4. MobileGS trains with densification disabled by EDGS and keeps final prune behavior.
5. DeblurMLP GTnet stays inactive during the default 3000-iteration warmup, then activates and scales xyz learning rate by `0.1`.

Default EDGS options are `matches_per_ref=15000`, `nns_per_ref=3`, `num_refs=len(train cameras)`, and `roma_model=outdoor`. EDGS is enabled by default and can be disabled with `fine_edgs_enabled=false`, which falls back to the current AMB3R sparse initialization path.
