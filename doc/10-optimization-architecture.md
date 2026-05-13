# Optimization And Reconstruction Architecture

## Fine Pipeline Split

`official_fastgs_big` is the image fine pipeline. It defaults to pycolmap/COLMAP initialization, writes a COLMAP-compatible `sparse/0` scene, runs vendored official FastGS-Big training, validates the final PLY, and converts it to Spark SPZ. `fine_sfm_backend=colmap` is accepted as an alias for the pycolmap implementation.

`video_artdeco_speed3r` is the video fine pipeline. It owns video frame extraction, ARTDECO `selfCaptured` calibration, ARTDECO VSLAM state/frontend/backend, ARTDECO Reconstruct h3dgsv3 SceneModel training, `point_clouds/gs.ply` validation, and Spark SPZ conversion.

The video path must not call image fine modules such as `train_mobile_3dgs`, `build_scene`, MobileGS, LM-RS, or DeblurMLP.

## Speed3R-Pi3 Boundary

Speed3R-Pi3 replaces ARTDECO's Pi3 inference object for accurate loop closure. The adapter loads from `model-cache/speed3r_pi3/` and returns the ARTDECO backend's expected pose/point/conf tensors. It does not replace ARTDECO training, mapping, SceneModel, optimizer, COLMAP/PLY output, or task orchestration.

## Runtime Constraints

The worker keeps one CUDA stack: the PyTorch/CUDA/cuDNN baseline already used by image fine. ARTDECO `mast3r_slam_backends` is compiled into that environment. The existing `diff_gaussian_rasterization` remains the only rasterizer; missing ARTDECO Adam symbols are installed as compatibility symbols instead of installing a second package with the same import name.

Model files are cached under `model-cache` and use the same downloader semantics as preview and image fine: skip existing files, write `.part`, hold `.lock`, and resume when possible.

## COLMAP Image Fine Boundary

Image fine uses pycolmap for feature extraction, matching, and mapping. Preview LiteVGGT remains separate and keeps speed-oriented defaults with fewer frames and points.

The image fine order is:

1. pycolmap extracts SIFT features, matches images, and writes the COLMAP-compatible `sparse/0` scene.
2. FastGS-Big reads that scene and trains with its existing densification/pruning behavior.
3. FastGS training progress is surfaced every 200 iterations.
4. The worker validates `final.ply` and converts it to Spark SPZ.

Default COLMAP options remain configurable through `fine_sift_max_num_features`, `fine_colmap_max_image_size`, `fine_colmap_threads`, `fine_colmap_matcher`, and `fine_min_registered_ratio`.
