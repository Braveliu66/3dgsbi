# DashDeblurGroupGS Fusion Status

This file replaces the earlier planning draft. It documents the current implementation in this repository.

## Current Pipeline

Default fine reconstruction pipeline:

```text
fine_pipeline = dash_deblur_group_gs
```

Execution order:

```text
prepare_fine_images()
  -> blur analysis and low-quality filtering
  -> COLMAP CLI scene: images/ + sparse/0
  -> build_training_config()
  -> worker/trainer/dash_deblur_group_gs/train.py --config
  -> final.ply
  -> final_web.spz
  -> metrics.json
```

`colmap_sparse` is only a legacy alias.

## Fusion Boundary

The trainer is integrated under:

```text
worker/trainer/dash_deblur_group_gs/
```

Only used algorithm code is kept:

- Deblurring-3DGS backbone: GaussianModel, GTnet, motion blur, defocus blur, `add_points()`, sharp canonical render.
- DashGaussian scheduler: frequency resolution schedule, render scale/size, densify rate, momentum budget.
- Group Training: temporary active/cached Gaussian split, non-destructive cache, merge-back.

Not included:

- Speedy-Splat renderer or pruning.
- FastGS renderer/pruning.
- SparseAdam.
- Dash antialiasing branch.
- 3dgs-accel replacement renderer.
- Non-native `birth_iter` / `protect_new_points_iters` pruning state.

The previous `birth_iter` idea was removed because Deblurring-3DGS densification and `add_points()` rebuild point tensors and optimizer state through several paths. A parallel point-age tensor is not part of the original algorithm contract and can desynchronize from prune masks.

## Deblur Mode

Public option:

```text
fine_deblur_mode = mix | motion | defocus | sharp
```

Default is `mix`.

`mix` means automatic branch selection before training, not mixed branch training:

- motion vote wins -> `deblur = 1`
- defocus vote wins -> `deblur = 2`
- uncertain -> conservative `motion`
- explicit `motion`, `defocus`, or `sharp` skips auto selection

The backend writes metrics:

```json
{
  "fine_deblur_mode_requested": "mix",
  "fine_deblur_mode_effective": "motion",
  "deblur_auto_confidence": "high"
}
```

## Config Presets

Only four concrete presets exist:

- `indoor_motion`
- `indoor_defocus`
- `outdoor_motion`
- `outdoor_defocus`

The UI/API exposes only `scene_type` and `fine_deblur_mode`. The generated config must not contain `protect_new_points_iters`.

## Runtime

The worker image copies the embedded trainer to `/opt/dash_deblur_group_gs` so CUDA extensions can be built from:

```text
worker/trainer/dash_deblur_group_gs/submodules/diff-gaussian-rasterization
worker/trainer/dash_deblur_group_gs/submodules/simple-knn
```

Local Docker Compose bind-mounts the working tree over the same path:

```yaml
./worker/trainer/dash_deblur_group_gs:/opt/dash_deblur_group_gs
```

It also bind-mounts:

```yaml
./backend/app:/app/app
./frontend:/app
```

So normal Python/TypeScript/trainer edits do not require image rebuild. Recreate services:

```powershell
docker compose up -d --force-recreate backend worker-preview worker-fine frontend
```

Rebuild only for Dockerfile, requirements, system packages, CUDA extensions, base image, or submodule changes.

## COLMAP

Default SfM:

```text
fine_sfm_backend = colmap_cli
```

Feature extraction, matching, and mapper bundle adjustment use GPU when `prefer_gpu=true`. Metrics record:

```json
{
  "colmap_use_gpu": true,
  "colmap_ba_use_gpu": true,
  "colmap_gpu_index": "0"
}
```

The worker image builds upstream COLMAP and checks these commands during image build:

```text
feature_extractor
mapper
global_mapper
hierarchical_mapper
image_undistorter
model_analyzer
model_clusterer
model_splitter
```

## Test Plan

Focused checks:

```powershell
python -m unittest backend.tests.test_colmap_cli_policy backend.tests.test_dash_deblur_group_runtime backend.tests.test_fine_runtime
python -m compileall -q backend/app/fine worker/trainer/dash_deblur_group_gs
```

Runtime check:

```powershell
docker compose exec worker-fine python -c "from pathlib import Path; s=Path('/opt/dash_deblur_group_gs/scene/gaussian_model.py').read_text(encoding='utf-8'); print('birth_iter' in s)"
```

Expected:

```text
False
```
