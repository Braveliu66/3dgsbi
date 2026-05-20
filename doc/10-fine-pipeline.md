# Fine Reconstruction Pipeline

This document is the authoritative description of the current fine reconstruction path.

## Pipeline

The default fine pipeline is `dash_deblur_group_gs`.

```text
uploaded images / extracted frames
  -> RGB JPEG normalization and optional quality filtering
  -> blur analysis
  -> PyCOLMAP scene construction (COLMAP CLI optional)
  -> DashDeblurGroupGS training
  -> final.ply
  -> Spark SPZ final_web.spz
  -> metrics.json and final_viewer_meta.json
```

`colmap_sparse` is only a legacy alias. Preview LiteVGGT is separate from this path.

## Algorithm Fusion

The embedded trainer lives in `worker/trainer/dash_deblur_group_gs`.

- Deblurring-3DGS is the training backbone. It owns GTnet, motion blur, defocus blur, point addition, and sharp canonical Gaussian rendering.
- Speedy-Splat, FastGS pruning, SparseAdam, Dash antialiasing, and renderer replacement are not part of the default path.

The default training preset keeps the Deblurring-3DGS motion/defocus backbone with `resolution=-1`, but random `add_points()` is disabled by default. EAP provides the initial point-cloud boost, while the standard gradient-based densify/prune path handles local refinement. For 5000-iteration training, `densify_until_iter` defaults to 3000 so late training does not keep doing expensive topology changes.

The embedded trainer also treats the old random-add default triplet (`pts_iter=2500`, `pts_rate=1.1`, `pts_N_pts=200000`) as disabled. This prevents stale saved presets from reintroducing the large random point burst that can stall training after the point cloud has already grown.

The trainer is not a copied upstream repository. Only the used algorithmic pieces are integrated into this repo and shaped around the local fine worker contract.

## Deblur Mode

The public option is:

```text
fine_deblur_mode = motion | defocus | sharp
```

Default is `motion`. When deblur is enabled, GTnet deblur rendering is applied to every training image, not only frames classified as blurred by preprocessing.

The upstream Deblurring-3DGS trainer selects the physical branch through `use_pos`: motion blur uses position deltas, while defocus blur disables them. The default no longer uses blur analysis to auto-switch branches. Legacy `mix`, `auto`, and `automatic` requests are accepted as aliases for `motion`.

- `motion` -> trainer config `deblur = 1`
- `defocus` -> trainer config `deblur = 1` with position deltas disabled
- `sharp` -> trainer config `deblur = 0`

Metrics record both requested and effective modes:

```json
{
  "fine_deblur_mode_requested": "motion",
  "fine_deblur_mode_effective": "motion",
  "deblur_auto_confidence": "explicit"
}
```

## Parameter Layers

The UI/API exposes only:

- `scene_type=indoor|outdoor`
- `fine_deblur_mode=motion|defocus|sharp`

The backend resolves those into scene-specific presets:

- `indoor_motion`
- `indoor_defocus`
- `outdoor_motion`
- `outdoor_defocus`

The removed `protect_new_points_iters` and `birth_iter` mechanism must not be reintroduced as an ad hoc side tensor. It was a local helper, not part of the Deblurring-3DGS densification model, and it broke tensor-length invariants after `add_points()` and prune.

The planned replacement for random `add_points()` is GDAGS-style density control from `final_mixed_blur_3dgs_codex_plan.md`. It must land in phases: stats-only first, then clone/split/prune decisions after canonical-gradient isolation and buffer synchronization tests pass. Any GDAGS age/protection state must be owned by the GDAGS manager and updated through the same clone/split/prune masks as GaussianModel.

## Runtime Layout

Docker images still include the trainer so CUDA extensions can be built during image creation:

```text
/opt/dash_deblur_group_gs
```

For local development, Docker Compose bind-mounts the working tree trainer over that path:

```yaml
./worker/trainer/dash_deblur_group_gs:/opt/dash_deblur_group_gs
```

The backend, preview worker, fine worker, and frontend are also mounted for source updates:

- `./backend/app:/app/app`
- `./worker/trainer/dash_deblur_group_gs:/opt/dash_deblur_group_gs`
- `./frontend:/app`

After Python or frontend source changes, use:

```powershell
docker compose up -d --force-recreate backend worker-preview worker-fine frontend
```

Do not rebuild for ordinary Python/TypeScript edits. Rebuild only when Dockerfile, requirements, CUDA extensions, system packages, base images, or submodules change.

## COLMAP

`fine_sfm_backend=colmap_global` is the default and uses COLMAP 4.x `global_mapper`. `fine_sfm_backend=gcolmap` is accepted as an alias. `fine_sfm_backend=colmap_cli` keeps the incremental `mapper` path, and `pycolmap` remains available as an explicit backend.

The COLMAP paths use GPU feature extraction and matching when `prefer_gpu=true`. Global mapper is required for `colmap_global`/`gcolmap`; if it is missing, the task fails instead of silently falling back to incremental mapping.

Metrics include:

```json
{
  "sfm_backend": "colmap_global",
  "sfm_registered_images": 45,
  "sfm_sparse_points": 6830
}
```

## Outputs

Fine tasks produce:

- `final.ply`
- `final_web.spz`
- `final_viewer_meta.json`
- `metrics.json`
- task log files

The pipeline must fail instead of creating fake artifacts when COLMAP, CUDA extensions, trainer dependencies, or output validation fail.

## Development Checks

Useful local checks:

```powershell
python -m unittest backend.tests.test_colmap_cli_policy backend.tests.test_dash_deblur_group_runtime backend.tests.test_fine_runtime
python -m compileall -q backend/app/fine worker/trainer/dash_deblur_group_gs
```

Confirm the running fine worker sees the mounted trainer:

```powershell
docker compose exec worker-fine python -c "from pathlib import Path; s=Path('/opt/dash_deblur_group_gs/scene/gaussian_model.py').read_text(encoding='utf-8'); print('birth_iter' in s)"
```

Expected output:

```text
False
```
