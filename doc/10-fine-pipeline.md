# Fine Reconstruction Pipeline

This document is the authoritative description of the current fine reconstruction path.

## Pipeline

The default fine pipeline is `dash_deblur_group_gs`.

```text
uploaded images / extracted frames
  -> RGB JPEG normalization and quality filtering
  -> blur analysis
  -> COLMAP CLI scene construction
  -> DashDeblurGroupGS training
  -> final.ply
  -> Spark SPZ final_web.spz
  -> metrics.json and final_viewer_meta.json
```

`colmap_sparse` is only a legacy alias. Preview LiteVGGT is separate from this path.

## Algorithm Fusion

The embedded trainer lives in `worker/trainer/dash_deblur_group_gs`.

- Deblurring-3DGS is the training backbone. It owns GTnet, motion blur, defocus blur, point addition, and sharp canonical Gaussian rendering.
- DashGaussian contributes only the frequency resolution scheduler and Gaussian growth budgeting.
- Group Training contributes only non-destructive active/cached Gaussian splitting and merge-back.
- Speedy-Splat, FastGS pruning, SparseAdam, Dash antialiasing, and renderer replacement are not part of the default path.

The trainer is not a copied upstream repository. Only the used algorithmic pieces are integrated into this repo and shaped around the local fine worker contract.

## Deblur Mode

The public option is:

```text
fine_deblur_mode = mix | motion | defocus | sharp
```

Default is `mix`.

`mix` does not run a mixed deblur branch. The backend uses the blur analysis from preprocessing to choose an effective branch before training:

- `motion` -> trainer config `deblur = 1`
- `defocus` -> trainer config `deblur = 2`
- `sharp` -> trainer config `deblur = 0`

If auto classification is uncertain, the conservative default is `motion`. Metrics record both requested and effective modes:

```json
{
  "fine_deblur_mode_requested": "mix",
  "fine_deblur_mode_effective": "motion",
  "deblur_auto_confidence": "low"
}
```

## Parameter Layers

The UI/API exposes only:

- `scene_type=indoor|outdoor`
- `fine_deblur_mode=mix|motion|defocus|sharp`

The backend resolves those into one of four concrete presets:

- `indoor_motion`
- `indoor_defocus`
- `outdoor_motion`
- `outdoor_defocus`

There is no fifth mixed-training preset.

The removed `protect_new_points_iters` and `birth_iter` mechanism must not be reintroduced. It was a local helper, not part of the Deblurring-3DGS densification model, and it broke tensor-length invariants after `add_points()` and prune. New point survival is controlled by the original densify/prune thresholds and Group cache timing.

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

`fine_sfm_backend=colmap_cli` is the default. `fine_sfm_backend=colmap` is accepted as an alias. `pycolmap` remains available for explicit use.

COLMAP feature extraction and matching use GPU when `prefer_gpu=true`. Mapper bundle adjustment also uses GPU when `prefer_gpu=true`.

Metrics include:

```json
{
  "sfm_backend": "colmap_cli",
  "sfm_registered_images": 45,
  "sfm_sparse_points": 6830,
  "colmap_use_gpu": true,
  "colmap_ba_use_gpu": true,
  "colmap_gpu_index": "0"
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
