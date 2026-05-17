# Current Progress

## Done

- FastAPI, Next.js, PostgreSQL, Redis, local object storage, preview worker, and fine worker are wired into the single-machine Docker Compose stack.
- Image preview uses LiteVGGT plus Spark SPZ conversion.
- Video preview uses LingBot-Map plus Spark SPZ conversion.
- Image fine reconstruction uses the active `dash_deblur_group_gs` pipeline: JPG/PNG or extracted-frame normalization, blur analysis and low-quality frame filtering, existing COLMAP-compatible `images/` + `sparse/0`, DashDeblurGroupGS training, and Spark SPZ conversion.
- The old MobileGS/LM-RS, LiteVGGT-fine, and video-fine code paths have been removed from the backend.

## Current Limits

- Fine reconstruction accepts image projects and single-video projects after frame extraction.
- Fine reconstruction requires COLMAP CLI/pycolmap plus the embedded DashDeblurGroupGS trainer at `/opt/dash_deblur_group_gs`; `DASH_DEBLUR_GROUP_REPO` is only for explicit compatible trainer overrides.
- The worker Dockerfile builds upstream COLMAP and requires `global_mapper`, `hierarchical_mapper`, `model_clusterer`, and `model_splitter` at image build time.
- `fine_sfm_backend=colmap_cli` is the implementation default; `fine_sfm_backend=colmap` is accepted as an alias and `pycolmap` remains available.

## Next

- Validate the current `dash_deblur_group_gs` fine pipeline on a real GPU worker after building the worker image with the merged training checkout.
