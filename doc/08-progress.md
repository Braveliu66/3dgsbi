# Current Progress

## Done

- FastAPI, Next.js, PostgreSQL, Redis, local object storage, preview worker, and fine worker are wired into the single-machine Docker Compose stack.
- Image preview uses LiteVGGT plus Spark SPZ conversion.
- Video preview uses LingBot-Map plus Spark SPZ conversion.
- Image fine reconstruction uses the active `dash_deblur_group_gs` pipeline: JPG/PNG or extracted-frame normalization, blur analysis, optional low-quality frame filtering, PyCOLMAP-compatible `images/` + `sparse/0`, mixed/motion/defocus DashDeblurGroupGS training, and Spark SPZ conversion.
- The old MobileGS/LM-RS, LiteVGGT-fine, and video-fine code paths have been removed from the backend.

## Current Limits

- Fine reconstruction accepts image projects and single-video projects after frame extraction.
- Fine reconstruction requires pycolmap plus the embedded DashDeblurGroupGS trainer mounted at `/opt/dash_deblur_group_gs`; COLMAP CLI remains an explicit optional backend and `DASH_DEBLUR_GROUP_REPO` is only for compatible trainer overrides.
- The worker Dockerfile builds upstream COLMAP and requires `global_mapper`, `hierarchical_mapper`, `model_clusterer`, and `model_splitter` at image build time.
- `fine_sfm_backend=pycolmap` is the implementation default; `fine_sfm_backend=colmap` is accepted as a PyCOLMAP alias and `colmap_cli` remains available for explicit CLI runs.
- Docker Compose bind-mounts backend app code, frontend code, and the embedded fine trainer, so ordinary Python/TypeScript changes require `docker compose up -d --force-recreate ...`, not an image rebuild.
- The fused trainer no longer uses `birth_iter` or `protect_new_points_iters`; generated configs must not include either field.

## Next

- Continue GPU smoke testing of the current `dash_deblur_group_gs` fine pipeline and tighten runtime diagnostics from real task failures.
