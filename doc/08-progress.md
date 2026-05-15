# Current Progress

## Done

- FastAPI, Next.js, PostgreSQL, Redis, local object storage, preview worker, and fine worker are wired into the single-machine Docker Compose stack.
- Image preview uses LiteVGGT plus Spark SPZ conversion.
- Video preview uses LingBot-Map plus Spark SPZ conversion.
- Image fine reconstruction uses the active `official_fastgs_big` pipeline: JPG/PNG normalization, blur analysis and low-quality frame filtering, pycolmap/COLMAP-compatible `sparse/0`, vendored official FastGS-Big training with optional GTnet deblur, and Spark SPZ conversion.
- The old MobileGS/LM-RS, LiteVGGT-fine, and video-fine code paths have been removed from the backend.

## Current Limits

- Fine reconstruction accepts image projects only. Video fine reconstruction is disabled; use video preview instead.
- Fine reconstruction requires a CUDA worker with `diff_gaussian_rasterization_fastgs`, `simple_knn`, `fused_ssim`, pycolmap, and the Spark SPZ converter available.
- `fine_sfm_backend=pycolmap` is the implementation default; `fine_sfm_backend=colmap` is accepted as an alias.

## Next

- Validate the current `official_fastgs_big` image fine pipeline on a real GPU worker after the cleanup.
