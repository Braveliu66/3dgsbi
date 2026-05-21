# Current Progress

Updated: 2026-05-21

## Done

- FastAPI, Next.js, PostgreSQL, Redis, local object storage, preview worker, and fine worker are wired into Docker Compose.
- Authentication, project CRUD subset, chunked upload, media thumbnails, task queues, task logs, artifacts, sharing, feedback, admin views, and pipeline parameter defaults are implemented.
- Image preview uses LiteVGGT plus Spark SPZ conversion.
- Single-video preview extracts frames with ffmpeg, applies LiteVGGT video speed defaults, then uses the same LiteVGGT plus Spark SPZ path.
- Image fine reconstruction uses `dash_deblur_group_gs`: RGB JPEG normalization, blur analysis, COLMAP scene construction, optional EAP, DashDeblurGroupGS training, final PLY filtering, Spark SPZ conversion, metrics, and viewer metadata.
- Single-video fine reconstruction is enabled by extracting and filtering frames before the same fine pipeline.
- Default fine SfM backend is `colmap_global`; `gcolmap` aliases to it. `colmap_cli`, `colmap`, and `pycolmap` remain explicit alternatives.
- The old MobileGS/LM-RS, LiteVGGT-fine, LingBot preview registry entry, EDGS/RoMA dense initialization, and non-native point protection fields are outside the current mainline.

## Current Limits

- Real-time camera capture and incremental preview are not implemented.
- Mesh export tasks are not implemented.
- RAD LOD generation is not implemented by default.
- Project PATCH/update is not implemented.
- Fine reconstruction requires a working worker image with COLMAP, pycolmap, CUDA extensions, Node/Spark, ffmpeg, and the embedded trainer mounted at `/opt/dash_deblur_group_gs`.
- Preview requires `model-cache/litevggt/te_dict.pt` unless auto-download can fetch it.
- Running external training processes are not yet forcibly interrupted on cancel.

## Next

- Continue GPU smoke testing for image/video preview and image/video fine.
- Tighten runtime diagnostics for COLMAP global mapper, EAP, CUDA extension, and SPZ failures.
- Add an end-to-end regression script for upload -> preview -> fine -> viewer-config.
- Decide whether the next feature should be Mesh export or thesis/experiment reporting support.
