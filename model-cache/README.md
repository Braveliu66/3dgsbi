# Model Cache

Third-party model weights live here. Large weights are ignored by Git and are not baked into Docker images.

Workers download only the weights required by the selected pipeline. Downloads use the shared model downloader: existing non-empty files are skipped, interrupted downloads keep a sibling `.part`, active downloads hold a `.lock`, and HTTP Range resume is used when the server supports it.

Expected paths:

- `litevggt/te_dict.pt`
- `lingbot/lingbot-map-long.pt`
- `torch/`
- `huggingface/`

Image fine reconstruction uses vendored official FastGS-Big and pycolmap/COLMAP initialization, so it has no task-specific model weight download.

Each completed download writes `<filename>.download.json` with URL, size and completion metadata. If a `.part` file is stuck or corrupted, stop the worker, delete only that `.part` and matching `.lock`, then start the worker again.
