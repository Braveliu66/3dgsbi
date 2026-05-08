# Model Cache

Third-party model weights live here. Large weights are intentionally ignored by Git and are not baked into Docker images.

`worker-preview` checks this mounted directory when a task starts and downloads only the weights required by that task's pipeline. Interrupted downloads keep a sibling `.part` file and resume on the next task run or `docker compose up`.

Expected paths:

- `litevggt/te_dict.pt`
- `lingbot-map/lingbot-map-long.pt`
- `roma/roma_indoor.pth`
- `roma/dinov2_vitl14_pretrain.pth`
- `torch/hub/checkpoints/vgg16-397923af.pth`
- `huggingface/`

Each completed download writes `<filename>.download.json` with URL, size and completion metadata. If a `.part` file is stuck or corrupted, stop `worker-preview`, delete only that `.part` and matching `.lock`, then start the worker again.

`litevggt/te_dict.pt` is shared by LiteVGGT preview and the default `mobilegs_lmrs` fine reconstruction pipeline. `roma/*` is only used by the EDGS preview path. `torch/*` and `huggingface/*` keep framework-managed runtime caches such as LPIPS VGG16 and Hugging Face files on the host. Keep these files in the mounted cache so rebuilt images and restarted containers do not download large models again.
