# Model Cache

Third-party model weights live here. Large weights are ignored by Git and are not baked into Docker images.

Workers download only the weights required by the selected pipeline. Downloads use the shared model downloader: existing non-empty files are skipped, interrupted downloads keep a sibling `.part`, active downloads hold a `.lock`, and HTTP Range resume is used when the server supports it.

Expected paths:

- `litevggt/te_dict.pt`
- `amb3r/amb3r.pt`
- `speed3r_pi3/config.json`
- `speed3r_pi3/model.safetensors`
- `mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`
- `mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth`
- `mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl`
- `torch/`
- `huggingface/`

`mobilegs_lmrs` fine reconstruction uses `amb3r/amb3r.pt`. `video_artdeco_speed3r` fine reconstruction uses the Speed3R-Pi3 files plus the MASt3R checkpoint and retrieval codebook. The video weights carry upstream research/non-commercial restrictions; verify ARTDECO, MASt3R and Speed3R-Pi3 terms before commercial use.

Each completed download writes `<filename>.download.json` with URL, size and completion metadata. If a `.part` file is stuck or corrupted, stop the worker, delete only that `.part` and matching `.lock`, then start the worker again.
