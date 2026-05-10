# 部署与本地运行

## 本机软件

- Docker Desktop 或 Docker Engine + Docker Compose v2。
- NVIDIA 显卡驱动；GPU worker 需要 NVIDIA Container Toolkit / Docker Desktop GPU 支持。
- Node.js 仅用于本地前端开发；worker 镜像内部会安装 Node 22 用于 Spark SPZ 转码。
- Python 3.11/3.12 仅用于本地 FastAPI 调试；CUDA 编译在 `worker-base` 镜像内完成。

## 模型缓存

预览算法源码已经内置在 `backend/app/preview/vendor`，运行时不需要下载 GitHub 仓库。
权重不提交 Git，也不烘进镜像，部署时挂载到 `model-cache`：

```text
model-cache/litevggt/te_dict.pt
model-cache/amb3r/amb3r.pt
model-cache/roma/roma_outdoor.pth
model-cache/roma/roma_indoor.pth
model-cache/roma/dinov2_vitl14_pretrain.pth
```

`amb3r/amb3r.pt` is part of the task-specific model auto-download path for fine
AMB3R-SfM when `MODEL_AUTO_DOWNLOAD=true`. The worker stores it under
`model-cache/amb3r/` and still supports pre-seeding the file manually.

RoMA/DINOv2 weights are part of the default image fine EDGS/RoMA initialization path. The worker stores them under `model-cache/roma/` and uses them when `fine_edgs_enabled` is not `false`.

Docker build and task-specific model downloads prefer `https://hf-mirror.com` through `HF_MIRROR_BASE_URL` / `HF_ENDPOINT`. The image fine runtime installs `romatch` and `scikit-learn`, but does not clone or copy the full CompVis/EDGS repository.

## Worker 镜像

`worker-base` 包含 CUDA devel、Python 3.10、PyTorch CUDA、Node 22、视觉依赖和编译工具。
`worker-preview` 运行 LiteVGGT 直接 Spark-SPZ 预览，并安装 Spark 转码 CLI。

构建和启动：

```powershell
docker compose up --build
```

## 本地开发

后端：

```powershell
$env:PYTHONPATH="Q:\3dgsbi\backend"
cd Q:\3dgsbi\backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

前端：

```powershell
cd Q:\3dgsbi\frontend
$env:NEXT_PUBLIC_API_BASE_URL="http://127.0.0.1:8000"
npm run dev
```

默认账号：

```text
admin / admin123
```

部署时必须替换 `SECRET_KEY` 和管理员密码。
## 2026-05-10 Video Fine Runtime

The unified worker image builds the video fine runtime in the same PyTorch/CUDA/cuDNN environment as image fine. It clones ARTDECO at `bb654395826e50ac9e4671682d901377115a24ce`, clones Speed3R at `5460f7309c87e5daac36385ff6611627de7d7267`, compiles ARTDECO `mast3r_slam_backends`, and keeps runtime source under:

```text
/opt/artdeco-runtime
/opt/speed3r-runtime
```

Video fine model cache paths:

```text
model-cache/speed3r_pi3/config.json
model-cache/speed3r_pi3/model.safetensors
model-cache/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
model-cache/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
model-cache/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl
```

Do not install a second torch/CUDA stack for ARTDECO or Speed3R. The worker intentionally excludes Open3D, xFormers, Gradio, pyrealsense2, GeoCalib, and full Depth-Anything dependencies.
