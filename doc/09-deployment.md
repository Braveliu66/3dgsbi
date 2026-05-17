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
```

## 算法依赖边界

所有算法复用同一个 worker Python/CUDA 环境，不再创建第二套 conda 或重复安装 torch：

| 算法 | 主要依赖 | 安装位置 |
| --- | --- | --- |
| LiteVGGT | torch/torchvision、Transformer Engine、einops、OpenCV、Pillow、huggingface_hub | worker 镜像 |
| LingBot Video Point Cloud Fast | torch、LingBot-Map pinned package、einops、safetensors、flashinfer、ffmpeg | worker 镜像 |
| Spark SPZ | Node 22、`@sparkjsdev/spark` CLI | worker 镜像 |
| DashDeblurGroupGS Fine | COLMAP CLI、pycolmap、torch/torchvision、lpips、pytorch-msssim、scikit-image、DashDeblurGroupGS CUDA extensions | 共享 worker 镜像；Dockerfile 从本仓库内置 trainer 构建扩展 |

DashDeblurGroupGS 的 `diff_gaussian_rasterization` 和 `simple_knn` 扩展从本仓库内置训练器的
`worker/trainer/dash_deblur_group_gs/submodules/` 安装进当前 worker Python，避免第二套 torch/CUDA。
首次 clone 后先初始化 submodules：

```powershell
git submodule update --init --recursive
docker compose build worker-preview
```

构建缓存默认启用：

- `docker-compose.yml` 会从 `3dgsbi-worker:local` 和 `.docker-build-cache/worker` 读取缓存，并把新缓存写回 `.docker-build-cache/worker`。
- `worker/Dockerfile` 对 apt、pip、npm、COLMAP git/build、CUDA extension wheel 和 `ccache` 都使用 BuildKit cache mount。`diff_gaussian_rasterization` 和 `simple_knn` 第一次成功编译后会写入 `/root/.cache/three-dgs-wheels`；后续同一 Docker builder、相同 Torch/CUDA/架构/源码会显示 `extension wheel cache hit` 并直接安装 wheel，不再运行 nvcc。构建完成后不要清理 Docker builder cache、`.docker-build-cache/` 或 `3dgsbi-worker:local` 镜像，否则扩展可能重新编译。
- 第一次完整构建后，日常启动优先使用 `docker compose up -d`；只有 Dockerfile、requirements、训练仓库 commit 或基础镜像变更时才需要重新 `docker compose build worker-preview`。
- 不要删除 `.docker-build-cache/`、Docker builder cache 或本地 `3dgsbi-worker:local`，否则 COLMAP 和 CUDA 扩展会重新编译。

当前默认训练器是本仓库的 `worker/trainer/dash_deblur_group_gs`。如果需要临时测试外部合并训练器，
设置 `DASH_DEBLUR_GROUP_REPO` 或任务参数 `fine_trainer_repo` 指向兼容 `train.py --config` 的目录。

管理员运行时预检会逐项检查算法依赖、命令和 CUDA 状态；缺少依赖时任务必须失败并说明原因，不能生成占位产物。


## Worker 镜像

`worker-base` 包含 CUDA devel、Python、PyTorch CUDA、Node 22、COLMAP、ffmpeg、视觉依赖和编译工具。
`worker` 同时运行 preview 和 fine worker；镜像会把内置训练器复制到 `/opt/dash_deblur_group_gs`，
运行时 `DASH_DEBLUR_GROUP_REPO` 指向该路径。

COLMAP 不再依赖 Ubuntu/apt 自带旧包。`worker/Dockerfile` 默认从 `COLMAP_REPO_URL`
构建当前 upstream 版本，并在镜像构建期强制检查以下命令存在：

```text
feature_extractor
mapper
global_mapper
hierarchical_mapper
image_undistorter
model_analyzer
model_clusterer
model_splitter
```

`COLMAP_REPO_COMMIT` 默认是 `4.0.4`，部署时可固定到验证过的 upstream commit。缺少
`global_mapper` 或分块命令会直接让 worker 镜像构建失败，而不是等到任务运行时才失败。

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
