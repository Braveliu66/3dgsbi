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
model-cache/lingbot-map/lingbot-map-long.pt
model-cache/roma/roma_indoor.pth
model-cache/roma/dinov2_vitl14_pretrain.pth
```

`amb3r/amb3r.pt` is part of the task-specific model auto-download path for fine
AMB3R-SfM when `MODEL_AUTO_DOWNLOAD=true`. The worker stores it under
`model-cache/amb3r/` and still supports pre-seeding the file manually.

`roma/*` 是 EDGS 的 RoMA correspondence 初始化依赖；缺失时 `litevggt_edgs`
会失败并写出明确错误，不会生成 artifact。

## Worker 镜像

`worker-base` 包含 CUDA devel、Python 3.10、PyTorch CUDA、Node 22、视觉依赖和编译工具。
`worker-preview` 从项目内置 vendor 路径编译 EDGS 的 `diff-gaussian-rasterization`、
`simple-knn`、`fused-ssim` 扩展，并安装 Spark 转码 CLI。

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
