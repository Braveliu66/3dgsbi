# 部署与本地运行

## 1. 本机要求

- Docker Desktop 或 Docker Engine + Docker Compose v2。
- NVIDIA 驱动和容器 GPU 支持。
- 首次构建需要网络访问 PyPI、npm、GitHub/COLMAP 源或配置好的镜像源。
- 本地非容器调试后端需要 Python 3.11/3.12。
- 本地非容器调试前端需要 Node.js。

## 2. Compose 服务

`docker-compose.yml` 包含：

| 服务 | 端口 | 说明 |
| --- | --- | --- |
| `postgres` | 5432 | PostgreSQL 16 |
| `redis` | 6379 | Redis 7 |
| `backend` | 8000 | FastAPI，热挂载 `backend/app` |
| `worker-preview` | 无 | 监听 `preview_tasks` |
| `worker-fine` | 无 | 监听 `fine_tasks` |
| `frontend` | 3001 -> 3000 | Next dev server |

启动：

```powershell
docker compose up -d
```

健康检查：

```powershell
curl http://127.0.0.1:8000/health
```

前端访问：

```text
http://127.0.0.1:3001
```

默认管理员：

```text
admin / admin123
```

部署时必须替换 `SECRET_KEY` 和管理员密码。

## 3. 缓存目录

| 路径 | 用途 |
| --- | --- |
| `model-cache/` | LiteVGGT 权重、torch/huggingface/xdg/inductor 缓存 |
| `repo-cache/` | 外部兼容仓库覆盖和 COLMAP 构建缓存 |
| `.docker-build-cache/` | BuildKit local cache |
| `data/storage/` | 本地对象存储 |
| `data/work/` | 任务临时目录 |

当前必需权重：

```text
model-cache/litevggt/te_dict.pt
```

`MODEL_AUTO_DOWNLOAD=true` 时 worker 会尝试下载缺失权重，默认优先 `https://hf-mirror.com`。

## 4. Worker 镜像

worker 默认基础镜像：

```text
docker.m.daocloud.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
```

关键组件：

- PyTorch 2.8 / CUDA 12.8。
- Node 22，用于 Spark SPZ 转码。
- upstream COLMAP，默认 `COLMAP_REPO_COMMIT=4.0.4`。
- ffmpeg、OpenCV、pycolmap、gsplat、lpips、pytorch-msssim。
- `diff_gaussian_rasterization` 和 `simple_knn` CUDA 扩展。
- embedded trainer：`worker/trainer/dash_deblur_group_gs`。

构建期强制检查 COLMAP 命令：

```text
feature_extractor
exhaustive_matcher
mapper
image_undistorter
point_triangulator
global_mapper
hierarchical_mapper
model_analyzer
model_clusterer
model_splitter
```

首次 clone 后初始化 submodule：

```powershell
git submodule update --init --recursive
docker compose build worker-preview worker-fine
```

## 5. 日常开发

Compose 已挂载：

- `./backend/app:/app/app`
- `./backend/blur_detection_no_percent_adaptive_canny.py:/app/blur_detection_no_percent_adaptive_canny.py`
- `./worker/trainer/dash_deblur_group_gs:/opt/dash_deblur_group_gs`
- `./frontend:/app`
- `./model-cache:/model-cache`
- `./data/storage:/app/data/storage`
- `./data/work:/app/data/work`

普通 Python、TypeScript 或 trainer 源码修改：

```powershell
docker compose up -d --force-recreate backend worker-preview worker-fine frontend
```

需要 rebuild 的情况：

- Dockerfile 变化。
- requirements/constraints 变化。
- CUDA 扩展源码或 submodule 变化。
- 系统包或基础镜像变化。
- COLMAP commit/构建参数变化。

不要无意删除：

- `.docker-build-cache/`
- Docker builder cache。
- `3dgsbi-worker:local` 镜像。

否则 COLMAP 和 CUDA 扩展可能重新编译。

## 6. 本地非容器调试

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
$env:NEXT_PUBLIC_UPLOAD_API_BASE_URL="http://127.0.0.1:8000"
npm run dev
```

注意：非容器后端通常不具备完整 CUDA worker 环境，只适合 API 和前端调试。

## 7. 运行时预检

管理员接口：

```text
GET /api/admin/runtime/preflight
```

用于检查：

- GPU / torch CUDA。
- preview worker 和 fine worker 心跳。
- LiteVGGT 权重。
- Spark SPZ converter。
- DashDeblurGroupGS trainer。
- ffmpeg、node、git、COLMAP、pycolmap。

任务失败时先看：

1. 项目详情任务日志。
2. `/api/tasks/{task_id}/log`。
3. 管理页 runtime preflight。
4. worker 容器日志。
