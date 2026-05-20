# Deblurring 3DGS 本地训练与可视化系统

本项目在 Deblurring-3D-Gaussian-Splatting 基础上，提供了一套可本地运行的 Web 控制台，用于完成多视角图像上传、COLMAP 重建、3DGS 训练、渲染评估与点云预览。

## 1. 项目定位

适用场景：
- 需要快速验证 Deblurring 3DGS 训练流程
- 希望通过图形化界面管理任务，而非全程命令行
- 需要在训练过程中实时查看日志、阶段进度和 ETA

主要能力：
- 多图上传并创建训练任务
- 支持 `mlp_deblur` 与 `official_3dgs` 两种训练方式
- 自动执行 COLMAP 全流程（特征提取、匹配、建图、去畸变）
- 实时展示任务状态、训练进度、阶段剩余与总剩余时间
- 产物管理：日志、渲染图、PLY 点云、`cameras.json`
- 前端本地点云预览（不依赖 three.js/CDN）

---

## 2. 环境要求

建议环境：
- 操作系统：Windows / Linux（已在 Windows PowerShell 环境中适配）
- Python：3.10（推荐与 `environment.yml` 保持一致）
- CUDA：可选（有 NVIDIA GPU 时可显著提升训练速度）
- COLMAP：必须可执行（加入 PATH，或通过环境变量指定）

关键依赖：
- PyTorch
- COLMAP
- diff-gaussian-rasterization
- simple-knn
- FastAPI / Uvicorn

---

## 3. 目录结构（核心）

```text
.
├─ app.py                  # Web 服务入口（FastAPI）
├─ web/
│  ├─ index.html           # 前端页面
│  ├─ styles.css           # 前端样式
│  └─ app.js               # 前端交互逻辑（任务提交/轮询/PLY 预览）
├─ web_backend/
│  └─ jobs.py              # 后端任务编排与状态管理
├─ scene/
│  ├─ __init__.py          # 场景加载、训练/测试相机划分、点云初始化
│  ├─ cameras.py           # 训练相机数据结构（矩阵、图像、相机中心）
│  └─ blur_kernel.py       # 去模糊核网络（GTnet）与位置编码
├─ configs/                # 去模糊配置
├─ train.py                # 训练脚本
├─ render.py               # 渲染脚本
├─ SIBR_viewers/           # SIBR 可视化器（含 C++ 相机实现）
└─ demo_jobs/              # 任务输出目录（运行后自动生成）
```

---

## 4. 系统架构

系统采用“前后端分离 + 本地文件服务”的结构：

1. 前端（`web/*`）
- 提供任务表单、状态看板、日志视图、结果画廊与点云预览。
- 通过轮询 `/api/jobs/{job_id}` 获取任务实时状态。

2. 后端 API（`app.py`）
- 提供任务提交、状态查询、系统诊断等接口。
- 挂载 `/static`（前端资源）与 `/jobs`（任务产物）。

3. 任务执行层（`web_backend/jobs.py`）
- 异步线程执行 COLMAP + 训练 + 渲染全链路。
- 将状态写入 `job.json`，日志写入 `job.log`。

4. 数据与产物
- 输入图像保存在 `demo_jobs/<job_id>/dataset/images`
- 模型与渲染产物保存在 `demo_jobs/<job_id>/output`

---

## 5. 参数自动策略（当前版本）

前端已隐藏以下参数，由后端自动决策：

1. `hold`（测试抽帧间隔）
- `<=16` 张图：留 `1` 张测试（其余训练）
- `17~40` 张图：留 `2` 张测试
- `41~80` 张图：留 `3` 张测试
- `>80` 张图：约 `10%` 做测试（最多 `10` 张）

2. `deblur_mode`
- `official_3dgs`：自动设为 `none`
- `mlp_deblur`：若场景名包含 `synthetic`，自动设为 `synthetic_camera_motion`，否则 `real_camera_motion`

3. `pts_iter` / `pts_N_pts`（随机 add_points）
- 默认关闭随机补点：`pts_iter=999999`, `pts_rate=0`, `pts_N_pts=0`
- 若显式启用随机补点，训练器仍会按 `pts_iter = clamp(iterations * 0.12, 800, 6000)` 调整触发时机
- 旧默认组合 `pts_iter=2500`, `pts_rate=1.1`, `pts_N_pts=200000` 会被视为历史默认并自动关闭

4. `densify_until_iter`
- 默认 5000 步训练只 densify/prune 到 3000 步
- 训练器会把过大的值压到 `max(iterations * 0.6, densify_from_iter + densification_interval)`，且不超过总迭代数
- 目的：避免后段训练持续做昂贵拓扑变化

---

## 5.1 核心模块说明（代码阅读指引）

1. `scene/blur_kernel.py`
- 提供 `Embedder` 与 `GTnet`
- `GTnet` 根据位置、视线方向和高斯参数预测模糊核相关增量
- 运动模糊分支可额外预测位置偏移（`pos_delta`）

2. `scene/cameras.py`
- `Camera`：训练用完整相机对象，含图像、view/proj/full_proj 矩阵与相机中心
- `MiniCam`：轻量相机对象，主要用于渲染或交互视图

3. `scene/__init__.py`
- `Scene` 统一加载 COLMAP/Blender 数据
- 自动读取 `hold=*` 规则划分训练/测试视角
- 支持从历史迭代加载点云或从输入点云初始化

4. `SIBR_viewers/src/core/graphics/Camera.cpp`
- C++ 侧相机实现
- 包含投影/反投影、视锥检测、立体/正交相机设置等逻辑

---

## 6. 本地运行（开发模式）

### 6.1 创建环境

```bash
git clone <your-repo> --recursive
cd <your-repo>
conda env create -f environment.yml
conda activate deblurring_3dgs
```

### 6.2 配置 COLMAP

确保 `colmap` 命令可用；或设置：

```bash
# Linux/macOS
export DBULR_COLMAP_BINARY=/path/to/colmap

# Windows PowerShell
$env:DBULR_COLMAP_BINARY="C:\\path\\to\\colmap.exe"
```

### 6.3 启动服务

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

浏览器访问：
- `http://127.0.0.1:8000`

---

## 7. Docker 部署

项目已包含 `Dockerfile` 与 `docker-compose.yml`，可直接容器化部署。

### 7.1 构建并启动

```bash
docker compose up --build
```

### 7.2 访问

- 默认访问地址：`http://127.0.0.1:8000`

> 若使用 GPU，请确保宿主机已安装 NVIDIA Container Toolkit，并在 compose 中正确配置 GPU 运行时。

---

## 8. API 说明（简版）

1. `GET /api/health`
- 返回服务健康状态

2. `GET /api/system`
- 返回运行依赖检测结果（COLMAP、CUDA、扩展模块等）

3. `POST /api/jobs`
- 提交训练任务（multipart/form-data）
- 必需：`files[]`
- 常用：`scene_name`、`training_method`、`iterations`、`use_gpu`

4. `GET /api/jobs/{job_id}`
- 获取任务状态、进度、ETA、日志链接、渲染与点云产物

---

## 9. 前端使用说明

1. 上传同一场景多视角图片（建议相邻视角有重叠）
2. 选择训练方式并提交任务
3. 在“任务工作区”查看阶段进度与日志
4. 训练完成后：
- 查看渲染结果
- 下载 PLY / `cameras.json`
- 点击“加载 PLY 预览”进行点云交互浏览
- 点击缩略图可查看大图（支持 ESC 关闭）

---

## 10. 常见问题

1. 页面仍报旧版前端错误
- 强制刷新浏览器缓存（`Ctrl+F5`）

2. COLMAP 不可用
- 检查 `colmap` 是否在 PATH
- 或设置 `DBULR_COLMAP_BINARY`

3. ETA 与日志不一致
- 当前实现已优先解析 tqdm 日志中的 `<ETA>`；若日志无 ETA，会回退到阶段经验值估算

4. 点云无法预览
- 确认任务已生成 `.ply`
- 确认浏览器支持 Canvas

---

## 11. 参考与致谢

- Deblurring 3D Gaussian Splatting (ECCV 2024)
- 3D Gaussian Splatting (Graphdeco-Inria)

