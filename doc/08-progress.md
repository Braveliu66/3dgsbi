# 当前工程进度

## 已完成

- FastAPI、Next.js、PostgreSQL、Redis、本地对象存储、preview worker、fine worker 的单机 Docker Compose 拓扑已建立。
- 极速预览继续使用真实 LiteVGGT、EDGS 和 Spark SPZ 路径；视频/实时视频极速预览管线已清理，待重写。
- 精细重建默认管线已切换为 `mobilegs_lmrs`，不再调用 EDGS preview runtime，也不再要求 `romatch`、RoMA 权重或 DINOv2 权重。
- `worker-fine` 权重检查已接入 `model-cache/amb3r/amb3r.pt` 自动下载；本地已有文件时优先复用。
- 精细重建入口现在按 MobileGS 顺序执行：JPG/PNG 归一化、模糊分析和低质帧过滤、AMB3R-SfM 生成无 EXIF 相机参数依赖的 COLMAP 兼容 `sparse/0`、DeblurMLP-MobileGS 训练、可选 LM-RS、Spark SPZ 转码。预览链路仍使用 LiteVGGT。
- Docker worker 已收敛为统一 `worker` 镜像：`worker-preview` 和 `worker-fine` 复用同一个 CUDA/PyTorch 环境，只用启动命令区分监听队列。
- 统一 worker 构建已增加 LM-RS 源码 BuildKit cache，任务运行时不下载算法源码，重复构建时优先复用缓存。
- Phase 2 已接入 LM-RS matrix-free wrapper：从 `fine_lm_start_iter` 起初始化 `cgState`、`CGOptimizer`，调用 `gauss_newton_step()` 并使用 LM-RS `get_JTv/get_Diag/get_JTJv` CUDA 符号；符号缺失时任务失败，不再假装回退为已完成的 LM。
- FastGS Compact Box 已作为 LM-RS rasterizer patch 固定进入构建输入：`worker/patches/lmrs-fastgs-compact-box.patch` 会在统一 worker 镜像和 `scripts/bootstrap-repos.*` 中应用，保留 LM-RS matrix-free 接口。
- AMB3R-SfM 默认不再自动降级 pycolmap；`pycolmap` 只作为显式诊断后端，旧 fine `litevggt` 后端值作为兼容别名映射到 AMB3R。
- DeblurMLP 已按 Deblurring-3DGS GTnet 核心算法接入训练期渲染，motion/mixed 使用 position moments，defocus 只调整 scale/rotation，最终仍导出标准 sharp Gaussian PLY。
- Worker Dockerfile 已拆分 `transformer-engine[pytorch]==2.4.0` 安装；安装前检查 `nvcc`、torch/torchvision 和 `cudnn.h`，缺 header 时只补系统 `libcudnn9-dev-cuda-12`；该独立层同步提前安装 `einops==0.8.0`，避免 TE 导入校验早于 LiteVGGT runtime requirements。

## 当前限制

- LM-RS Phase 2 和 Compact Box 现在都要求统一 worker 镜像中的 patched LM-RS rasterizer 可用；缺少 matrix-free 符号或 Compact Box patch marker 会在 preflight/smoke check 失败。
- AMB3R-SfM 少于 3 张真实图片会失败；预览 LiteVGGT 不受 fine 后端替换影响。
- `pycolmap` 只保留为显式开发诊断路径；生产 worker requirements 不再默认安装。
- GPU 端到端验收仍依赖可用 NVIDIA GPU、CUDA worker 镜像和完整 `amb3r/amb3r.pt`；本地静态测试不能替代统一 worker CUDA 编译和短迭代真实图片验证。

## 下一步

- 在真实 GPU 环境启动统一 worker，确认 patched LM-RS rasterizer 的 `get_JTv/get_Diag/get_JTJv` 和 `MOBILEGS_COMPACT_BOX` smoke check 通过。
- 确认 `model-cache/amb3r/amb3r.pt` 自动下载或手动预置完成后，用 3 张和 8 张以上 JPG/PNG 手机图像分别验证 AMB3R-SfM、DeblurMLP 训练、`final.ply`、`final_web.spz` 和 `metrics.json`；同时确认预览 LiteVGGT 任务仍可运行。
