# 当前工程进度

## 已完成

- FastAPI、Next.js、PostgreSQL、Redis、本地对象存储、preview worker、fine worker 的单机 Docker Compose 拓扑已建立。
- 极速预览继续使用真实 LiteVGGT 和 Spark SPZ 路径；视频/实时视频极速预览管线已清理，待重写。
- 精细重建入口现在按 MobileGS 顺序执行：JPG/PNG 归一化、模糊分析和低质帧过滤、pycolmap 生成 COLMAP 兼容 `sparse/0`、DeblurMLP-MobileGS 训练、可选 LM-RS、Spark SPZ 转码。预览链路仍使用 LiteVGGT。
- Docker worker 已收敛为统一 `worker` 镜像：`worker-preview` 和 `worker-fine` 复用同一个 CUDA/PyTorch 环境，只用启动命令区分监听队列。
- 统一 worker 构建已增加 LM-RS 源码 BuildKit cache，任务运行时不下载算法源码，重复构建时优先复用缓存。
- Phase 2 已接入 LM-RS matrix-free wrapper：从 `fine_lm_start_iter` 起初始化 `cgState`、`CGOptimizer`，调用 `gauss_newton_step()` 并使用 LM-RS `get_JTv/get_Diag/get_JTJv` CUDA 符号；符号缺失时任务失败，不再假装回退为已完成的 LM。
- FastGS Compact Box 已作为 LM-RS rasterizer patch 固定进入构建输入：`worker/patches/lmrs-fastgs-compact-box.patch` 会在统一 worker 镜像和 `scripts/bootstrap-repos.*` 中应用，保留 LM-RS matrix-free 接口。
- pycolmap 是图片 fine 默认 SfM 后端；旧 fine `litevggt` 后端值不再作为图片 fine 后端。
- DeblurMLP 已按 Deblurring-3DGS GTnet 核心算法接入训练期渲染，motion/mixed 使用 position moments，defocus 只调整 scale/rotation，最终仍导出标准 sharp Gaussian PLY。
- Worker Dockerfile 已拆分 `transformer-engine[pytorch]==2.4.0` 安装；安装前检查 `nvcc`、torch/torchvision 和 `cudnn.h`，缺 header 时只补系统 `libcudnn9-dev-cuda-12`；该独立层同步提前安装 `einops==0.8.0`，避免 TE 导入校验早于 LiteVGGT runtime requirements。

## 当前限制

- LM-RS Phase 2 和 Compact Box 现在都要求统一 worker 镜像中的 patched LM-RS rasterizer 可用；缺少 matrix-free 符号或 Compact Box patch marker 会在 preflight/smoke check 失败。
- pycolmap 少于 3 张真实图片会失败；预览 LiteVGGT 不受 fine 后端替换影响。
- `pycolmap` 只保留为显式开发诊断路径；生产 worker requirements 不再默认安装。

## 下一步

- 在真实 GPU 环境启动统一 worker，确认 patched LM-RS rasterizer 的 `get_JTv/get_Diag/get_JTJv` 和 `MOBILEGS_COMPACT_BOX` smoke check 通过。


- DeblurMLP warmup defaults to 3000 iterations, then activates GTnet and applies xyz lr scale `0.1`.
- No local Python smoke was run for this update; coverage is static checks and code review against the requested plan.
