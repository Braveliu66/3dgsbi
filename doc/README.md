# 3DGS 重建系统文档索引

本文档目录描述当前代码库的真实系统边界。阅读顺序建议如下：

1. `01-requirements.md`：系统定位、已实现能力、规划能力和验收口径。
2. `02-architecture.md`：前端、后端、Worker、存储和算法适配层结构。
3. `03-workflows-state.md`：项目状态、上传、预览、精细重建和事件流。
4. `04-data-api.md`：数据库模型、对象路径、REST API、SSE 和任务消息。
5. `05-implementation-plan.md`：当前里程碑完成情况和后续工作。
6. `06-frontend-pages.md`：Next.js 页面与前端交互规格。
7. `07-backend-constraints.md`：真实算法、权重缓存、队列、权限和可观测性约束。
8. `08-progress.md`：截至 2026-05-21 的工程状态。
9. `09-deployment.md`：Docker Compose、本地开发、缓存和重建规则。
10. `10-fine-pipeline.md`：当前精细重建主线的权威说明。

## 当前系统主线

- 前端：Next.js / React / TypeScript，运行端口由 Compose 暴露为 `3001`。
- 后端：FastAPI，提供认证、项目、上传、任务、产物、分享、反馈、管理和参数默认值 API。
- 数据层：PostgreSQL 目标部署；本机测试可用 SQLite。Redis 用于任务队列。
- 存储：支持本地对象存储和 S3/MinIO 风格配置，路径按用户和项目隔离。
- 预览：图片和单视频项目均走 LiteVGGT，视频先抽帧，再生成 Spark SPZ。
- 精细重建：默认 `dash_deblur_group_gs`，SfM 默认 `colmap_global`，再经 EAP、DashDeblurGroupGS 训练、PLY 过滤和 SPZ 转码。
- Worker：`worker-preview` 和 `worker-fine` 使用同一个 CUDA worker 镜像，但监听不同队列。

## 重要约束

- 算法任务必须调用真实代码。缺权重、缺 CUDA、缺命令或产物无效时任务失败，不能创建假产物。
- `model-cache/` 存放模型权重，`repo-cache/` 只用于显式外部仓库缓存或兼容覆盖。
- 默认精细重建训练器位于 `worker/trainer/dash_deblur_group_gs`，Compose 挂载到 `/opt/dash_deblur_group_gs`。
- 普通 Python、TypeScript 或 trainer 源码修改后重建服务即可；只有依赖、Dockerfile、CUDA 扩展、系统包或 submodule 变化才需要重新 build。
- `birth_iter`、`protect_new_points_iters`、EDGS/RoMA dense initialization、LingBot 视频预览、Speedy-Splat/FastGS 默认路径均不属于当前主线。
