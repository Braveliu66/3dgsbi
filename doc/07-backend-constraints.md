# 后端实现约束

## 1. 真实算法约束

允许：

- 缺少算法环境时任务失败。
- 缺少权重时任务失败或按配置下载权重。
- GPU、CUDA、COLMAP、ffmpeg、Spark、pycolmap 或 CUDA 扩展不可用时返回明确错误。
- 上传任务日志 artifact 帮助诊断。

禁止：

- 生成空 `preview.spz` 或 `final.ply` 并标记成功。
- 写固定假 `metrics.json` 作为训练结果。
- 用 `sleep` 模拟算法成功。
- 在 API 请求线程里执行长算法。
- 引入与当前主线无关的训练器、重复 CUDA/Torch 环境或假兼容层。

## 2. 当前算法边界

| 算法 | 当前状态 |
| --- | --- |
| LiteVGGT | bundled preview vendor，必须有 `model-cache/litevggt/te_dict.pt` |
| Spark SPZ | worker 内 Node 22 + `@sparkjsdev/spark` 转码 |
| DashDeblurGroupGS Fine | bundled trainer，Compose 挂载到 `/opt/dash_deblur_group_gs` |
| COLMAP | worker Dockerfile 从 upstream 构建，默认 fine 使用 `global_mapper` |
| PyCOLMAP | 显式 fine backend 和 EAP/metadata 读取依赖 |
| LingBot 视频预览 | 已不属于当前算法 registry |
| EDGS/RoMA dense initialization | 后端显式拒绝 |
| Speedy-Splat/FastGS | 不属于默认 fine 主线 |

## 3. 模型权重缓存

- 权重统一放在 `model-cache/`，通过 Compose 挂载给 worker。
- 当前预览权重：`model-cache/litevggt/te_dict.pt`。
- 权重不提交 Git。
- `MODEL_AUTO_DOWNLOAD=true` 时 worker 可下载缺失权重。
- 下载优先 `https://hf-mirror.com`，但 fallback 必须可配置。
- 下载必须使用 `.part`、lock 和 Range 续传，不能只写容器临时目录。

## 4. 任务与队列

- `POST /tasks/preview` 只创建 task 并推入 `preview_tasks`。
- `POST /tasks/fine` 只创建 task 并推入 `fine_tasks`。
- Worker 更新 `running/succeeded/failed/canceled`、阶段、进度、ETA 和 metrics。
- fine worker 启动时恢复中断的 `queued/running` fine task。
- 任务失败后不盲目重试；具体重试策略后续再设计。

## 5. 存储要求

- 存储路径必须按 `users/{user_id}/projects/{project_id}` 隔离。
- 大文件上传使用分片上传，分片大小上限 64MB。
- 删除项目时要删除数据库记录；对象清理策略可异步增强。
- 本地下载 token 默认 1 小时过期。
- artifact 必须记录大小、checksum、kind、source_version 和 metadata。

## 6. 权限要求

- 普通用户只能访问自己的项目、素材、任务和产物。
- 管理员可访问管理 API。
- 分享链接只返回公开项目摘要和 Viewer 配置。
- `/api/algorithms` 和 `/api/pipeline-parameters/schema` 可公开读取。

## 7. 可观测性

每个任务应记录：

- `current_stage`
- `progress`
- `eta_seconds`
- `worker_id`
- `started_at` / `finished_at`
- `error_code` / `error_message`
- `logs`
- `metrics`
- 任务日志 artifact

管理员运行时预检应覆盖：

- Python。
- GPU。
- torch/CUDA。
- Transformer Engine。
- preview worker。
- fine worker。
- LiteVGGT 权重。
- Spark SPZ。
- DashDeblurGroupGS trainer。
- ffmpeg、git、node、COLMAP、pycolmap。

## 8. Fine 参数安全

- 默认 `fine_sfm_backend=colmap_global`。
- `gcolmap/global/global_mapper/colmap_glomap` 归一化到 `colmap_global`。
- `colmap` 和 `colmap_cli` 使用 CLI incremental mapper；`pycolmap` 显式走 PyCOLMAP。
- `fine_deblur_mode` 只接受 `motion|defocus|sharp`；旧 `mix/auto/automatic` 视为 `motion`。
- `fine_edgs_enabled=true` 必须失败。
- `birth_iter` 和 `protect_new_points_iters` 不能重新进入 trainer contract。
- `add_points()` 默认禁用：`pts_iter=999999`、`pts_rate=0`、`pts_N_pts=0`。

## 9. 构建与镜像

- Worker 镜像需要构建 COLMAP，并检查 `feature_extractor`、`mapper`、`hierarchical_mapper`、`image_undistorter`、`model_analyzer`、`model_splitter`。
- 普通源码修改使用 `docker compose up -d --force-recreate ...`。
- 依赖、Dockerfile、CUDA 扩展、系统包、基础镜像或 submodule 变化才 rebuild。
- 不要随意清理 `.docker-build-cache/`、Docker builder cache 或 `3dgsbi-worker:local`，否则 COLMAP/CUDA 扩展会重新编译。
