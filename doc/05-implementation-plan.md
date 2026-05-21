# 开发实施计划

## 1. 开发原则

- 算法相关功能不能用占位函数、假产物或固定假结果冒充成功。
- API 请求不执行长任务；预览和精细重建必须进入 Redis 队列。
- 文档、测试和前端文案要区分“当前已实现”和“规划能力”。
- 修改默认管线时必须同步 `pipeline_parameters.py`、测试、Worker、文档和运行时预检。

## 2. 当前技术栈

| 层级 | 技术 |
| --- | --- |
| 前端 | Next.js / React / TypeScript |
| Viewer | Spark SPZ viewer，PLY fallback/debug path |
| 后端 | FastAPI / SQLAlchemy |
| 数据库 | PostgreSQL；测试可用 SQLite |
| 队列 | Redis |
| 存储 | local storage；可配置 S3/MinIO |
| Preview Worker | LiteVGGT + Spark SPZ |
| Fine Worker | COLMAP/PyCOLMAP + EAP + DashDeblurGroupGS + Spark SPZ |
| 资源监控 | psutil / nvidia-smi / torch CUDA |
| 部署 | Docker Compose |

## 3. 已落地里程碑

### M1：基础框架

已完成：

- FastAPI、Next.js、Docker Compose、健康检查、资源接口。
- 首页、上传页、项目页、详情页、分享页、反馈页、关于页、管理页、参数页。
- 默认管理员账号初始化。

### M2：用户、项目与上传闭环

已完成：

- 用户、项目、素材、上传会话、反馈等模型。
- Bearer token 认证和项目归属校验。
- 直接上传和分片上传。
- 图片缩略图、视频封面、素材统计、补传和删除。
- 项目 `source_version` 与 stale preview 规则。

### M3：任务队列、Worker 和运行时检查

已完成：

- `preview_tasks`、`fine_tasks` 队列。
- `worker-preview`、`worker-fine` 心跳和任务日志。
- 运行时预检：GPU、torch、Transformer Engine、Spark SPZ、LiteVGGT、DashDeblurGroupGS、ffmpeg、pycolmap、COLMAP CLI。
- 算法失败不创建成功模型 artifact。

### M4：真实预览

已完成：

- 图片 LiteVGGT -> Spark SPZ。
- 单视频 ffmpeg 抽帧 -> LiteVGGT speed defaults -> Spark SPZ。
- 中间 PLY、debug splats、preview meta、log artifact。
- 缺权重或产物为空时失败。

### M5：实时摄像头

未实现：

- 没有摄像头采集页面。
- 没有实时帧上传 API。
- 没有增量预览 artifact 语义。

### M6：精细重建

已完成主线：

- 图片和单视频 fine task。
- 默认 `dash_deblur_group_gs`。
- 默认 `colmap_global`，显式支持 `gcolmap`、`colmap_cli`、`colmap`、`pycolmap`。
- EAP 点云增强、DashDeblurGroupGS 训练、far-noise filtered `final.ply`、`final_web.spz`、`final_viewer_meta.json`、`metrics.json`。
- 管理员可配置 scene defaults。

未完成或限制：

- 多机/多 GPU 智能调度仍是后续工作。
- `.rad` LOD 生成未接入。
- Mesh export 未实现。
- 外部训练器覆盖仅支持兼容 `train.py --config` 的 DashDeblurGroupGS 风格目录。

### M7：分享、反馈和管理

已完成：

- 项目分享 token。
- Artifact 下载 URL 和本地 token 文件访问。
- 用户反馈和管理员反馈列表。
- 管理端项目、用户、任务、Worker、资源、运行时预检。

未完成：

- OBJ/GLB Mesh 导出。
- 更细的任务中断外部进程能力。

## 4. 当前优先级

1. 用真实 GPU 数据继续 smoke test 图片/视频预览和图片/视频 fine。
2. 强化 fine 失败诊断：COLMAP global mapper、EAP、CUDA 扩展、SPZ 转码各阶段给出更短、更可操作的错误。
3. 补充端到端回归脚本，覆盖上传 -> 预览 -> fine -> viewer-config。
4. 决定是否实现 Mesh export 或先完善论文实验图表脚本。

## 5. 风险与处理

| 风险 | 处理 |
| --- | --- |
| COLMAP global mapper 构建或命令缺失 | Docker build 阶段强制检查，运行时 preflight 再检查 |
| 权重重复下载或中断 | `model-cache` 挂载，`.part` 和 Range 续传 |
| CUDA 扩展重编译耗时 | BuildKit cache、wheel cache、保留 `.docker-build-cache` 和 `3dgsbi-worker:local` |
| 旧参数污染当前训练 | preset marker 校验，移除 `birth_iter/protect_new_points_iters` |
| 视频输入质量不稳定 | 视频先抽帧、去重、筛帧，metrics 记录保留帧数 |
| 用户以为规划功能已可用 | 文档和前端只把当前能力标为可用 |
