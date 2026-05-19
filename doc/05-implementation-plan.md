# 开发实施计划

## 1. 开发原则

- 算法相关功能不能用占位函数、假产物、固定假结果冒充成功。
- 未接入或未配置真实算法时，接口应返回明确错误，例如 `ALGORITHM_NOT_CONFIGURED`。
- 前端不提供算法选择按钮，只提供用户意图入口。
- 每个阶段都要有可验证结果，避免只写页面或只写后端。
- 测试可以覆盖失败路径和环境检查，但不能通过伪造算法输出替代真实功能验收。

## 2. 推荐技术栈

| 层级 | 技术 |
| --- | --- |
| 前端 | Next.js / React / TypeScript |
| 3D Viewer | Spark 2.0 |
| 后端 | FastAPI / Python |
| 数据库 | PostgreSQL |
| 队列与缓存 | Redis |
| 对象存储 | MinIO，生产或实验部署可换 S3 |
| Worker | Python GPU Worker，调用项目融合算法代码和已登记真实算法组件 |
| 资源监控 | NVML / nvidia-smi / psutil |
| 部署 | Docker Compose 起步，后续拆分多 GPU Worker |

## 3. 里程碑

### M1：项目骨架与基础页面

目标：前后端可以启动，首页、新建项目、项目管理和登录入口可访问。

交付内容：

- 前端路由：首页、新建项目、完整素材上传页、项目管理页、项目详情页、管理面板、关于页面；摄像头预览入口待新管线重写后恢复。
- 后端健康检查接口。
- 基础配置文件和环境变量。
- CPU、GPU、显存资源采集接口。

验证：

- 浏览器能打开首页和项目管理页。
- `GET /health` 返回正常。
- 首页能显示系统名称和资源占用数据；无 GPU 环境时展示明确不可用状态。

### M2：用户、项目与上传闭环

目标：用户可以登录、创建项目、上传图片或视频、查看素材统计。

交付内容：

- `users`、`projects`、`media_assets`、`upload_sessions` 数据表。
- 登录、项目创建、项目查询、标签编辑、上传、补传、删除素材接口。
- 前端上传组件、缩略图、大图查看、素材统计。
- MinIO 保存原始素材和缩略图。

验证：

- 上传后能在对象存储路径中看到文件。
- 项目状态从 `CREATED` 变化到 `UPLOADING`、`PREPROCESSING`。
- 用户只能查看自己的项目和素材。

### M3：任务队列、Worker 框架与真实环境检查

目标：任务可以入队、分配 Worker、上报进度；算法未配置时明确失败，不伪造成功。

交付内容：

- `tasks`、`artifacts`、`worker_heartbeats` 数据表。
- Redis 队列和任务优先级。
- Worker 心跳、GPU 显存和利用率上报。
- 算法环境检查：仓库路径、依赖、权重、许可证登记。
- 任务失败路径：缺少算法时返回 `ALGORITHM_NOT_CONFIGURED`。

验证：

- 创建预览任务后，任务能从 `queued` 到 `running`。
- 若算法环境缺失，任务进入 `failed`，错误原因清晰。
- 不能生成假的 `preview.spz` 或假产物记录。

### M4：真实极速预览管线

目标：图片或视频可以通过真实算法生成粗糙 3DGS 预览模型。

交付内容：

- LiteVGGT 适配器。
- 视频/摄像头极速预览适配器待重新设计。
- 真实 `preview.spz` 和 `preview_lod1.rad` 产物上传。
- 素材质量分析和补拍建议。

验证：

- 图片素材可以生成真实 `preview.spz`。
- 视频素材可以生成真实预览产物。
- 预览完成后 Viewer 可以加载模型。
- 算法失败时前端展示明确错误，不展示假模型。

### M5：实时摄像头粗重建

目标：摄像头预览入口和对应预览管线待重写。

交付内容：

- 摄像头采集页面待重新设计。
- 帧上传或实时流通道待重新设计。
- 实时视频分片处理待重新设计。
- 实时预览产物或增量更新待重新设计。
- 结束录制后的素材统计、重拍和精细重建入口待重新设计。

验证：

- 新管线未实现前不提供页面入口、分片 API 或渐进式预览 artifact。

### M6：精细重建与 LOD

目标：支持精细重建、LOD 生成和最终产物管理。

交付内容：

- 图片项目精细重建任务入口，可直接从上传素材启动，不要求先完成极速预览。
- `worker-preview` 与 `worker-fine` 复用同一个统一 CUDA worker 镜像和同一个 `model-cache`，不创建第二套 conda/CUDA 11.x 环境。
- 现有 COLMAP CLI / pycolmap 作为 fine SfM 前端，输出标准 `images/` 与 `sparse/0`。
- DashDeblurGroupGS 作为外部训练仓库接入，backend 只生成配置、启动训练、定位 `final.ply`、转码 `final_web.spz`。
- Speedy-Splat、FastGS pruning、MobileGS/LM-RS 路径不进入默认主线。
- `metrics.json` 保存和展示。

验证：

- 精细重建产出真实非空 `final.ply`、`final_web.spz` 和 `metrics.json`。
- RAD 只在配置真实转换器时生成 `final_lod.rad`；未配置时任务仍可成功，但不会创建假 RAD。
- Viewer 优先加载同一 `source_version` 的 `final_spz`，没有最终产物时才回退极速预览。

### M7：Mesh 导出、分享、反馈和管理面板

目标：补齐项目交付、用户反馈和管理员资源管理能力。

交付内容：

- Mesh 导出任务。
- PLY 下载和可选 OBJ、GLB 导出。
- 签名下载链接。
- 分享页面。
- 用户问题反馈。
- 管理面板 GPU、队列、任务、用户项目数、存储占用、训练占用统计。
- 算法许可证登记和关于页面。

验证：

- 用户可以下载 `.ply`、`.obj`、`.glb`。
- 分享链接能打开模型。
- 用户可以提交反馈。
- 管理员能看到每个用户的占用、项目数和训练占用。

## 4. 首轮编码建议

新窗口第一次让 Codex 写代码时，建议这样提出：

```text
请基于 doc 目录的文档，先实现项目基础框架和真实任务生命周期：
1. FastAPI 后端，提供登录、项目创建、项目列表、素材上传、任务创建、任务状态查询接口；
2. 使用 PostgreSQL/SQLite 开发库保存用户、项目、素材、任务；
3. 使用 MinIO 或本地可替换存储保存真实上传文件；
4. Worker 必须实现真实环境检查，算法未配置时返回 ALGORITHM_NOT_CONFIGURED，不能生成假 preview.spz；
5. Next.js 前端实现首页、上传页、项目管理页和项目详情页，显示资源占用、上传统计、任务进度和错误状态。
暂时未接入的算法功能只能显示不可用或失败，不能用占位产物冒充成功。
```

## 5. 主要风险

| 风险 | 处理方式 |
| --- | --- |
| 真实算法仓库集成复杂 | 先实现适配器接口和环境检查，再接入 LiteVGGT；视频/实时视频管线另行重写 |
| GPU 资源不足 | 明确显示资源不可用；算法任务失败返回可解释错误，不伪造结果 |
| SPZ/RAD 测试资产缺失 | 使用真实算法生成的最小样例资产；没有资产时 Viewer 展示未完成状态 |
| 许可证限制 | 从第一版就维护 `algorithm_registry` 和关于页面 |
| 长任务失败难排查 | 每个任务都保存日志、错误、阶段、进度和 ETA |
| 前端功能过多 | 先完成首页、新建项目、上传、任务进度和项目详情主流程 |

## 6. 当前批次落地状态

本批次已完成 M1-M4 的主要工程闭环，并开始落地 M6 图片精细重建：

- FastAPI API、Next.js 前端、登录页、项目页、上传页、详情页、管理页、反馈页和关于页已具备可启动骨架。
- PostgreSQL 目标模型、Alembic、Redis 任务队列、MinIO 对象存储适配、独立 preview worker 已接入。
- 本机单元测试使用 SQLite 和本地对象存储后端，验证认证、权限隔离、任务入队、worker 失败路径和禁止假 artifact。
- Docker Compose 已覆盖 backend、worker、postgres、redis、minio、frontend；真实算法 bootstrap 仍独立执行，不在 API 启动时下载仓库或权重。
- 当前成功产物仍只允许来自真实 LiteVGGT → Spark-SPZ 命令输出；未配置算法时任务失败且 artifact 表为空。
- DashDeblurGroupGS 训练器内置在 `worker/trainer/dash_deblur_group_gs`。worker Dockerfile 会复制它用于 CUDA 扩展构建；本地 Docker Compose 会把同一目录 bind mount 到 `/opt/dash_deblur_group_gs`，也可由 `DASH_DEBLUR_GROUP_REPO` 显式覆盖；训练失败时不会创建假 artifact。
- Worker Dockerfile 从 upstream COLMAP 源码构建运行时，并在 build 阶段要求 `global_mapper`、`hierarchical_mapper`、`model_clusterer`、`model_splitter` 存在，避免运行时才发现 apt COLMAP 能力不足。

## 7. 2026-05-18 Implementation Update

- M6 fine pipeline now uses `pycolmap` as the default production SfM frontend, with `colmap_cli` still available for explicit CLI runs.
- Fine input contract is JPG/PNG images or extracted video frames. Missing EXIF camera metadata is normal; COLMAP estimates camera/intrinsics and sparse points from images.
- Preview LiteVGGT remains separate from fine COLMAP and keeps its own runtime/package namespace.
- GTnet deblur, densification, and `add_points()` live in the embedded trainer, not in backend request handling code. Backend only resolves scene/deblur presets and writes config.
- Worker dependency goal: one PyTorch/CUDA baseline and one repo-integrated DashDeblurGroupGS trainer. Do not add Speedy-Splat, duplicate renderer pruning paths, Kaolin, Open3D, Gradio, duplicate torch/CUDA, pip cuDNN, or non-native `birth_iter`/`protect_new_points_iters` pruning state.
- Local Docker Compose bind-mounts backend app code, frontend code, and the embedded trainer. Ordinary Python/TypeScript edits require service recreation, not image rebuild.
- Earlier FastGS large-scene notes map to the current DashDeblurGroupGS path: global COLMAP first, then chunk-compatible DashDeblurGroupGS training on one shared coordinate system.
- Tests cover COLMAP routing, DashDeblurGroupGS config/command generation, pipeline aliases, and static runtime checks.
