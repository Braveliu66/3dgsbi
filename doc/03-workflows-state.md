# 核心流程与状态机

## 1. 项目状态机

```text
CREATED
  -> UPLOADING
  -> PREPROCESSING
  -> PREVIEW_RUNNING
  -> PREVIEW_READY
  -> FINE_QUEUED
  -> GLOBAL_OPTIMIZING
  -> FINE_RUNNING
  -> COMPLETED

任意运行中状态
  -> FAILED
  -> CANCELED

COMPLETED
  -> MESH_EXPORT_RUNNING
  -> MESH_EXPORT_READY
```

`GLOBAL_OPTIMIZING` 只在长视频精细重建且开启全局优化时出现。

## 2. 图片极速预览流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant FE as 前端
    participant API as 后端
    participant S3 as 对象存储
    participant Q as Redis 队列
    participant W as GPU Worker
    participant V as Viewer

    U->>FE: 创建项目并选择图片
    FE->>API: 创建项目
    API-->>FE: 返回 project_id
    FE->>API: 分片上传图片
    API->>S3: 保存 raw/images
    API->>Q: 创建 preview 任务
    API-->>FE: 返回 task_id
    W->>Q: 拉取 preview 任务
    W->>W: 分析图片数量和质量
    W->>W: LiteVGGT 获取位姿和点云
    W->>W: Spark 转码生成 preview.spz
    W->>S3: 上传 preview.spz 和 preview_lod1.rad
    W->>API: 回写任务完成
    API-->>FE: 推送 PREVIEW_READY
    FE->>V: 加载预览模型
```

## 3. 视频极速预览流程

视频极速预览管线已清理，待重新设计与实现。当前系统可以保存视频素材，但不会创建视频预览任务或生成视频预览 artifact。

## 4. 实时摄像头粗重建流程

实时摄像头采集页面、分片上传 API 和实时预览管线已清理。新管线实现后再定义录制素材、任务队列、artifact 和 Viewer 加载流程。

## 5. 精细重建流程

```mermaid
flowchart TD
    Start["用户点击精细重建"] --> Queue["创建 fine 任务"]
    Queue --> Input["图片/视频帧归一化与低质量过滤"]
    Input --> Colmap["现有 COLMAP CLI / pycolmap 生成 images + sparse/0"]
    Colmap --> Config["生成 DashDeblurGroupGS 配置"]
    Config --> Train["Deblur + Dash + Group 训练"]
    Train --> Export["导出 final.ply 并转码 final_web.spz"]
    Export --> LOD
    LOD --> Save["保存 final 产物和 metrics.json"]
```

## 6. Mesh 导出流程

1. 用户在项目详情页选择导出格式。
2. 后端创建 `mesh_export` 任务。
3. Worker 读取 `final.ply`。
4. Worker 使用 MeshSplatting 生成 `.ply`、`.obj`、`.glb`。
5. 产物上传到对象存储。
6. 后端生成签名 URL 并返回前端。

## 7. 任务优先级

| 任务类型 | 优先级 | GPU 策略 |
| --- | --- | --- |
| 实时摄像头 | 待重写 | 新管线实现后重新定义 |
| 极速预览 | 高 | 可并发，快速返回 |
| LOD 生成 | 中 | 可与部分轻量任务错峰 |
| Mesh 导出 | 中 | 默认独占 GPU |
| 精细重建 | 低 | 默认独占 GPU，长任务 |

## 8. 失败处理

- 上传失败：允许用户重新上传失败分片。
- 预处理失败：项目进入 `FAILED`，保留错误日志。
- 预览失败：允许重新发起预览任务。
- 精细重建失败：保留预览结果，不删除已有产物。
- 导出失败：不影响项目最终重建结果。
- 用户取消：任务进入 `CANCELED`，Worker 应尽快停止并清理临时目录。

## 9. 当前实现同步

当前预览任务的真实执行路径如下：

```mermaid
sequenceDiagram
    participant U as 用户
    participant FE as Next.js 前端
    participant API as FastAPI API
    participant DB as PostgreSQL
    participant S as MinIO
    participant Q as Redis
    participant W as Preview Worker
    participant A as 真实算法适配层

    U->>FE: 登录并上传图片
    FE->>API: Authorization: Bearer token
    API->>DB: 校验用户和项目归属
    API->>S: 保存真实上传文件
    API->>DB: 写入 media_assets
    FE->>API: 创建 preview task
    API->>DB: tasks.status = queued
    API->>Q: rpush preview task id
    W->>Q: blpop preview task id
    W->>DB: tasks.status = running
    W->>S: 下载真实上传文件到 work_dir
    W->>A: LiteVGGT -> Spark-SPZ
    alt 成功且 preview.spz 非空
        W->>S: 上传 preview/preview.spz
        W->>DB: 创建 preview_spz artifact
        W->>DB: tasks.status = succeeded, projects.status = PREVIEW_READY
    else 算法未配置或产物无效
        W->>DB: tasks.status = failed, error_code/error_message
        W->>DB: 不创建 artifact
    end
```

当前状态规则：

- API 创建预览任务后只允许进入 `queued`；不得在请求线程内直接执行算法。
- worker 接手后进入 `running`，并写入 `worker_id`、`current_stage`、`started_at`。
- 算法环境失败时进入 `failed`，项目进入 `FAILED`，`artifacts` 表不新增成功产物。
- 只有真实非空 `preview.spz` 上传成功后，任务才进入 `succeeded`，项目进入 `PREVIEW_READY`。
- viewer config 存在 `preview_spz` 时返回 `mode=single`；不存在时返回 `unavailable`。视频/实时视频加载语义待新管线重新定义。
- 当前取消接口只持久化 `canceled` 状态；正在执行的外部算法进程中断和临时目录清理属于后续增强。

## 2026-05-18 Fine Workflow Update

Fine workflow now runs:

1. Frontend sends `scene_type=indoor|outdoor`; backend does not run a scene-classification model.
2. Normalize uploaded JPG/PNG or extracted video frames into RGB JPEG. Missing EXIF is valid; EXIF orientation is only a pixel-rotation hint.
3. Analyze blur and keep at least 3 real images.
4. Run the PyCOLMAP path by default. The worker image still builds upstream COLMAP for explicit `fine_sfm_backend=colmap_cli` runs.
5. Select indoor/outdoor PyCOLMAP matching policy by image count and capture order. Explicit CLI runs still use the COLMAP CLI policy code.
6. Write a COLMAP-compatible scene with `images/` and `sparse/0`.
7. Use `fine_deblur_mode=motion` by default and apply the deblur branch to all training images; legacy `mix` requests are treated as `motion`.
8. Generate a scene-specific DashDeblurGroupGS config for `motion`, `defocus`, or `sharp`.
9. Train DashDeblurGroupGS from the COLMAP scene and surface progress from training logs.
10. Export standard `final.ply`, transcode `final_web.spz`, write `final_viewer_meta.json`, and write `metrics.json`.

`colmap_sparse` is now only a legacy fine pipeline alias. The default fine pipeline is `dash_deblur_group_gs`. The deprecated `fine_sfm_backend=litevggt` fine option remains unsupported; preview LiteVGGT is unchanged.

The trainer is mounted at `/opt/dash_deblur_group_gs` in local Docker Compose, so Python trainer edits are picked up after recreating the workers. The removed `protect_new_points_iters` and `birth_iter` fields are not part of the workflow.

