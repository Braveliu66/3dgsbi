# 数据模型与接口草案

本文件是面向后续编码的草案，用于统一前后端和 Worker 的数据边界。字段可在实现时按实际框架调整。

## 1. 核心数据表

### 1.1 users

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | uuid | 用户 ID |
| username | varchar | 用户名 |
| email | varchar | 邮箱 |
| role | varchar | `user` 或 `admin` |
| created_at | timestamp | 创建时间 |

### 1.2 projects

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | uuid | 项目 ID |
| owner_id | uuid | 所属用户 |
| name | varchar | 项目名称 |
| input_type | varchar | `images`、`video`、`camera` |
| status | varchar | 项目状态 |
| cover_artifact_id | uuid | 封面或默认预览产物 |
| error_message | text | 最近一次失败原因 |
| tags | text[] | 项目标签 |
| total_size_bytes | bigint | 项目总占用 |
| preview_image_uri | text | 卡片预览图，训练中可使用原始素材缩略图 |
| created_at | timestamp | 创建时间 |
| updated_at | timestamp | 更新时间 |

### 1.3 media_assets

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | uuid | 素材 ID |
| project_id | uuid | 项目 ID |
| kind | varchar | `image` 或 `video` |
| object_uri | text | 原始文件路径 |
| thumbnail_uri | text | 缩略图路径 |
| file_name | varchar | 文件名 |
| file_size | bigint | 文件大小 |
| width | int | 宽度 |
| height | int | 高度 |
| duration_seconds | numeric | 视频时长 |
| quality_flags | jsonb | 模糊、曝光、覆盖不足等质量标记 |
| created_at | timestamp | 创建时间 |

### 1.4 upload_sessions

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | uuid | 上传会话 ID |
| project_id | uuid | 项目 ID |
| file_name | varchar | 文件名 |
| file_size | bigint | 文件大小 |
| chunk_size | bigint | 分片大小 |
| total_chunks | int | 分片总数 |
| uploaded_chunks | int | 已上传分片数 |
| status | varchar | `uploading`、`completed`、`failed` |
| object_uri | text | 合并后的对象存储路径 |

### 1.5 tasks

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | uuid | 任务 ID |
| project_id | uuid | 项目 ID |
| type | varchar | `preview`、`fine`、`lod`、`mesh_export` |
| status | varchar | `queued`、`running`、`succeeded`、`failed`、`canceled` |
| priority | int | 数值越大优先级越高 |
| progress | int | 0 到 100 |
| worker_id | varchar | 执行 Worker |
| options | jsonb | 系统生成的执行参数 |
| metrics | jsonb | 任务指标 |
| current_stage | varchar | 当前阶段，例如 `pose_estimation`、`training`、`lod_generation` |
| eta_seconds | int | 预计剩余时间 |
| error_message | text | 失败原因 |
| created_at | timestamp | 创建时间 |
| started_at | timestamp | 开始时间 |
| finished_at | timestamp | 结束时间 |

### 1.6 artifacts

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | uuid | 产物 ID |
| project_id | uuid | 项目 ID |
| task_id | uuid | 任务 ID |
| kind | varchar | `preview_spz`、`final_ply`、`lod_rad`、`mesh_glb` 等 |
| object_uri | text | 对象存储路径 |
| file_name | varchar | 文件名 |
| file_size | bigint | 文件大小 |
| checksum | varchar | 校验值 |
| metadata | jsonb | 额外信息 |
| created_at | timestamp | 创建时间 |

### 1.7 feedback

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | uuid | 反馈 ID |
| user_id | uuid | 提交用户 |
| project_id | uuid | 可选关联项目 |
| title | varchar | 标题 |
| content | text | 描述 |
| attachment_uri | text | 截图或附件 |
| status | varchar | `open`、`processing`、`closed` |
| created_at | timestamp | 创建时间 |

### 1.8 worker_heartbeats

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| worker_id | varchar | Worker ID |
| hostname | varchar | 主机名 |
| gpu_index | int | GPU 序号 |
| gpu_name | varchar | GPU 名称 |
| gpu_memory_total | bigint | 总显存 |
| gpu_memory_used | bigint | 已用显存 |
| gpu_utilization | numeric | GPU 利用率 |
| current_task_id | uuid | 当前任务 |
| last_seen_at | timestamp | 最近心跳 |

### 1.9 algorithm_registry

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | uuid | 记录 ID |
| name | varchar | 算法名称 |
| repo_url | text | 仓库地址 |
| license | varchar | 许可证 |
| commit_hash | varchar | 集成版本 |
| weight_source | text | 权重来源 |
| enabled | boolean | 是否启用 |
| notes | text | 合规备注 |

## 2. 对象存储路径

```text
users/{user_id}/projects/{project_id}/raw/images/{file_name}
users/{user_id}/projects/{project_id}/raw/video/{file_name}
users/{user_id}/projects/{project_id}/thumbs/{media_id}.jpg
users/{user_id}/projects/{project_id}/preview/preview.spz
users/{user_id}/projects/{project_id}/preview/preview_lod1.rad
users/{user_id}/projects/{project_id}/final/final.ply
users/{user_id}/projects/{project_id}/final/final_web.spz
users/{user_id}/projects/{project_id}/final/lod/final_lod0.rad
users/{user_id}/projects/{project_id}/final/lod/final_lod1.rad
users/{user_id}/projects/{project_id}/final/lod/final_lod2.rad
users/{user_id}/projects/{project_id}/final/lod/final_lod3.rad
users/{user_id}/projects/{project_id}/exports/final_mesh.glb
users/{user_id}/projects/{project_id}/logs/{task_id}.log
users/{user_id}/projects/{project_id}/metrics/{task_id}.json
```

## 3. REST API 草案

### 3.1 项目

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/projects` | 获取当前用户项目列表 |
| POST | `/api/projects` | 创建项目 |
| GET | `/api/projects/{project_id}` | 获取项目详情 |
| PATCH | `/api/projects/{project_id}` | 更新项目名称等信息 |
| DELETE | `/api/projects/{project_id}` | 删除项目 |
| GET | `/api/projects/summary` | 当前用户项目总览、总数、训练中数量、总占用 |

### 3.2 上传

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/projects/{project_id}/uploads` | 创建上传会话 |
| PUT | `/api/uploads/{upload_id}/chunks/{chunk_index}` | 上传分片 |
| POST | `/api/uploads/{upload_id}/complete` | 合并分片并入队预处理 |
| GET | `/api/uploads/{upload_id}` | 查询上传状态 |
| GET | `/api/projects/{project_id}/media` | 查询项目素材列表 |
| DELETE | `/api/projects/{project_id}/media/{media_id}` | 删除某个原始素材 |
| GET | `/api/projects/{project_id}/media/stats` | 查询图片数量、视频大小、分辨率、时长等统计 |

### 3.3 任务

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/projects/{project_id}/tasks/preview` | 创建极速预览任务 |
| POST | `/api/projects/{project_id}/tasks/fine` | 创建精细重建任务 |
| POST | `/api/projects/{project_id}/tasks/mesh-export` | 创建 Mesh 导出任务 |
| GET | `/api/tasks/{task_id}` | 查询任务状态 |
| POST | `/api/tasks/{task_id}/cancel` | 取消任务 |

### 3.4 产物

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/projects/{project_id}/artifacts` | 获取项目产物列表 |
| GET | `/api/artifacts/{artifact_id}/download-url` | 获取签名下载链接 |
| GET | `/api/projects/{project_id}/viewer-config` | 获取 Viewer 加载配置 |
| POST | `/api/camera/sessions` | 创建实时摄像头项目 |
| POST | `/api/projects/{project_id}/camera/chunks` | 上传摄像头 MediaRecorder 分片并创建 camera preview task |
| POST | `/api/projects/{project_id}/camera/finish` | 结束实时摄像头 session |

### 3.5 用户与反馈

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/auth/login` | 登录 |
| POST | `/api/auth/logout` | 退出 |
| GET | `/api/me` | 当前用户信息 |
| POST | `/api/feedback` | 提交问题反馈 |
| GET | `/api/feedback` | 当前用户反馈列表 |

### 3.6 管理

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/admin/workers` | Worker 状态 |
| GET | `/api/admin/gpus` | GPU 使用情况 |
| GET | `/api/admin/tasks` | 任务列表 |
| GET | `/api/admin/storage` | 用户存储统计 |
| GET | `/api/admin/users/{user_id}/usage` | 单个用户项目数、存储占用和训练占用 |
| GET | `/api/admin/system/resources` | 首页和管理端使用的 CPU、GPU、显存占用 |
| GET | `/api/admin/algorithms` | 算法许可证登记 |
| POST | `/api/admin/algorithms` | 新增算法登记 |

## 4. 事件通道草案

前端可以使用 WebSocket 或 SSE 订阅项目和任务事件。

```text
GET /api/projects/{project_id}/events
```

事件格式：

```json
{
  "event": "task_progress",
  "project_id": "project-id",
  "task_id": "task-id",
  "task_type": "preview",
  "status": "running",
  "progress": 42,
  "message": "EDGS training",
  "created_at": "2026-04-25T20:00:00+08:00"
}
```

常见事件：

| event | 说明 |
| --- | --- |
| `project_status_changed` | 项目状态变化 |
| `task_queued` | 任务已入队 |
| `task_started` | 任务开始执行 |
| `task_progress` | 任务进度变化 |
| `task_succeeded` | 任务成功 |
| `task_failed` | 任务失败 |
| `artifact_created` | 新产物已生成 |
| `preview_segment_ready` | 新增量 SPZ 片段已生成 |
| `resource_usage` | CPU、GPU、显存占用变化 |
| `capture_suggestion` | 补拍建议或素材质量提示 |

## 5. Worker 任务消息草案

```json
{
  "task_id": "task-id",
  "project_id": "project-id",
  "user_id": "user-id",
  "type": "preview",
  "input_type": "images",
  "raw_uri": "s3://bucket/users/user-id/projects/project-id/raw/images/",
  "output_prefix": "s3://bucket/users/user-id/projects/project-id/preview/",
  "options": {
    "pipeline": "litevggt_edgs",
    "timeout_seconds": 300
  },
  "real_algorithm_required": true
}
```

Worker 返回结果：

```json
{
  "task_id": "task-id",
  "status": "succeeded",
  "artifacts": [
    {
      "kind": "preview_spz",
      "object_uri": "s3://bucket/users/user-id/projects/project-id/preview/preview.spz"
    }
  ],
  "metrics": {
    "duration_seconds": 120,
    "splat_count": 500000
  },
  "suggestions": [
    {
      "type": "missing_view",
      "message": "建议补拍物体背面和右侧视角"
    }
  ]
}
```

如果真实算法环境不可用，Worker 必须返回失败或不可用状态，不能创建假产物：

```json
{
  "task_id": "task-id",
  "status": "failed",
  "error": {
    "code": "ALGORITHM_NOT_CONFIGURED",
    "message": "LiteVGGT weights are not configured"
  }
}
```

## 6. 当前实现同步

截至 2026-04-25，后端实现已采用工程化数据层：

- 数据库模型已覆盖 `users`、`projects`、`media_assets`、`tasks`、`artifacts`、`feedback`、`worker_heartbeats`、`algorithm_registry`。
- 当前已实现的认证接口为 `POST /api/auth/register`、`POST /api/auth/login`、`POST /api/auth/logout`、`GET /api/me`，使用 JWT Bearer token。
- 当前上传接口为 `POST /api/projects/{project_id}/media`，直接写入对象存储并记录真实文件大小；分片上传接口仍是后续项。
- `POST /api/projects/{project_id}/tasks/preview` 只创建 task 并入 Redis 队列，不在 API 请求内运行算法。
- `GET /api/projects/{project_id}/viewer-config` 存在真实 `preview_spz` artifact 时返回 `mode=single`；存在 `preview_spz_segment` artifact 时返回 `mode=progressive` 和 segment URL 列表；否则返回 unavailable。
- `GET /api/algorithms` 为公开算法合规信息；`GET /api/admin/algorithms`、`GET /api/admin/tasks`、`GET /api/admin/workers`、`GET /api/admin/system/resources` 需要管理员角色。
- artifact 下载优先使用 MinIO presigned URL；本地开发后端会发放 1 小时 artifact token 访问 `/api/artifacts/{artifact_id}/file`。

## 7. Runtime Preflight 与预览输入规则

- 管理员接口 `GET /api/admin/runtime/preflight` 返回 Python、CUDA、torch、GPU、算法仓库、权重、命令和 commit 检查结果。
- 图片预览任务创建前要求至少 1 张图片；超过 800 张时任务 options 中记录采样上限。
- 视频预览任务创建前要求已上传视频；video-worker 使用 LingBot-Map 按完整时长采样，少于配置最小帧数时标记失败且不创建 artifact。
- 实时摄像头分片进入 `preview_camera_tasks`，camera-worker 使用 LingBot-Map streaming 处理，成功后创建 `preview_spz_segment` artifact，metadata 记录 `segment_index`、时间窗口、LOD 和估算 splat 数。
- `Task.options.input_frame_policy` 记录 `min_input_frames`、`max_input_frames`、`available_input_frames` 和 `selected_input_frames`。
