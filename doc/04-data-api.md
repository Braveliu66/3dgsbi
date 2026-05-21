# 数据模型与接口

本文档按当前 SQLAlchemy 模型和 `backend/app/main.py` 的 API 实现整理。

## 1. 数据表

### 1.1 users

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | string uuid | 用户 ID |
| username | string | 唯一用户名 |
| email | string nullable | 邮箱 |
| password_hash | string | 密码哈希 |
| role | string | `user` 或 `admin` |
| created_at | datetime | 创建时间 |

### 1.2 projects

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | string uuid | 项目 ID |
| owner_id | string uuid | 所属用户 |
| name | string | 项目名称 |
| input_type | string | `images` 或 `video` |
| status | string | 项目状态 |
| tags | json list | 标签 |
| total_size_bytes | int | 素材总大小 |
| preview_image_uri | text nullable | 卡片封面 |
| share_token | string nullable | 分享 token |
| error_message | text nullable | 最近错误 |
| source_version | int | 素材版本 |
| preview_source_version | int nullable | 当前预览对应素材版本 |
| created_at / updated_at | datetime | 时间戳 |

### 1.3 media_assets

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | string uuid | 素材 ID |
| project_id | string uuid | 项目 ID |
| kind | string | `image` 或 `video` |
| object_uri | text | 原始文件 URI |
| thumbnail_uri | text nullable | 缩略图/封面 URI |
| file_name | string | 原文件名 |
| file_size | int | 文件大小 |
| width / height | int nullable | 尺寸 |
| duration_seconds | int nullable | 视频时长 |
| quality_flags | json | 质量标记 |
| source_version | int | 创建时的素材版本 |
| client_order | int | 前端排序 |
| created_at | datetime | 创建时间 |

### 1.4 upload_sessions

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | string uuid | 上传会话 ID |
| project_id / user_id | string uuid | 归属 |
| file_hash | string | 文件签名哈希 |
| file_name / file_size | string / bigint | 文件信息 |
| chunk_size / total_chunks | bigint / int | 分片配置 |
| content_type | string nullable | MIME |
| kind | string | `image` 或 `video` |
| client_order | int | 前端排序 |
| status | string | `uploading`、`completed`、`failed` |
| object_uri | text nullable | 合并后 URI |
| media_id | string nullable | 完成后的素材 ID |
| error_message | text nullable | 错误 |
| created_at / updated_at | datetime | 时间戳 |

### 1.5 tasks

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | string uuid | 任务 ID |
| project_id | string uuid | 项目 ID |
| type | string | `preview`、`fine`、`lod`、`mesh_export` |
| status | string | `queued`、`running`、`succeeded`、`failed`、`canceled` |
| priority | int | 优先级 |
| progress | int | 0-100 |
| worker_id | string nullable | Worker ID |
| options | json | 任务参数 |
| metrics | json | 运行指标 |
| current_stage | string | 当前阶段 |
| eta_seconds | int nullable | 预计剩余秒数 |
| error_code / error_message | string/text nullable | 错误 |
| logs | json list | 日志摘要 |
| started_at / finished_at | datetime nullable | 时间戳 |

### 1.6 artifacts

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| id | string uuid | 产物 ID |
| project_id / task_id | string uuid | 归属 |
| kind | string | `preview_spz`、`original_ply`、`final_spz`、`final_ply`、`metrics_json` 等 |
| object_uri | text | 存储 URI |
| file_name | string | 文件名 |
| file_size | int | 字节数 |
| checksum | string nullable | sha256 |
| metadata | json | 额外元数据 |
| source_version | int | 对应素材版本 |
| created_at | datetime | 创建时间 |

### 1.7 其他表

| 表 | 用途 |
| --- | --- |
| `stored_objects` / `stored_object_chunks` | 本地对象存储索引和分块 |
| `feedback` | 用户反馈 |
| `worker_heartbeats` | worker 心跳和资源状态 |
| `algorithm_registry` | bundled 算法、许可证、权重和命令登记 |
| `pipeline_parameter_defaults` | 管线 scene 默认参数 |
| `task_events` | SSE 事件持久化 |

## 2. API

### 2.1 健康与认证

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| POST | `/api/auth/register` | 注册 |
| POST | `/api/auth/login` | 登录，返回 Bearer token |
| POST | `/api/auth/logout` | 前端清 token |
| GET | `/api/me` | 当前用户 |

### 2.2 项目

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/projects/summary` | 当前用户项目摘要 |
| GET | `/api/projects` | 当前用户项目列表 |
| POST | `/api/projects` | 创建项目，`input_type=images|video` |
| GET | `/api/projects/{project_id}` | 项目详情 |
| DELETE | `/api/projects/{project_id}` | 删除项目 |
| POST | `/api/projects/bulk-delete` | 批量删除 |

当前没有 PATCH 项目更新接口。

### 2.3 上传与素材

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/projects/{project_id}/media` | 直接上传单文件 |
| POST | `/api/projects/{project_id}/uploads/check` | 创建/检查分片上传会话 |
| PUT | `/api/uploads/{upload_id}/chunks/{chunk_index}` | multipart 分片上传 |
| PUT | `/api/uploads/{upload_id}/chunks/{chunk_index}/raw` | octet-stream 原始分片上传 |
| POST | `/api/uploads/{upload_id}/complete` | 合并分片并创建 media |
| GET | `/api/projects/{project_id}/media` | 素材列表 |
| GET | `/api/projects/{project_id}/media/stats` | 素材统计 |
| DELETE | `/api/projects/{project_id}/media/{media_id}` | 删除素材 |
| GET | `/api/media/{media_id}/thumbnail` | 缩略图 |
| GET | `/api/media/{media_id}/file` | 原始素材文件 |

### 2.4 任务

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/projects/{project_id}/tasks/preview` | 创建预览任务 |
| POST | `/api/projects/{project_id}/tasks/fine` | 创建精细重建任务 |
| GET | `/api/tasks/{task_id}` | 查询任务 |
| POST | `/api/tasks/{task_id}/cancel` | 取消任务 |
| GET | `/api/tasks/{task_id}/log` | 下载任务日志文本 |

当前没有 Mesh export 任务创建接口。

### 2.5 产物、Viewer 与分享

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/projects/{project_id}/artifacts` | 项目产物 |
| GET | `/api/artifacts/{artifact_id}/download-url` | 下载 URL |
| GET | `/api/artifacts/{artifact_id}/original-ply/download-url` | 预览原始 PLY 下载 URL |
| GET | `/api/artifacts/{artifact_id}/file` | token 下载 |
| GET | `/api/artifacts/{artifact_id}/original-ply/file` | token 下载原始 PLY |
| GET | `/api/projects/{project_id}/viewer-config` | Viewer 配置 |
| POST | `/api/projects/{project_id}/share` | 创建分享 |
| DELETE | `/api/projects/{project_id}/share` | 删除分享 |
| GET | `/api/shared-projects/{share_token}` | 公开分享项目 |

### 2.6 反馈、算法和参数

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/feedback` | 提交反馈 |
| GET | `/api/algorithms` | 公开算法列表 |
| GET | `/api/pipeline-parameters/schema` | 参数 schema，公开读取 |
| GET | `/api/admin/pipeline-parameter-defaults` | 管线默认参数 |
| PUT | `/api/admin/pipeline-parameter-defaults/{pipeline}/{scene_type}` | 保存默认参数 |

### 2.7 管理

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/system/resources` | 当前用户可读资源摘要 |
| GET | `/api/admin/system/resources` | 管理资源摘要 |
| GET | `/api/admin/runtime/preflight` | 运行时预检 |
| GET | `/api/admin/tasks` | 所有任务 |
| GET | `/api/admin/projects` | 所有项目统计 |
| GET | `/api/admin/users` | 用户统计 |
| GET | `/api/admin/feedback` | 反馈列表 |
| GET | `/api/admin/workers` | Worker 心跳 |
| GET | `/api/admin/algorithms` | 管理算法登记 |

## 3. 任务 options

### 3.1 Preview

核心字段：

```json
{
  "pipeline": "litevggt_spz",
  "preview_pipeline": "litevggt_spz",
  "input_type": "images",
  "scene_type": "indoor",
  "preview_scene_profile": "indoor_full",
  "litevggt_target_size": 420,
  "preview_max_points": 3200000
}
```

视频预览会额外写入 LiteVGGT video speed defaults，例如 `preview_video_fps`、`preview_video_max_frames`、`litevggt_max_input_frames`、`litevggt_inference_mode`。

### 3.2 Fine

核心字段：

```json
{
  "fine_pipeline": "dash_deblur_group_gs",
  "input_type": "images",
  "scene_type": "indoor",
  "fine_scene_type": "indoor",
  "fine_scene_profile": "indoor_full",
  "fine_sfm_backend": "colmap_global",
  "fine_eap_enabled": true,
  "fine_deblur_enabled": true,
  "fine_deblur_mode": "motion",
  "fine_gsplat_enabled": true,
  "fine_spz_enabled": true
}
```

Fine metrics 包含输入、blur、COLMAP、EAP、训练、SPZ、bbox、source commits 和最终点数。

## 4. 事件通道

```text
GET /api/projects/{project_id}/events?token=<bearer-token>
```

事件示例：

```json
{
  "event": "task_progress",
  "project_id": "project-id",
  "task_id": "task-id",
  "status": "running",
  "progress": 42,
  "current_stage": "fine_training",
  "eta_seconds": 3600
}
```

## 5. Worker 消息

Redis 队列当前只存 task id。Worker 根据 task id 从数据库读取项目、素材和 options。

```text
preview_tasks: task_id
fine_tasks: task_id
```

Worker 成功后写入 artifacts、task metrics、task events 和 project status；失败后写入错误码、错误信息和日志 artifact。
