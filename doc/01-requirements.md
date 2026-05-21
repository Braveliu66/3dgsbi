# 系统需求说明

## 1. 项目定位

本系统是面向非商业研究、毕业设计和实验室内部使用的 3D Gaussian Splatting 重建平台。当前代码提供从图片或单视频上传，到 LiteVGGT 快速预览、DashDeblurGroupGS 精细重建、Web 查看、下载、分享、反馈和后台管理的单机闭环。

用户不需要选择底层算法。前端只暴露项目类型、室内/室外场景、预览、精细重建和少量管理员参数；后端负责把这些意图解析为固定管线。

## 2. 用户角色

| 角色 | 主要目标 | 当前操作 |
| --- | --- | --- |
| 普通用户 | 上传素材并查看结果 | 注册/登录、创建项目、上传图片或单视频、启动预览、启动精细重建、下载产物、分享项目、提交反馈 |
| 管理员 | 管理资源和任务 | 查看系统资源、任务、Worker、用户、项目、反馈、运行时预检和管线默认参数 |
| 研究维护者 | 管理算法边界 | 维护 bundled 算法、权重路径、许可证说明、训练参数和运行时检查 |

## 3. 当前功能需求

### 3.1 项目管理

- 支持创建、列表、详情、单个删除和批量删除项目。
- 项目输入类型为 `images` 或 `video`；实时摄像头未实现。
- 项目记录名称、标签、状态、大小、封面缩略图、错误信息、source version、任务和产物。
- 上传素材变更会递增 `source_version`，旧预览产物会在 Viewer 配置中标记 stale。
- 项目详情优先展示同一 `source_version` 下的最终模型；没有最终模型时回退到当前预览模型。

### 3.2 素材上传

- 支持直接上传接口 `POST /api/projects/{project_id}/media`。
- 前端默认使用分片上传：`uploads/check`、`chunks/{index}/raw`、`complete`。
- 分片上传支持续传、已完成秒传、64MB 分片上限、文件签名和 `client_order`。
- 图片项目只能上传图片；视频项目只能上传视频，精细重建和预览均要求单视频项目只有一个视频文件。
- 上传后生成缩略图或视频封面，记录尺寸、时长、大小和质量标记。

### 3.3 极速预览

- 当前预览管线为 `litevggt_spz`。
- 图片预览：归一化图片后调用 LiteVGGT，生成点云 PLY、中间调试文件和 Spark SPZ。
- 视频预览：单视频先用 ffmpeg 抽帧和筛帧，再使用 LiteVGGT speed defaults 生成 Spark SPZ。
- 预览任务进入 Redis `preview_tasks` 队列，由 `worker-preview` 执行。
- 缺少 GPU、权重、Spark 转码器或非空产物校验失败时任务失败，不创建成功模型 artifact。

### 3.4 精细重建

- 图片项目至少 3 张图片即可发起精细重建。
- 视频项目支持单视频精细重建：先抽帧、过滤，再进入同一 fine pipeline。
- 默认精细管线为 `dash_deblur_group_gs`；旧 `colmap_sparse` 仅为别名。
- 默认 SfM 后端为 `colmap_global`，使用 COLMAP 4.x `global_mapper`；`gcolmap` 是别名，`colmap_cli` 和 `pycolmap` 可显式选择。
- 默认启用 EAP 点云增强、DashDeblurGroupGS 训练、final PLY 远端噪声过滤、`final_web.spz` 转码和 `final_viewer_meta.json`。
- 产物包括 `final.ply`、`final_web.spz`、`metrics.json`、`final_viewer_meta.json` 和任务日志。

### 3.5 分享、下载与反馈

- 支持创建和删除项目分享链接，公开分享页只返回必要项目信息和 Viewer 配置。
- 支持 artifact 下载 URL；本地存储使用 1 小时 token 访问 `/api/artifacts/{artifact_id}/file`。
- 支持预览 artifact 的原始 PLY 下载 URL。
- 用户可以提交标题、内容和可选项目关联的反馈；管理员可查看反馈列表。

### 3.6 管理与参数

- 管理端可查看系统资源、运行时预检、任务日志、项目、用户、反馈和 Worker 心跳。
- 管理员可编辑 `litevggt_spz` 和 `dash_deblur_group_gs` 在 `indoor/outdoor` 下的默认参数。
- 保存的参数带 preset marker；过期 marker 会被忽略，避免旧默认污染当前管线。

## 4. 规划或未实现能力

- 实时摄像头采集和实时增量预览未实现。
- Mesh 导出任务接口未实现；当前只提供已有 artifact 下载。
- `.rad` LOD 生成未接入真实转换器，fine 任务默认不生成 `final_lod.rad`。
- 项目 PATCH 更新接口未实现；当前项目创建后主要通过任务和上传流程变化状态。
- 多机 GPU 调度仍是规划能力；当前 Compose 是单机多服务模型。

## 5. 非功能需求

| 类别 | 要求 |
| --- | --- |
| 真实性 | 算法成功必须来自真实非空产物；失败必须保留错误码、日志和阶段 |
| 性能 | 预览优先快速可交互，精细重建优先质量和诊断信息 |
| 并发 | API 不直接执行长任务；preview/fine 分别进入 Redis 队列 |
| 可观测 | 每个任务记录阶段、进度、ETA、worker、日志、metrics 和 artifact |
| 存储 | 用户和项目路径隔离；大文件分片；下载链接过期 |
| 安全 | Bearer token 认证；普通用户只能访问自己的项目；分享页只暴露公开信息 |

## 6. 验收标准

1. 图片和单视频预览能完成上传、入队、真实算法执行、产物上传和 Viewer 加载。
2. 图片和单视频精细重建能生成真实非空 `final.ply`、`final_web.spz`、`metrics.json`。
3. 算法环境缺失、权重缺失、GPU 不可用或产物无效时任务失败，artifact 表不新增成功模型。
4. 分片上传支持续传、完整性校验和完成后创建真实 media。
5. Viewer 能识别 stale preview，并优先使用同源版本 final artifact。
6. 管理端能显示运行时预检、Worker 心跳、任务日志和用户/项目统计。
7. 文档、配置和源码使用 UTF-8；PowerShell 显示乱码时以 `-Encoding UTF8` 读取。
