# 前端页面规格

## 1. 全局壳层

`frontend/components/AppShell.tsx` 负责：

- 登录态检查和 public path 放行。
- 顶部导航：首页、上传、项目、参数、反馈、关于、管理。
- 系统资源轮询。
- 当前运行/排队任务弹层。
- 管理员额外拉取所有任务。

公共路径：

- `/login`
- `/about`
- `/share/[token]`

## 2. 首页 `/`

首页是工作入口，不是营销页。当前入口：

- 新建项目/上传素材。
- 查看项目列表。
- 查看系统资源和运行中任务入口由全局壳层提供。

## 3. 登录页 `/login`

支持：

- 登录。
- 注册。
- 登录成功后保存 Bearer token 到 localStorage。

## 4. 上传页 `/upload`

页面职责：

- 创建图片或视频项目。
- 编辑项目名和标签。
- 上传图片多文件或单视频。
- 展示素材、缩略图、上传进度、素材统计。
- 启动预览任务。
- 启动精细重建任务。
- 展示当前 task progress 和 Viewer 配置。

上传实现：

- 前端默认走 `api.uploadMedia()` 的分片上传。
- 每个文件先进入 checking，再并发上传缺失分片，最后 complete。
- 图片项目允许多图；视频项目应保持单视频。

预览参数：

- 用户选择 `scene_type=indoor|outdoor`。
- 后端合并系统默认、管理员默认和请求参数。
- 图片和视频均使用 `litevggt_spz`；视频额外应用 speed defaults。

精细重建参数：

- 用户选择 `scene_type=indoor|outdoor` 和 `fine_deblur_mode=motion|defocus|sharp`。
- 管理员可在参数页配置更细的 COLMAP、EAP、DashDeblurGroupGS 参数。

## 5. 项目列表 `/projects`

当前能力：

- 获取当前用户项目。
- 按名称搜索。
- 按状态筛选。
- 批量选择和批量删除。
- 项目卡片显示封面、名称、状态、输入类型、大小、时间和标签。
- 进入项目详情。

注意：

- 卡片封面来自 `preview_image_uri`，通常是第一张缩略图或视频封面。
- 训练中进度通过全局任务弹层和详情页更完整展示。

## 6. 项目详情 `/projects/[id]`

页面职责：

- 拉取项目、Viewer 配置和 artifacts。
- 展示 Spark SPZ 或 PLY fallback viewer。
- 展示素材列表、任务历史和日志。
- 支持重新启动 fine。
- 支持取消运行任务。
- 支持下载 artifact 和预览原始 PLY。
- 支持创建/删除分享链接。
- 支持删除素材和删除项目。

Viewer 加载顺序：

1. 当前 source version 的 final artifact。
2. 当前 source version 的 preview artifact。
3. stale 或 unavailable 状态提示。

## 7. 分享页 `/share/[token]`

公开访问，不需要 token。显示：

- 项目名称。
- 标签、大小、时间。
- Viewer 配置。

分享页不暴露用户私有 token，也不提供项目管理操作。

## 8. 管线参数页 `/pipeline-parameters`

管理员页面。能力：

- 读取 `/api/pipeline-parameters/schema`。
- 读取和保存 `/api/admin/pipeline-parameter-defaults`。
- 支持 `litevggt_spz` 和 `dash_deblur_group_gs`。
- 支持 `indoor/outdoor` 两套 scene defaults。

字段来自后端 schema，不在前端硬编码完整参数列表。

## 9. 管理页 `/admin`

管理员页面。当前展示：

- 当前用户权限检查。
- 系统资源。
- 项目列表和最新任务。
- 用户统计。
- 反馈列表。
- 任务取消。

更多运行时细节通过 API 已提供：`runtimePreflight()`、`workers()`、`adminTasks()`，可继续扩展 UI。

## 10. 用户页 `/profile`

显示：

- 当前用户信息。
- 项目总览。
- 跳转反馈页。

## 11. 反馈页 `/feedback`

支持提交：

- 标题。
- 内容。
- 可选项目 ID。

当前未实现附件上传。

## 12. 关于页 `/about`

公开展示算法列表：

- LiteVGGT。
- Spark SPZ。
- DashDeblurGroupGS Fine。

数据来自 `/api/algorithms`，包括 repo、license、commit/local path、权重路径、enabled 和 license notice。
