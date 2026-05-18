# 3DGS 重建系统文档索引

本文档目录由根目录 `system.md` v2.1 拆解而来，目的是让后续新窗口编写代码时能快速获得稳定上下文。

## 使用方式

如果新建窗口后需要让 Codex 编写代码，可以先让它读取：

```text
请先阅读 Q:\3dgsbi\doc\README.md，并按其中的文档顺序理解项目，然后根据我的具体需求编写代码。
```

建议阅读顺序：

1. `01-requirements.md`：系统目标、用户角色、功能需求、非功能需求、验收标准。
2. `02-architecture.md`：整体架构、模块职责、算法适配层、部署结构。
3. `03-workflows-state.md`：上传、预览、精细重建、导出、状态机与任务规则。
4. `04-data-api.md`：数据模型、对象存储路径、后端 API 与事件通道草案。
5. `05-implementation-plan.md`：推荐开发顺序、里程碑、验证方法和风险项。
6. `06-frontend-pages.md`：首页、上传页、项目管理页和详情页的页面规格。
7. `07-backend-constraints.md`：真实算法接入、高并发、多 GPU、存储、权限和禁止占位实现的约束。
8. `08-progress.md`：当前工程实现进度、已验证项、未接入项和下一步。
9. `09-deployment.md`：Docker/WSL 启动方式、环境变量、迁移、种子账号和运行验证。

## 重要约束

- 系统定位为非商业研究、毕业设计和实验室内部使用。
- 前端不暴露算法选择，用户只选择操作意图，后端按规则自动选择管线。
- 代码实现算法流程未接入真实实现前，必须明确返回不可用或失败。
- 后端算法能力不能用占位函数、假文件、固定假结果冒充通过；未接入真实算法时必须返回明确的不可用状态。
- 所有第三方算法源码都先放入项目根目录 `repo-cache/<repo-name>`；Docker 构建必须优先复用本地缓存，缺失时先尝试国内镜像源再回退官方源。
- 源码、配置和文档统一使用 UTF-8 编码；Windows PowerShell 若显示乱码，应先切换终端编码或使用 UTF-8 读取。

## 2026-05-18 Image Fine COLMAP/DashDeblurGroup Notes

- `10-fine-pipeline.md` is the source of truth for the current fine reconstruction implementation.
- Image fine `dash_deblur_group_gs` defaults to PyCOLMAP initialization, writes a COLMAP-compatible `images/` and `sparse/0` scene, then runs the embedded trainer from `worker/trainer/dash_deblur_group_gs` mounted at `/opt/dash_deblur_group_gs` in local Docker Compose. `DASH_DEBLUR_GROUP_REPO` is only for explicit compatible trainer overrides.
- `fine_sfm_backend=pycolmap` is the implementation default; `fine_sfm_backend=colmap` is accepted as a PyCOLMAP alias and `colmap_cli` remains available for explicit CLI runs.
- `colmap_sparse` is only a legacy fine pipeline alias.
- Preview LiteVGGT keeps separate speed-oriented defaults with fewer frames and points.
- Docker and task-specific downloads must prefer `https://hf-mirror.com`.
- `birth_iter` and `protect_new_points_iters` were removed from the fused trainer and must not appear in generated configs.


