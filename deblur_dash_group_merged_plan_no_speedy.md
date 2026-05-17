# 去模糊 3D Gaussian 重建系统融合方案（Deblur + Dash + Group，无 Speedy-Splat）

> 面向开发者与 Codex 的工程级实现文档  
> 基础仓库：`benhenryL/Deblurring-3D-Gaussian-Splatting`  
> 融合模块：`DashGaussian` + `3DGS-with-Group-Training`  
> 删除模块：`Speedy-Splat` 训练与推理接入  
> 推荐项目名：`DashDeblurGroupGS`

---

## 0. 最终结论

本版采用 **三仓库融合**：

```text
Deblurring-3D-Gaussian-Splatting：主干仓库，负责模糊物理建模和 sharp canonical Gaussian 学习
DashGaussian：只移植分辨率调度与 Gaussian 增长调度
3DGS-with-Group-Training：只作为非破坏式训练减负模块
Speedy-Splat：删除，不进入训练，也不作为默认推理路径
```

删除 Speedy-Splat 的原因：

```text
1. Speedy-Splat 的 pruning / sparse primitive 策略与 Deblur 的 blurry-supervision 目标冲突。
2. Deblur 训练阶段需要通过 GTnet 生成多个虚拟视图 / 模糊变换，不能提前裁剪或稀疏化关键 Gaussian。
3. Speedy-Splat 的推理加速收益依赖特定 renderer / pruning 策略，接入成本较高，且对最终去模糊质量帮助不明显。
4. 第一阶段目标应是稳定提升训练速度与控制 Gaussian 数量，而不是引入 renderer 级 CUDA kernel 冲突。
```

因此，本方案的核心原则是：

```text
模糊物理由 Deblurring-3DGS 保持不变；
训练调度由 DashGaussian 负责，但不得改写 blur-space 假设；
训练减负由 Group Training 负责，但只做临时 cache，不永久删除 Gaussian；
所有 pruning、renderer 替换、部署加速都不进入第一版融合。
```

### 本仓库当前对接方式

当前平台实现不把 COLMAP 和训练代码混在一起：

```text
backend/app/fine/colmap_cli.py
  只负责现有 COLMAP CLI / pycolmap SfM，输出 COLMAP scene：
  images/、sparse/0/、database.db

backend/app/fine/dash_deblur_group.py
  只负责 DashDeblurGroupGS 训练配置生成、训练进程启动、final PLY 定位和 SPZ 转换

backend/app/fine/runner.py
  编排：输入预处理 -> 现有 COLMAP -> DashDeblurGroupGS -> final.ply/final_web.spz/metrics.json
```

运行时默认从 `repo-cache/DashDeblurGroupGS/train.py` 启动训练；也可以通过环境变量
`DASH_DEBLUR_GROUP_REPO` 或任务参数 `fine_trainer_repo` 指向外部训练仓库。旧的
`colmap_sparse` fine pipeline 名称仅作为兼容别名，新的默认名称是
`dash_deblur_group_gs`。

依赖策略：

```text
所有算法共用当前 worker 的 PyTorch/CUDA/Node/COLMAP 环境；
LiteVGGT、LingBot、Spark 的依赖在 worker 镜像中安装；
DashDeblurGroupGS 的 pure Python 依赖也在 worker 镜像中安装；
DashDeblurGroupGS 的 diff_gaussian_rasterization/simple_knn CUDA 扩展从训练仓库 submodules 安装到同一个 Python 环境；
worker/Dockerfile 通过 `DASH_DEBLUR_GROUP_REPO_URL` / `DASH_DEBLUR_GROUP_REPO_COMMIT` 完成训练仓库 checkout、submodule 初始化和扩展 wheel 构建；
worker/Dockerfile 同时从源码构建新版 COLMAP，并要求 `global_mapper`、`hierarchical_mapper`、`model_clusterer`、`model_splitter` 在镜像构建期可用；
本文及后续落地说明中的 FastGS 分块训练语义，在本平台内统一替换为 DashDeblurGroupGS：先全局 COLMAP sparse/0，再基于统一坐标做 DashDeblurGroupGS 分块/调度；
不额外引入 Speedy-Splat、FastGS renderer 替换或第二套 torch/CUDA。
```

---

## 1. 总体架构

```text
Raw blurry images
    │
    ▼
COLMAP preprocessing
    │
    ├── images/
    ├── sparse/
    └── database.db
    │
    ▼
train.py / train_deblur_dash_group.py
    │
    ├── Deblurring-3DGS backbone
    │     ├── GaussianModel + GTnet
    │     ├── motion blur multi-moment render
    │     ├── defocus blur scale / rotation transform
    │     ├── point addition
    │     └── sharp canonical inference path
    │
    ├── DashGaussian scheduler
    │     ├── render_scale / render_size
    │     ├── low-resolution early training
    │     └── densify_rate / Gaussian growth budgeting
    │
    └── Group Training
          ├── opacity-weighted active group
          ├── temporary Gaussian caching
          ├── periodic merge-back
          └── disabled around point addition
    │
    ▼
Canonical Gaussian checkpoint
    │
    ├── standard render.py for sharp validation
    └── metrics.py for PSNR / SSIM / LPIPS
```

---

## 2. 仓库职责边界

### 2.1 Deblurring-3DGS：主干，不破坏

保留全部核心机制：

```text
GTnet
motion blur 多 moment 虚拟视图渲染
defocus blur 的 scale / rotation delta
point addition
Deblur 原始 densify / prune 语义
sharp inference with GTnet off
```

训练路径保持：

```text
canonical Gaussian
    → GTnet 根据 view / position / scale / rotation 预测 blur transform
    → render transformed / multi-moment Gaussian
    → blurry render 与 blurry GT 监督
    → 反传更新 Gaussian + GTnet
```

推理路径保持：

```text
canonical Gaussian
    → deblur = 0
    → GTnet 不参与
    → sharp render
```

推荐统一 `deblur` 模式映射：

```text
0 = sharp / no deblur / inference
1 = camera motion blur
2 = defocus blur
```

如果实际基础仓库的 CLI 使用 `deblur + use_pos` 组合而不是 `0/1/2` 枚举，应以仓库代码为准，但配置文件中必须统一写清楚映射，避免 motion / defocus 配置混淆。

---

### 2.2 DashGaussian：只做 scheduler

只移植三部分：

```text
TrainingScheduler
render_scale / render_size
get_densify_rate() + update_momentum()
```

第一版不要移植：

```text
SparseAdam
DashGaussian exposure compensation
DashGaussian antialiasing branch
3dgs-accel rasterizer branch
Speedy-Splat / FastGS 相关 renderer 优化
```

Dash 的职责是：

```text
当前 iteration 用什么分辨率训练；
当前 Gaussian 数量下允许增长多少；
通过 momentum budgeting 避免 Gaussian 数量爆炸。
```

Dash 不应该：

```text
改变 GTnet 输出意义；
改变 Deblur render 的 motion / defocus 分支；
提前做 aggressive pruning；
替换 Deblur 的 rasterizer。
```

---

### 2.3 Group Training：只做非破坏式训练减负

Group Training 的职责：

```text
每隔 N iter：
  1. 先把上次 cached Gaussians merge 回完整模型；
  2. 根据 opacity-weighted 策略选择 active Gaussians；
  3. 临时 cache 非 active Gaussians；
  4. 当前 interval 只训练 active subset；
  5. 下一次 grouping 或关键事件前 merge 回来。
```

必须遵守：

```text
不永久剪枝；
不在训练早期启动；
不在 pts_iter 前后启动；
densify / add_points / save checkpoint 前必须 merge cache；
motion blur 场景 UTR 更高，避免 canonical geometry 欠训练。
```

---

## 3. 推荐目录结构

```text
DashDeblurGroupGS/
├── README.md
├── environment.yml
├── arguments/
│   └── __init__.py
├── gaussian_renderer/
│   └── __init__.py
├── scene/
│   ├── gaussian_model.py
│   ├── blur_kernel.py
│   └── cameras.py
├── utils/
│   ├── loss_utils.py
│   ├── image_utils.py
│   ├── general_utils.py
│   └── schedule_utils.py          # 从 DashGaussian 移植并做 Deblur-safe 包装
├── gaussians_grouping/            # Group Training submodule
│   └── ...
├── configs/
│   ├── indoor_motion_dash_group.txt
│   ├── indoor_defocus_dash_group.txt
│   ├── outdoor_motion_dash_group.txt
│   └── outdoor_defocus_dash_group.txt
├── scripts/
│   ├── colmap_indoor.sh
│   ├── colmap_outdoor.sh
│   ├── train_indoor_motion.sh
│   ├── train_indoor_defocus.sh
│   ├── train_outdoor_motion.sh
│   └── train_outdoor_defocus.sh
├── train.py                       # 或 train_deblur_dash_group.py
├── render.py                      # 标准 sharp render，不接 Speedy
└── metrics.py
```

删除这些文件 / 章节：

```text
render_speedy.py
export_speedy.py
submodules/diff-gaussian-rasterization-speedy/
Speedy-Splat renderer-only evaluation
Speedy pruning / soft pruning / hard pruning
Speedy FPS benchmark as default path
```

---

## 4. 核心冲突与解决方案

### 4.1 Dash 分辨率调度 vs Deblur GTnet

问题：

```text
Dash 早期低分辨率训练会改变像素尺度；
Deblur 的 GTnet 学习 blur transform，如果过早改变 render resolution，可能导致模糊核尺度不稳定。
```

推荐保守策略：

```text
前 deblur_warmup_iter 步 render_scale 固定为 1；
GTnet 和 canonical geometry 初步稳定后再启动 Dash resolution scheduling；
不要在第一版里动态缩放 lambda_s / lambda_p，避免引入额外不确定性。
```

配置：

```ini
dash_enable = True
dash_start_iter = 3000        # indoor
# outdoor 推荐 5000~6500
```

训练逻辑：

```python
if scheduler is not None and iteration >= opt.dash_start_iter:
    render_scale = scheduler.get_res_scale(iteration)
else:
    render_scale = 1
```

---

### 4.2 Dash densify 接口 vs Deblur densify 接口

问题：

```text
DashGaussian 需要 prune_and_densify() 返回 momentum_add；
Deblurring-3DGS 原始 densify_and_prune() 不返回 momentum_add，且有 deblur-specific 参数。
```

解决：新增 Deblur-safe 适配方法，不直接覆盖原函数。

```python
def prune_and_densify_deblur_safe(
    self,
    max_grad,
    min_opacity,
    extent,
    max_screen_size,
    densify_with_depth=False,
    prune_range=None,
    densify_rate=1.0,
    iteration=None,
    protect_new_points_iters=1500,
):
    """
    Deblur + Dash 兼容版本：
    1. 保留 Deblur 原始 densify / prune 语义；
    2. 用 densify_rate 限制本轮新增 Gaussian 数量；
    3. 返回 momentum_add 给 Dash scheduler；
    4. 保护 add_points 后的新点。
    """
    n_before = self.get_xyz.shape[0]

    # 可选：设置 current_iteration，供 birth_iter 使用
    if iteration is not None:
        self.current_iteration = iteration

    # 第一版实现可以先调用原 Deblur 方法，保证正确性
    self.densify_and_prune(
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        densify_with_depth,
        prune_range,
    )

    n_after = self.get_xyz.shape[0]
    raw_add = n_after - n_before

    # momentum_add 表示净增长，用于 scheduler 的预算反馈
    momentum_add = int(raw_add)
    return momentum_add
```

更严格的第二阶段实现：

```text
把 Dash 的 top-k clone/split 合并进 GaussianModel，
用 densify_rate 限制 clone/split 的候选数量，
而不是在事后乘 densify_rate。
```

第一阶段建议先用 wrapper 保守接入，等 baseline 跑通后再做 top-k densify 精细化。

---

### 4.3 Deblur motion blur 的 list 型 visibility_filter

Deblur motion blur 分支可能返回多次虚拟视图渲染结果：

```text
visibility_filter: list[Tensor]
radii: list[Tensor]
viewspace_point_tensor: 可能对应多 moment render
```

必须兼容：

```python
if isinstance(visibility_filter, list):
    denom = 1.0 / len(visibility_filter)
    for vf, rd in zip(visibility_filter, radii):
        gaussians.max_radii2D[vf] = torch.max(
            gaussians.max_radii2D[vf],
            rd[vf],
        )
else:
    denom = 1.0
    gaussians.max_radii2D[visibility_filter] = torch.max(
        gaussians.max_radii2D[visibility_filter],
        radii[visibility_filter],
    )

gaussians.add_densification_stats(
    viewspace_point_tensor,
    visibility_filter,
    denom,
)
```

---

### 4.4 Group Training cache 与 densify / add_points 冲突

凡是 Gaussian 数量或 optimizer tensor 发生变化，都必须让 Group cache 失效或先 merge：

```text
densify_and_prune 后：point_caching = None
prune_and_densify_deblur_safe 后：point_caching = None
add_points 前：如果 point_caching 不为空，先 merge 回来
add_points 后：point_caching = None
保存 checkpoint 前：如果 point_caching 不为空，先 merge 回来
训练结束前：如果 point_caching 不为空，先 merge 回来
```

推荐辅助函数：

```python
def merge_group_cache_if_needed(gaussians, point_caching):
    if point_caching is not None:
        gaussians.densification_postfix(**point_caching)
        point_caching = None
    return point_caching
```

---

### 4.5 point addition 保护

Deblurring-3DGS 的 `add_points()` 不删除。它用于缓解 blurry SfM 初始点云稀疏问题。

推荐策略：

```text
pts_iter 前后 800~1500 iter 禁止 Group Training；
pts_iter 后 1500~2500 iter 保护新点不被 aggressive prune；
outdoor 保护窗口更长；
add_points 后重置 Group cache。
```

配置：

```ini
grouping_freeze_around_pts = 1000        # indoor
grouping_freeze_around_pts = 1500        # outdoor
protect_new_points_iters = 1500          # indoor
protect_new_points_iters = 2500          # outdoor
```

---

## 5. 文件级实现方案

### 5.1 `arguments/__init__.py`

保留 Deblur 原参数，并新增：

```python
# DashGaussian scheduling
self.dash_enable = False
self.resolution_mode = "const"       # const | freq
self.densify_mode = "free"           # free | freq
self.max_n_gaussian = -1
self.dash_start_iter = 3000
self.dash_max_reso_scale = 4
self.dash_max_densify_rate_per_step = 0.10
self.dash_start_significance_factor = 4

# Group Training safety
self.grouping_enable = True
self.grouping_from_iter = 4500
self.grouping_until_iter = 20000
self.grouping_interval = 600
self.grouping_freeze_around_pts = 1000
self.protect_new_points_iters = 1500

# Safety flags
self.speedy_train_pruning = False      # 保留为显式禁止项，可不暴露到 CLI
self.use_speedy_renderer = False       # 保留为显式禁止项，可不暴露到 CLI
```

如果使用 Group Training 官方 `GroupingParams(parser)`，需要确保 CLI 名称与配置文件一致：

```ini
Grouping = True
UTR = 0.75
grouping_method = Opacity-weighted
grouping_from_iter = 4500
grouping_until_iter = 20000
grouping_interval = 600
```

---

### 5.2 `gaussian_renderer/__init__.py`

给 Deblur 原始 `render()` 增加 `render_size=None`。

```python
def render(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    scaling_modifier=1.0,
    deblur=0,
    use_pos=False,
    lambda_s=0.01,
    lambda_p=0.01,
    max_clamp=1.1,
    render_size=None,
):
    if render_size is None:
        image_height = int(viewpoint_camera.image_height)
        image_width = int(viewpoint_camera.image_width)
    else:
        image_height = int(render_size[0])
        image_width = int(render_size[1])

    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
```

注意：

```text
只新增 render_size；
不要改 Deblur motion / defocus 分支；
不要在 renderer 内接入 Speedy-Splat sparse pixel / primitive pruning。
```

---

### 5.3 `utils/schedule_utils.py`

从 DashGaussian 复制 `TrainingScheduler`，再加 Deblur-safe 包装：

```python
class DeblurDashScheduler(TrainingScheduler):
    def __init__(self, opt, pipe, gaussians, original_images):
        super().__init__(opt, pipe, gaussians, original_images)
        self.max_reso_scale = getattr(opt, "dash_max_reso_scale", 4)
        self.max_densify_rate_per_step = getattr(
            opt,
            "dash_max_densify_rate_per_step",
            0.10,
        )

    def enabled_for_resolution(self, iteration, start_iter):
        return iteration >= start_iter
```

推荐默认值：

```text
indoor:  dash_max_reso_scale = 4
outdoor: dash_max_reso_scale = 4，稳定后可试 6
不建议第一版使用 8
```

---

### 5.4 `scene/gaussian_model.py`

必须保留：

```python
def create_GTnet(...):
    ...

# optimizer 中保留 GTnet 参数组
{'params': self.GTnet.parameters(), 'lr': training_args.gtnet_lr, 'name': 'GTnet'}

# prune / cat optimizer tensor 时跳过 GTnet
if group['name'] == 'GTnet':
    continue
```

新增可选 birth_iter：

```python
self.birth_iter = torch.empty(0, device="cuda", dtype=torch.long)
```

在 `create_from_pcd()` 后：

```python
self.birth_iter = torch.zeros(
    (self.get_xyz.shape[0],),
    device="cuda",
    dtype=torch.long,
)
```

在 `densification_postfix()` 后：

```python
if hasattr(self, "birth_iter"):
    current_iter = getattr(self, "current_iteration", 0)
    new_birth = torch.full(
        (new_xyz.shape[0],),
        fill_value=current_iter,
        device="cuda",
        dtype=torch.long,
    )
    self.birth_iter = torch.cat([self.birth_iter, new_birth], dim=0)
```

在 `prune_points()` 后同步：

```python
if hasattr(self, "birth_iter"):
    self.birth_iter = self.birth_iter[valid_points_mask]
```

第一版可先不强依赖 birth_iter，只通过 `grouping_freeze_around_pts` 和保守 prune 阈值控制风险。

---

### 5.5 `train.py`

#### import

```python
import torch.nn.functional as F
from utils.schedule_utils import DeblurDashScheduler
from gaussians_grouping import gaussians_grouping_and_caching, GroupingParams
```

#### 初始化顺序

```python
gaussians = GaussianModel(dataset.sh_degree, deblur)
scene = Scene(dataset, gaussians)

gaussians.create_GTnet(
    hidden=opt.hidden,
    width=opt.width,
    pos_delta=opt.use_pos,
    num_moments=opt.num_moments,
)

gaussians.training_setup(opt)

scheduler = None
if getattr(opt, "dash_enable", False):
    scheduler = DeblurDashScheduler(
        opt,
        pipe,
        gaussians,
        [cam.original_image for cam in scene.getTrainCameras()],
    )

point_caching = None
```

必须先 `create_GTnet()` 再 `training_setup()`。

---

## 6. 主训练循环关键逻辑

```python
for iteration in range(first_iter, opt.iterations + 1):
    gaussians.current_iteration = iteration
    gaussians.update_learning_rate(iteration)

    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()

    # 1. sample camera
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

    # 2. Dash resolution scheduling
    if scheduler is not None and iteration >= opt.dash_start_iter:
        render_scale = scheduler.get_res_scale(iteration)
    else:
        render_scale = 1

    gt_image = viewpoint_cam.original_image.cuda()
    if render_scale > 1:
        gt_image = F.interpolate(
            gt_image[None],
            scale_factor=1.0 / render_scale,
            mode="bilinear",
            recompute_scale_factor=True,
            antialias=True,
        )[0]
    render_size = gt_image.shape[-2:]

    # 3. Group Training before render
    use_group_now = False
    if group_training is not None and group_training.Grouping:
        in_group_window = (
            iteration >= opt.grouping_from_iter
            and iteration <= opt.grouping_until_iter
        )
        away_from_pts = abs(iteration - opt.pts_iter) > opt.grouping_freeze_around_pts
        on_group_iter = iteration in group_training.grouping_iteration
        use_group_now = in_group_window and away_from_pts and on_group_iter

    if use_group_now:
        point_caching = gaussians_grouping_and_caching(
            iteration,
            gaussians,
            group_training,
            _points_caching=point_caching,
        )

    # 4. Deblur render, only add render_size
    render_pkg = render(
        viewpoint_cam,
        gaussians,
        pipe,
        background,
        deblur=deblur,
        use_pos=opt.use_pos,
        lambda_s=opt.lambda_s,
        lambda_p=opt.lambda_p,
        max_clamp=opt.max_clamp,
        render_size=render_size,
    )

    image = render_pkg["render"]
    viewspace_point_tensor = render_pkg["viewspace_points"]
    visibility_filter = render_pkg["visibility_filter"]
    radii = render_pkg["radii"]

    # 5. loss
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    loss.backward()

    with torch.no_grad():
        # 6. densification stats
        if iteration < opt.densify_until_iter:
            if isinstance(visibility_filter, list):
                denom = 1.0 / len(visibility_filter)
                for vf, rd in zip(visibility_filter, radii):
                    gaussians.max_radii2D[vf] = torch.max(
                        gaussians.max_radii2D[vf],
                        rd[vf],
                    )
            else:
                denom = 1.0
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

            gaussians.add_densification_stats(
                viewspace_point_tensor,
                visibility_filter,
                denom,
            )

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                # Group cache must be merged before changing tensor size
                point_caching = merge_group_cache_if_needed(gaussians, point_caching)

                if scheduler is not None and opt.densify_mode == "freq" and iteration >= opt.dash_start_iter:
                    densify_rate = scheduler.get_densify_rate(
                        iteration,
                        gaussians.get_xyz.shape[0],
                        render_scale,
                    )
                    momentum_add = gaussians.prune_and_densify_deblur_safe(
                        opt.densify_grad_threshold,
                        opt.densify_prune_threshold,
                        scene.cameras_extent,
                        None,
                        densify_with_depth=opt.densify_with_depth,
                        prune_range=opt.prune_range,
                        densify_rate=densify_rate,
                        iteration=iteration,
                        protect_new_points_iters=opt.protect_new_points_iters,
                    )
                    scheduler.update_momentum(momentum_add)
                else:
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.densify_prune_threshold,
                        scene.cameras_extent,
                        None,
                        opt.densify_with_depth,
                        opt.prune_range,
                    )

                point_caching = None

        # 7. Deblur point addition
        if iteration == opt.pts_iter:
            point_caching = merge_group_cache_if_needed(gaussians, point_caching)
            gaussians.current_iteration = iteration
            gaussians.add_points(
                training_args=opt,
                dist=opt.pts_dist,
                N=opt.pts_N_intpl,
                num_pts=opt.pts_N_pts,
                bound=opt.pts_add_bound,
            )
            point_caching = None

        # 8. save / checkpoint：先 merge cache
        if iteration in saving_iterations or iteration in checkpoint_iterations:
            point_caching = merge_group_cache_if_needed(gaussians, point_caching)

        # 9. optimizer
        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
```

---

## 7. 推荐训练 schedule

### 7.1 Indoor

```text
0 ~ 800:
  Deblur warm-up
  GTnet + canonical Gaussian 初步稳定
  Dash resolution 固定为 1
  不 grouping

800 ~ 3000:
  原始 Deblur densify
  仍不 grouping
  可开始轻量 Dash densify，但建议先等 dash_start_iter

2500 / 3000:
  point addition
  merge Group cache if exists
  reset point_caching

3000 ~ 4500:
  Dash resolution / densify 开始生效
  Group 仍关闭

4500 ~ 18000/20000:
  Dash + Group Training
  UTR = 0.75~0.80
  grouping_interval = 600

最后 3000~5000 iter:
  Group 关闭或 UTR=1.0
  full Gaussian refinement
  standard sharp render 验证
```

### 7.2 Outdoor

```text
0 ~ 1000:
  warm-up
  render_scale = 1
  不 grouping

1000 ~ 5000:
  Deblur + Dash densify 逐步启动
  不 grouping

3500 / 5000:
  point addition
  outdoor 新点保护更长

6500 ~ 24500/26000:
  Dash + Group Training
  UTR = 0.70~0.75
  grouping_interval = 1000

最后 4000~5000 iter:
  Group 关闭
  full Gaussian refinement
```

---

## 8. 配置文件

### 8.1 `configs/indoor_motion_dash_group.txt`

```ini
# Base
iterations = 24000
resolution = 2
white_background = False
eval = True

# Deblur: camera motion blur
deblur = 1
use_pos = 1
num_moments = 4
hidden = 3
width = 64
gtnet_lr = 0.001
lambda_s = 0.01
lambda_p = 0.008
max_clamp = 1.08

# Optimization
position_lr_init = 0.0016
position_lr_final = 0.000016
position_lr_delay_mult = 0.01
position_lr_max_steps = 24000
feature_lr = 0.0025
opacity_lr = 0.05
scaling_lr = 0.005
rotation_lr = 0.001
lambda_dssim = 0.2
percent_dense = 0.01

# Densification
densify_from_iter = 800
densify_until_iter = 17000
densification_interval = 100
densify_grad_threshold = 0.00045
densify_prune_threshold = 0.008
densify_with_depth = 1
prune_range = 3

# Point addition
pts_iter = 2500
pts_rate = 1.1
pts_dist = 2
pts_N_intpl = 4
pts_N_pts = 200000
pts_add_bound = 10
protect_new_points_iters = 1500

# DashGaussian
dash_enable = True
dash_start_iter = 3000
resolution_mode = freq
densify_mode = freq
max_n_gaussian = -1
dash_max_reso_scale = 4
dash_start_significance_factor = 4
dash_max_densify_rate_per_step = 0.12

# Group Training
Grouping = True
grouping_method = Opacity-weighted
UTR = 0.78
grouping_from_iter = 4500
grouping_until_iter = 20000
grouping_interval = 600
grouping_freeze_around_pts = 1000

# Save/test
test_iterations = 12000 20000 24000
save_iterations = 24000
checkpoint_iterations = 24000
```

---

### 8.2 `configs/indoor_defocus_dash_group.txt`

```ini
# Base
iterations = 22000
resolution = 2
white_background = False
eval = True

# Deblur: defocus blur
deblur = 2
use_pos = 1
num_moments = 3
hidden = 2
width = 64
gtnet_lr = 0.001
lambda_s = 0.008
lambda_p = 0.0
max_clamp = 1.06

# Optimization
position_lr_init = 0.0016
position_lr_final = 0.000016
position_lr_delay_mult = 0.01
position_lr_max_steps = 22000
feature_lr = 0.0025
opacity_lr = 0.05
scaling_lr = 0.005
rotation_lr = 0.001
lambda_dssim = 0.2
percent_dense = 0.01

# Densification
densify_from_iter = 800
densify_until_iter = 16000
densification_interval = 100
densify_grad_threshold = 0.00018
densify_prune_threshold = 0.0045
densify_with_depth = 1
prune_range = 3

# Point addition
pts_iter = 2500
pts_rate = 1.1
pts_dist = 2
pts_N_intpl = 4
pts_N_pts = 200000
pts_add_bound = 10
protect_new_points_iters = 1500

# DashGaussian
dash_enable = True
dash_start_iter = 3000
resolution_mode = freq
densify_mode = freq
max_n_gaussian = -1
dash_max_reso_scale = 4
dash_start_significance_factor = 4
dash_max_densify_rate_per_step = 0.10

# Group Training
Grouping = True
grouping_method = Opacity-weighted
UTR = 0.82
grouping_from_iter = 4500
grouping_until_iter = 18500
grouping_interval = 600
grouping_freeze_around_pts = 1000

# Save/test
test_iterations = 10000 18000 22000
save_iterations = 22000
checkpoint_iterations = 22000
```

---

### 8.3 `configs/outdoor_motion_dash_group.txt`

```ini
# Base
iterations = 30000
resolution = 4
white_background = False
eval = True

# Deblur: camera motion blur
deblur = 1
use_pos = 1
num_moments = 4
hidden = 3
width = 64
gtnet_lr = 0.001
lambda_s = 0.01
lambda_p = 0.01
max_clamp = 1.10

# Optimization
position_lr_init = 0.0012
position_lr_final = 0.000012
position_lr_delay_mult = 0.01
position_lr_max_steps = 30000
feature_lr = 0.0025
opacity_lr = 0.05
scaling_lr = 0.005
rotation_lr = 0.001
lambda_dssim = 0.2
percent_dense = 0.01

# Densification
densify_from_iter = 1000
densify_until_iter = 22000
densification_interval = 100
densify_grad_threshold = 0.0005
densify_prune_threshold = 0.008
densify_with_depth = 1
prune_range = 4

# Point addition
pts_iter = 3500
pts_rate = 1.3
pts_dist = 3
pts_N_intpl = 4
pts_N_pts = 200000
pts_add_bound = 20
protect_new_points_iters = 2500

# DashGaussian
dash_enable = True
dash_start_iter = 5000
resolution_mode = freq
densify_mode = freq
max_n_gaussian = -1
dash_max_reso_scale = 4
dash_start_significance_factor = 4
dash_max_densify_rate_per_step = 0.10

# Group Training
Grouping = True
grouping_method = Opacity-weighted
UTR = 0.75
grouping_from_iter = 6500
grouping_until_iter = 26000
grouping_interval = 1000
grouping_freeze_around_pts = 1500

# Save/test
test_iterations = 15000 24000 30000
save_iterations = 30000
checkpoint_iterations = 30000
```

---

### 8.4 `configs/outdoor_defocus_dash_group.txt`

```ini
# Base
iterations = 28000
resolution = 4
white_background = False
eval = True

# Deblur: defocus blur
deblur = 2
use_pos = 1
num_moments = 3
hidden = 2
width = 64
gtnet_lr = 0.001
lambda_s = 0.008
lambda_p = 0.0
max_clamp = 1.08

# Optimization
position_lr_init = 0.0012
position_lr_final = 0.000012
position_lr_delay_mult = 0.01
position_lr_max_steps = 28000
feature_lr = 0.0025
opacity_lr = 0.05
scaling_lr = 0.005
rotation_lr = 0.001
lambda_dssim = 0.2
percent_dense = 0.01

# Densification
densify_from_iter = 1000
densify_until_iter = 21000
densification_interval = 100
densify_grad_threshold = 0.00022
densify_prune_threshold = 0.004
densify_with_depth = 1
prune_range = 4

# Point addition
pts_iter = 3500
pts_rate = 1.3
pts_dist = 3
pts_N_intpl = 4
pts_N_pts = 200000
pts_add_bound = 20
protect_new_points_iters = 2500

# DashGaussian
dash_enable = True
dash_start_iter = 5000
resolution_mode = freq
densify_mode = freq
max_n_gaussian = -1
dash_max_reso_scale = 4
dash_start_significance_factor = 4
dash_max_densify_rate_per_step = 0.09

# Group Training
Grouping = True
grouping_method = Opacity-weighted
UTR = 0.78
grouping_from_iter = 6500
grouping_until_iter = 24500
grouping_interval = 1000
grouping_freeze_around_pts = 1500

# Save/test
test_iterations = 14000 22000 28000
save_iterations = 28000
checkpoint_iterations = 28000
```

> 注意：上面 `outdoor_defocus` 中 `densify_with_depth = 1` 是保守默认。如果户外 COLMAP 深度噪声很大，可改为 `0`，并适当提高 `densify_grad_threshold`。

---

## 9. 训练命令

### Indoor motion

```bash
python train.py \
  -s data/indoor_motion_blur/scene_name \
  --expname indoor_motion_dash_group \
  --config configs/indoor_motion_dash_group.txt
```

### Indoor defocus

```bash
python train.py \
  -s data/indoor_defocus_blur/scene_name \
  --expname indoor_defocus_dash_group \
  --config configs/indoor_defocus_dash_group.txt
```

### Outdoor motion

```bash
python train.py \
  -s data/outdoor_motion_blur/scene_name \
  --expname outdoor_motion_dash_group \
  --config configs/outdoor_motion_dash_group.txt
```

### Outdoor defocus

```bash
python train.py \
  -s data/outdoor_defocus_blur/scene_name \
  --expname outdoor_defocus_dash_group \
  --config configs/outdoor_defocus_dash_group.txt
```

---

## 10. 消融实验顺序

不要一次性全开。按以下顺序验证：

```text
Experiment 0:
  Deblur original baseline
  Dash = off
  Group = off

Experiment 1:
  Deblur + Dash resolution only
  resolution_mode = freq
  densify_mode = free
  Group = off

Experiment 2:
  Deblur + Dash resolution + Dash densify scheduler
  resolution_mode = freq
  densify_mode = freq
  Group = off

Experiment 3:
  Deblur + Dash + Group late-start
  Grouping from 4500 / 6500
  UTR >= 0.75 indoor
  UTR >= 0.70 outdoor

Experiment 4:
  Final full refinement
  Group off in last stage
  standard render.py sharp validation
```

通过标准后再考虑第二阶段优化：

```text
更精细的 Dash top-k densify；
SparseAdam；
3dgs-accel rasterizer；
自定义 lightweight inference renderer。
```

不建议第二阶段立即重新引入 Speedy-Splat，除非已经证明 standard renderer 输出质量稳定且需要单独做部署 FPS benchmark。

---

## 11. 日志与 debug 指标

训练日志必须记录：

```text
iteration
render_scale
N_GS
densify_rate
momentum_add
active_N / cached_N / UTR
GTnet scale_delta mean / max
GTnet pos_delta mean / max, motion blur only
PSNR / SSIM / LPIPS
blurry train render
sharp test render
```

建议每 1000 iter 保存：

```text
debug/train_blurry_render_iter.png
debug/test_sharp_render_iter.png
debug/test_gt_iter.png
debug/scale_delta_hist.png
debug/pos_delta_hist.png
```

常见问题判断：

```text
blurry render 好，但 sharp render 差：
  GTnet 过强，canonical geometry 欠训练；降低 lambda_s/lambda_p，推迟 Group，增大 UTR。

N_GS 爆炸：
  dash_max_densify_rate_per_step 太高，或 densify_until_iter 太晚；降低 densify rate 或设置 max_n_gaussian。

PSNR 突然掉：
  Grouping 太早、UTR 太低，或 cache 没有在 densify/add_points 前 merge。

pts_iter 后质量波动大：
  grouping_freeze_around_pts 太小，或 protect_new_points_iters 太短。

defocus 近景 halo：
  max_clamp 过大，lambda_s 过高，或 Group active ratio 太低。

motion blur 鬼影：
  num_moments 不够、lambda_p 不稳，或 GTnet 在低分辨率阶段过早学习。
```

---

## 12. 给 Codex 的实现指令

```text
Implement DashDeblurGroupGS using benhenryL/Deblurring-3D-Gaussian-Splatting as the base repository.

High-level rules:
1. Preserve Deblurring-3DGS blur physics exactly:
   - GTnet stays inside GaussianModel.
   - Training render uses deblur=1 for motion blur and deblur=2 for defocus blur.
   - Inference render uses deblur=0 and canonical Gaussians.
   - Do not rewrite motion blur or defocus blur branches.
   - Do not remove add_points().

2. Add DashGaussian only as a scheduler:
   - Copy utils/schedule_utils.py from DashGaussian.
   - Add DeblurDashScheduler wrapper.
   - Add render_size support to gaussian_renderer.render().
   - Downsample gt_image according to render_scale.
   - Pass render_size=gt_image.shape[-2:] to render().
   - Start Dash only after dash_start_iter.
   - Add prune_and_densify_deblur_safe() wrapper that returns momentum_add.

3. Add Group Training as a submodule:
   - Add gaussians_grouping from Chengbo-Wang/3DGS-with-Group-Training.
   - Import gaussians_grouping_and_caching and GroupingParams.
   - Run grouping before render only after grouping_from_iter.
   - Disable grouping around pts_iter using grouping_freeze_around_pts.
   - Always merge cached Gaussians before densify, add_points, save, checkpoint, and final exit.
   - Use high UTR for deblur: indoor 0.78~0.82, outdoor 0.75~0.78.

4. Remove Speedy-Splat completely from this implementation:
   - Do not add render_speedy.py.
   - Do not add export_speedy.py.
   - Do not add diff-gaussian-rasterization-speedy.
   - Do not call Speedy prune(), soft pruning, hard pruning, sparse primitive pruning, or sparse pixel localization.
   - Do not include Speedy FPS as a default benchmark.

5. First implementation should not use:
   - SparseAdam
   - Dash antialiasing branch
   - 3dgs-accel rasterizer
   - FastGS VCD/VCP
   - any renderer-level pruning

6. Add config files:
   - configs/indoor_motion_dash_group.txt
   - configs/indoor_defocus_dash_group.txt
   - configs/outdoor_motion_dash_group.txt
   - configs/outdoor_defocus_dash_group.txt

7. Add logging:
   - render_scale
   - densify_rate
   - momentum_add
   - number of Gaussians
   - active/cached Gaussian counts
   - GTnet scale_delta statistics
   - GTnet pos_delta statistics for motion blur
   - sharp render metrics
```

---

## 13. 最终推荐组合

```text
训练：
  Deblurring-3DGS 原始 blur render
  + DashGaussian resolution scheduling
  + DashGaussian Gaussian growth scheduling
  + Group Training late-start / high-UTR temporary caching

推理：
  Deblurring-3DGS standard render.py
  deblur = 0
  canonical Gaussian only
```

不推荐：

```text
Deblur + Speedy full train
Deblur + Speedy pruning
Deblur + Speedy sparse primitive pruning
Deblur + aggressive early Group Training
Deblur + Dash max_reso_scale=8 first run
Deblur + SparseAdam + Group + Dash + renderer replacement 一次性全开
```

一句话总结：

**本方案删除 Speedy-Splat，把系统收敛为更稳的 Deblur + Dash + Group 三模块架构：Deblur 只管模糊物理，Dash 只管分辨率与 Gaussian 增长，Group 只管临时训练减负。所有可能改变 canonical Gaussian 语义或提前剪枝的推理加速策略，都暂时移出主线。**
