# Deblurring-3DGS 融合工程完整实施方案

> **本文档用途**：直接交给 codex 执行。  
> 第一次阅读即可理解背景、目标、每个决策的原因、实现细节和验收标准。  
> 按照第 9 章 Phase 顺序逐步实施，每个 Phase 独立提交、独立验证。

---

## 0. Current Implementation Progress (2026-05-20)

This section is the active progress record for the current EAP + gsplat integration and supersedes the older Phase 3/Phase 4 notes below where they conflict.

### 0.1 Implemented

- Admin pipeline parameters are now schema-driven:
  - `fine_eap_enabled` defaults to `true`
  - `fine_eap_dbscan_eps = 30`
  - `fine_eap_min_samples = 10`
  - `fine_eap_mask_radius = 20`
  - `fine_eap_max_point_multiplier = 10`
  - `fine_gsplat_enabled` defaults to `false`
- EAP is implemented as an initialization-stage augmentation after COLMAP/undistort and before DashDeblurGroupGS training:
  - Reads `scene_dir/images` and `scene_dir/sparse/0`
  - Writes `points3D_eap.bin` or `points3D_eap.txt`, plus `points3D_eap.ply` and `points3D_eap_meta.json`
  - Does not overwrite original `points3D.*`
  - Reuses original camera poses for `_aug` images
  - Fails if EAP output exceeds `original_points * fine_eap_max_point_multiplier`
- DashDeblurGroupGS config generation now writes:
  - EAP on: `pc_name = points3D_eap`
  - EAP off: `pc_name = points3D`
  - gsplat on: `renderer_backend = gsplat`
  - deblur renderer fixed: `renderer_backend_deblur = original`
- Trainer scene loading now supports `pc_name` instead of always reading `points3D.*`.
- gsplat is wired only into the sharp/canonical render path and imported lazily. Motion/defocus multi-moment render paths keep the original rasterizer.
- Metrics now include `fine_eap_enabled`, `fine_eap_original_points`, `fine_eap_points`, `fine_eap_multiplier`, `fine_gsplat_enabled`, `pc_name`, `renderer_backend`, and `renderer_backend_deblur`.

### 0.2 Environment And Dependency Status

- No new Python requirement files need package additions:
  - `Pillow` and `numpy` already come from `backend/requirements.txt`
  - `scikit-learn` and `gsplat` already come from `worker/requirements.txt`
  - `pycolmap==3.12.6` is installed separately in `worker/Dockerfile`
- Environment checks have been tightened for the new default behavior:
  - `gsplat` is included in fine runtime dependency status
  - worker image COLMAP build validation checks `exhaustive_matcher` and `point_triangulator`
  - runtime COLMAP status treats `exhaustive_matcher` and `point_triangulator` as required for default fine worker capability
- If EAP is enabled and COLMAP CLI or `point_triangulator` is unavailable, the task must fail clearly. It must not silently fall back to raw `points3D`.

### 0.3 Key Files Updated

- Backend: `backend/app/pipeline_parameters.py`, `backend/app/main.py`, `backend/app/algorithms.py`, `backend/app/fine/runner.py`, `backend/app/fine/eap.py`, `backend/app/fine/dash_deblur_group.py`, `backend/app/fine/colmap_cli.py`
- Trainer: `worker/trainer/dash_deblur_group_gs/arguments/__init__.py`, `worker/trainer/dash_deblur_group_gs/scene/__init__.py`, `worker/trainer/dash_deblur_group_gs/scene/dataset_readers.py`, `worker/trainer/dash_deblur_group_gs/gaussian_renderer/__init__.py`, `worker/trainer/dash_deblur_group_gs/gaussian_renderer/backends/__init__.py`, `worker/trainer/dash_deblur_group_gs/gaussian_renderer/backends/gsplat_backend.py`
- Environment: `worker/Dockerfile`
- Tests: `backend/tests/test_eap_augmentation.py`, `backend/tests/test_gsplat_backend.py`, `backend/tests/test_dash_deblur_group_runtime.py`, `backend/tests/test_dash_deblur_group_dataset_readers.py`, `backend/tests/test_colmap_cli_policy.py`, `backend/tests/test_fine_runtime.py`, `backend/tests/test_storage_responses.py`

### 0.4 Verification

Passed:

```bash
python -m py_compile backend/app/fine/colmap_cli.py backend/app/algorithms.py backend/tests/test_colmap_cli_policy.py backend/tests/test_fine_runtime.py

python -m unittest backend.tests.test_colmap_cli_policy backend.tests.test_fine_runtime backend.tests.test_eap_augmentation backend.tests.test_gsplat_backend backend.tests.test_dash_deblur_group_dataset_readers backend.tests.test_dash_deblur_group_runtime
```

Result: 57 tests passed, 4 skipped.

Not yet verified:

- `backend.tests.test_storage_responses`: local Python environment is missing `sqlalchemy`.
- `backend.tests.test_worker_logging`: local Python environment is missing `redis`.
- Real COLMAP + EAP end-to-end smoke test on an image set.
- Real CUDA `gsplat` render smoke test.

### 0.5 Differences From Older Notes Below

- The old rule "all modules default off" no longer applies to EAP. Current admin requirement is `fine_eap_enabled = true` by default.
- EAP is not a training-loop dynamic module in this implementation. It only produces initialization point cloud files `points3D_eap.*`.
- First version does not integrate DetectorFreeSfM. It implements the COLMAP + augmentation flow only.
- gsplat is only an optional rasterizer backend. It does not use gsplat strategy, densification, or pruning.

## 目录

1. 背景与核心问题
2. 解决方案总览
3. 核心约束与禁止事项
4. 模块设计：Per-image 混合模糊路由（Phase 1）
5. 模块设计：Per-image 曝光校正（Phase 2）
6. 模块设计：EAP 初始化增强（Phase 3）
7. 模块设计：gsplat Rasterizer Backend（Phase 4）
8. 模块设计：Deblur-aware GDAGS（Phase 5）
9. 模块设计：NexusGS-lite 稀疏补点（Phase 6）
10. 关键问题修正（6 项工程细节）
11. Render 函数完整实现
12. train.py 主循环完整实现
13. 实施阶段与验收标准
14. 参数配置完整清单
15. 附录

---

## 1. 背景与核心问题

### 1.1 原始仓库是什么

`https://github.com/benhenryL/Deblurring-3D-Gaussian-Splatting`

这个仓库（以下简称 **Deblurring-3DGS**）的核心思想是：

- 3D Gaussian Splatting（3DGS）本身表示的是**理想清晰**的场景。
- 训练时，通过一个叫 **GTnet** 的 MLP，给每个 Gaussian 施加 scale/rotation/position 的变形，让渲染结果变模糊，去拟合输入的模糊图像。
- 推理时，关掉 GTnet，直接渲染 3DGS 本体，得到清晰结果。

训练分两种模式，**全局选一种**：

```
use_pos=1 → motion blur 模式：
    GTnet 输出 scale_delta + rotation_delta + position_delta
    多次渲染再平均，模拟相机运动模糊

use_pos=0 → defocus blur 模式：
    GTnet 只输出 scale_delta + rotation_delta
    单次变换，模拟散焦模糊
```

### 1.2 原始仓库的局限

**局限 1：只支持单一模糊类型**

`use_pos` 是全局开关。真实手机拍摄的数据集里，同一场景的不同图片可能同时包含：
- 清晰图（手持静止拍摄）
- 运动模糊图（手持移动时拍摄）
- 散焦模糊图（对焦不准）

原始代码无法处理这种混合情况——如果设 `use_pos=1`，所有图都走 motion 分支，散焦图会被错误建模；反之亦然。

**局限 2：GTnet 没有 per-image 感知能力**

原始 GTnet 的输入只有 Gaussian 的几何属性（位置、方向、scale）和视线方向，没有"当前处理的是哪张图"的信息。

这意味着：同一个视角下，所有图片产生的模糊变形量是一样的，但实际上每张图的模糊程度和方向各不相同。

**局限 3：手机图特有问题没有处理**

手机自动曝光会导致同一场景不同帧之间亮度不一致（有时差 0.5 EV 以上），这个亮度误差会混入训练 loss，干扰 GTnet 区分"模糊残差"和"亮度差异"。

**局限 4：初始点云质量影响上限**

手机拍摄的数据，COLMAP 生成的初始点云往往在纹理弱的区域（墙面、天空、光滑物体）非常稀疏。初始点云质量直接影响 3DGS 的几何重建上限。

**局限 5：密度控制不适应 deblur 训练**

原始 3DGS 的 densify_and_prune 使用 viewspace gradient 决定是否 split/clone/prune。但 Deblurring-3DGS 的梯度来自 deblur 路径，里面混有几何误差、模糊未收敛、相机位姿误差等多种信号，直接用来控制几何密度会产生"把模糊残差当成几何缺失"的问题。

### 1.3 目标

在不破坏 Deblurring-3DGS 核心去模糊逻辑的前提下，逐步叠加以下改进：

```
目标 1：支持混合模糊类型（每张图独立选 sharp/motion/defocus 路径）
目标 2：每张图有独立的模糊强度感知（blur embedding）
目标 3：修正手机自动曝光导致的亮度不一致（曝光校正）
目标 4：改善初始点云质量（EAP 初始化增强）
目标 5：可选用更快的 rasterizer（gsplat backend）
目标 6：更合理的训练中密度控制（Deblur-aware GDAGS）
目标 7：高置信度补充稀疏区域（NexusGS-lite）
```

---

## 2. 解决方案总览

### 2.1 整体 Pipeline

```
手机输入图像
    ↓
[自动检测或标注] 每张图 blur_type: 0=sharp, 1=motion, 2=defocus
    ↓
COLMAP 重建
    ↓
[Phase 3 可选] EAP-style 初始化点云增强
    ↓
初始化 GaussianModel
    ↓
训练循环：
    采样一张图（weighted sampler，sharp 图优先）
    ↓
    按 blur_type 选择 render 路径：
      sharp   → 直接 rasterize（不走 GTnet）
      motion  → motion_GTnet → 多 moment 渲染平均
      defocus → defocus_GTnet → 单次 scale 变换
    ↓
    [Phase 2 可选] 在 loss 端对渲染图做曝光校正
    ↓
    计算 loss（数据项 + 正则项）
    ↓
    backward → optimizer.step() → zero_grad
    ↓
    [Phase 4 可选] gsplat backend（只对 sharp 图）
    ↓
    [Phase 5 可选] GDAGS canonical probe + 密度控制
    ↓
    [Phase 6 可选] NexusGS-lite 补点
    ↓
推理：deblur=False，渲染清晰 3DGS
```

### 2.2 各模块职责边界（非常重要）

每个模块只做自己的事，不能越界：

| 模块 | 职责 | 不能做 |
|---|---|---|
| Deblurring-3DGS（主系统） | 建模模糊过程，latent sharp 3DGS | 不能被其他模块破坏 |
| Per-image 混合路由 | 让每张图走正确的 blur 分支 | 不改 GTnet 内部逻辑 |
| PerImageExposureModel | 修曝光/亮度不一致 | 不修颜色，不修模糊 |
| EAP 初始化 | 训练前增密点云 | 不参与训练梯度 |
| gsplat backend | 替换 rasterizer | 不引入 gsplat 的 strategy/pruning |
| Deblur-aware GDAGS | 控制训练中密度 | canonical probe 不污染参数梯度 |
| NexusGS-lite | 高置信补稀疏洞 | 不全图 dense，不在模糊区补 |

### 2.3 模块间依赖关系

```
Phase 1（混合路由）← 所有后续 Phase 的基础，必须最先实现
Phase 2（曝光校正）← 依赖 Phase 1 的 image_id
Phase 3（EAP init）← 独立，但在 Phase 1 的训练环境里运行
Phase 4（gsplat）  ← 依赖 Phase 1（需要 blur_type 路由来决定用哪个 backend）
Phase 5（GDAGS）   ← 依赖 Phase 1（canonical probe 需要 sharp_cameras 列表）
Phase 6（Nexus）   ← 依赖 Phase 5（新点需要 GDAGS 保护期机制）
```

---

## 3. 核心约束与禁止事项

### 3.1 不可违背的原则

```
① 主系统不能破坏
  Deblurring-3DGS 的 GTnet / blur_kernel / deblur render 逻辑一行都不能删。
  mixed_blur=False 时，行为必须和原始仓库完全一致。

② 推理路径不能污染
  推理时永远 deblur=False，不传 blur_type，不调用任何 GTnet。
  GTnet 只在训练时建模模糊，推理时不存在。

③ 曝光校正只在 loss 端
  PerImageExposureModel 只修改传入 loss 的渲染图，
  不改 Gaussian 的 SH（颜色属性），不改 render 输出本身。

④ canonical probe 绝对隔离
  GDAGS 的 canonical probe 在主 optimizer.step() + zero_grad() 之后执行。
  用 torch.autograd.grad 取梯度，不调用 loss.backward()，不调用 optimizer.step()。
  执行完后所有模型参数的 .grad 必须为 None。

⑤ GDAGS buffer 必须和 Gaussian 数量一致
  每次 clone/split/prune 操作后，立刻同步 GDAGS buffer。
  不一致时立即报错（assert），不能静默失败。

⑥ 新点必须有保护期
  EAP/Nexus/clone/split 产生的新点，在保护期内不参与 prune。
  EAP 点：前 5000 iter 不 prune。
  Nexus 点：1500 iter 保护期。

⑦ 所有模块有独立开关，默认关闭
  开关全部在 arguments/__init__.py 里，默认 False/original。
  关掉所有开关，结果必须等同于原始仓库。
```

### 3.2 严格禁止事项

```
禁止用 deblur residual（模糊残差）直接 add / split / prune Gaussian。
禁止用 blurry 图做 canonical gradient 统计（GDAGS probe 只用 sharp 图）。
禁止 canonical probe 之后调用 optimizer.step()。
禁止 GDAGS buffer 和 gaussians.get_xyz 数量不同步。
禁止对四元数做逐元素乘法后不归一化（会破坏旋转合法性）。
禁止 gsplat 接管 motion/defocus 分支（第一版只对 sharp 图用 gsplat）。
禁止 PerImageExposureModel 使用 per-channel scale（会退化为 RGB scale+bias）。
禁止 PerImageExposureModel 添加 tone curve 或 color matrix。
禁止 EAP/Nexus 新点在保护期内被 prune。
禁止 sharp 图的采样比例低于 30%（防止 deblur 图主导几何）。
```

---

## 4. 模块设计：Per-image 混合模糊路由（Phase 1）

### 4.1 为什么需要这个

原始代码的 `use_pos` 是全局开关，整个训练只能走一种 blur 模式。手机数据是混合的，必须让每张图独立选择路径。

同时，原始 GTnet 没有"这是第几张图"的感知。两张看起来相似的运动模糊图，模糊方向和强度可能完全不同，GTnet 需要知道自己在处理哪张图，才能输出图片特定的变形量。

解决方案：
- 给每张图一个 `blur_type`（0/1/2）决定走哪条路径
- 给每张图一个可学习的 `blur_code`（embedding），拼入 GTnet 输入，让 GTnet 知道当前图的模糊特征

### 4.2 blur_type 来源

优先使用标注文件 `blur_labels.json`，不存在时自动检测：

```json
{
  "version": "1.0",
  "images": {
    "000001.png": {"blur_type": "sharp",   "sharpness": 0.92},
    "000002.png": {"blur_type": "motion",  "sharpness": 0.31},
    "000003.png": {"blur_type": "defocus", "sharpness": 0.44}
  }
}
```

自动检测逻辑（`utils/auto_blur_detect.py`）：

```python
def auto_detect_blur_type(image_tensor: torch.Tensor,
                           sharp_threshold: float = 500.0,
                           motion_dir_ratio: float = 2.5):
    """
    基于 Laplacian variance 和梯度方向性区分 sharp/motion/defocus。
    
    原理：
      - Laplacian variance 高 → 图像清晰
      - 模糊图中，motion blur 的梯度在某一方向上更强（方向性）
      - defocus blur 各向同性，梯度在各方向均匀分布
    
    返回 (blur_type: int, sharpness_score: float)
    """
    import cv2
    import numpy as np

    # tensor [3, H, W] → numpy [H, W, 3]
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # 清晰度
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if lap_var > sharp_threshold:
        return 0, float(lap_var)  # sharp

    # 方向性判断：motion blur 在主方向上梯度更强
    sobelx_var = cv2.Sobel(gray, cv2.CV_64F, 1, 0).var()
    sobely_var = cv2.Sobel(gray, cv2.CV_64F, 0, 1).var()
    ratio = sobelx_var / (sobely_var + 1e-6)

    if ratio > motion_dir_ratio or ratio < 1.0 / motion_dir_ratio:
        return 1, float(lap_var)  # motion
    else:
        return 2, float(lap_var)  # defocus
```

### 4.3 Camera 对象新增字段

在 `scene/cameras.py` 的 `Camera.__init__` 里新增：

```python
self.image_id: int          # 全局连续 ID，从 0 开始，用于 Embedding 索引
self.blur_type: int         # 0=sharp, 1=motion, 2=defocus
self.sharpness_score: float # Laplacian variance，供 EAP/Nexus 使用
```

注意：`image_id` 必须是连续整数（0, 1, 2, ...），不能用 COLMAP 的 colmap_id（可能不连续）。

### 4.4 ConditionalGTnet 设计

基于原始 `GTnet`，只增加一个改动：把 per-image `blur_code` 拼入 MLP 输入。

**为什么用 blur_code_dim=8**：
- 维度太大（如 16/32）会给 GTnet 过多的自由度，可能把曝光、颜色变化也"吸收"进去，和 PerImageExposureModel 抢解释权
- 8 维足以表示每张图的模糊特征（方向、强度），又不会过拟合

```python
# scene/blur_kernel.py
# 在原始 GTnet 下方新增，不删除原始 GTnet（兼容性保留）

class ConditionalGTnet(nn.Module):
    """
    扩展自原始 GTnet，增加 per-image blur embedding。
    
    核心改动（只有这一处）：
      原始：x = cat([pos_emb, view_emb, scales, rotations])
      改后：x = cat([pos_emb, view_emb, scales, rotations, blur_code])
    
    blur_code 是可学习的 nn.Embedding，每张训练图一个向量，
    让 GTnet 知道当前处理的是哪张图的模糊状态。
    
    pos_delta=True  → motion 分支（输出 position delta）
    pos_delta=False → defocus 分支（不输出 position delta）
    """

    def __init__(
        self,
        num_images: int,
        blur_code_dim: int = 8,
        res_pos: int = 3,
        res_view: int = 10,
        num_hidden: int = 3,
        width: int = 64,
        pos_delta: bool = False,
        num_moments: int = 4,
        enable_rotation_delta: bool = True,
    ):
        super().__init__()
        self.pos_delta = pos_delta
        self.num_moments = num_moments
        self.enable_rotation_delta = enable_rotation_delta

        # blur embedding：小 std 初始化，防止训练初期 GTnet 输出过大
        self.blur_codes = nn.Embedding(num_images, blur_code_dim)
        nn.init.normal_(self.blur_codes.weight, mean=0.0, std=0.005)

        # 位置和方向编码（与原始相同）
        self.embed_pos,  embed_pos_dim  = get_embedder(res_pos,  3)
        self.embed_view, embed_view_dim = get_embedder(res_view, 3)

        # 输入维度 = 位置编码 + 方向编码 + scale(3) + rotation(4) + blur_code
        in_dim = embed_pos_dim + embed_view_dim + 7 + blur_code_dim

        # MLP 结构（与原始相同，只有输入维度变了）
        layers = [nn.Linear(in_dim, width), nn.ReLU()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        self.linears = nn.Sequential(*layers)

        # 输出头
        if not pos_delta:
            # defocus 分支
            self.out_s = nn.Linear(width, 3)                        # scale delta
            self.out_r = nn.Linear(width, 4) if enable_rotation_delta else None
        else:
            # motion 分支：(M+1) 组 scale/rotation + M 组 position
            self.out_s = nn.Linear(width, 3 * (num_moments + 1))
            self.out_r = nn.Linear(width, 4 * (num_moments + 1))
            self.out_p = nn.Linear(width, 3 * num_moments)

        # 小初始化防止训练早期输出过大
        self._init_all_weights()

    def _init_all_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, pos, scales, rotations, viewdirs, image_id):
        """
        pos:       [N, 3]   Gaussian 中心坐标
        scales:    [N, 3]   Gaussian 缩放
        rotations: [N, 4]   Gaussian 旋转四元数
        viewdirs:  [N, 3]   视线方向
        image_id:  int 或 LongTensor  当前图的 ID
        
        返回：(s_delta, r_delta, p_delta, blur_code_z)
        """
        N = pos.shape[0]

        pos_emb  = self.embed_pos(pos)
        view_emb = self.embed_view(viewdirs)

        if not isinstance(image_id, torch.Tensor):
            image_id = torch.tensor(image_id, device=pos.device, dtype=torch.long)

        # blur_code：当前图的模糊特征向量
        z = self.blur_codes(image_id)               # [blur_code_dim]
        z_exp = z.view(1, -1).expand(N, -1)         # [N, blur_code_dim]

        x = torch.cat([pos_emb, view_emb, scales, rotations, z_exp], dim=-1)
        h = self.linears(x)

        s_delta = self.out_s(h)
        r_delta = self.out_r(h) if self.out_r is not None else None
        p_delta = self.out_p(h) if self.pos_delta else None

        return s_delta, r_delta, p_delta, z
```

### 4.5 GaussianModel 改动

在 `scene/gaussian_model.py` 中新增两个方法：

```python
def create_deblur_nets(self, num_images: int, opt):
    """
    创建 GTnet。
    
    mixed_blur=False（兼容原始）：
        只创建一个 ConditionalGTnet，行为等同于原始 GTnet（加了 blur_code）。
    
    mixed_blur=True（新功能）：
        创建两个分支：motion_GTnet 和 defocus_GTnet。
        每张图根据 blur_type 动态选择分支。
    """
    if not opt.mixed_blur:
        # 兼容模式：单一网络，use_pos 全局控制
        self.GTerr = ConditionalGTnet(
            num_images=num_images,
            blur_code_dim=opt.blur_code_dim,
            num_hidden=opt.hidden,
            width=opt.width,
            pos_delta=bool(opt.use_pos),
            num_moments=opt.num_moments,
        ).cuda()
    else:
        # 混合模式：双分支
        self.motion_GTnet = ConditionalGTnet(
            num_images=num_images,
            blur_code_dim=opt.blur_code_dim,
            num_hidden=opt.hidden,
            width=opt.width,
            pos_delta=True,
            num_moments=opt.num_moments,
            enable_rotation_delta=True,
        ).cuda()

        self.defocus_GTnet = ConditionalGTnet(
            num_images=num_images,
            blur_code_dim=opt.blur_code_dim,
            num_hidden=opt.hidden,
            width=opt.width,
            pos_delta=False,
            num_moments=opt.num_moments,
            enable_rotation_delta=opt.defocus_enable_rotation_delta,
            # 注意：defocus 默认关闭 rotation delta，只学 scale
            # 原因：rotation 四元数乘法需要特殊处理，等 scale 版稳定后再开放
        ).cuda()

def get_deblur_optimizer_params(self, lr: float) -> list:
    """
    返回所有 GTnet 参数组，用于加入 optimizer。
    在 training_setup() 里的 l = [...] 中调用此方法并 extend。
    """
    params = []
    for name in ["GTerr", "motion_GTnet", "defocus_GTnet"]:
        net = getattr(self, name, None)
        if net is not None:
            params.append({
                "params": net.parameters(),
                "lr": lr,
                "name": name,
            })
    return params
```

### 4.6 加权采样策略

**为什么需要加权采样**：  
如果 blurry 图远多于 sharp 图，训练时 3DGS 几何会被 blurry 图主导，GTnet 会过度变形，导致推理时的清晰渲染比 baseline 更糊。Sharp 图对几何约束最直接，必须保证足够的采样比例。

```python
def sample_camera(iteration, train_cameras, sharp_cameras, motion_cameras,
                   defocus_cameras, opt):
    """
    加权采样策略：
    
    前 warmup_iters（默认 3000）：
        70% 概率从 sharp 图采样
        目的：先让 3DGS 建立基本几何，再引入模糊图
    
    之后：
        按 sharp_sample_ratio / motion_sample_ratio / defocus 比例采样
        默认 35% / 35% / 30%
    
    注意：sharp_sample_ratio 不能低于 0.30，否则几何会被模糊图带偏。
    """
    if not opt.mixed_blur:
        idx = randint(0, len(train_cameras) - 1)
        return train_cameras[idx]

    if iteration < opt.warmup_iters and len(sharp_cameras) > 0:
        if random.random() < 0.70:
            return random.choice(sharp_cameras)

    r = random.random()
    if r < opt.sharp_sample_ratio and len(sharp_cameras) > 0:
        return random.choice(sharp_cameras)
    elif r < opt.sharp_sample_ratio + opt.motion_sample_ratio and len(motion_cameras) > 0:
        return random.choice(motion_cameras)
    elif len(defocus_cameras) > 0:
        return random.choice(defocus_cameras)
    return random.choice(train_cameras)
```

---

## 5. 模块设计：Per-image 曝光校正（Phase 2）

### 5.1 为什么需要这个，以及为什么只用 2 个参数

手机自动曝光（AEC）会在不同帧之间产生亮度差异。如果不处理，训练 loss 里会混入"这张图整体比较亮"这种与场景几何无关的误差，GTnet 可能会错误地用模糊变形来解释亮度变化。

**为什么不用 per-channel RGB scale + bias（6参数）或 3x3 matrix（12参数）**：

手机图的问题主要是曝光不一致，颜色一般比较稳定（白平衡通常由同一相机的自动白平衡处理）。引入颜色自由度会：
1. 和 blur_code 抢解释权（两者都是 per-image 的可学习参数）
2. 可能把局部颜色误差、甚至模糊残差都"吸收"进来
3. 增加训练不稳定性

结论：只用 **2 个标量参数**（曝光增益 + 亮度偏置），是最保守、最安全的选择。

**gain 为什么用 log 域**：
- 确保 gain > 0（exp 的值域）
- 初始化为 0 时 gain = exp(0) = 1，天然 identity
- log 域的范围限制更自然（[-0.30, 0.30] 对应 [0.74, 1.35]）

**clamp 为什么是硬约束而不是只靠正则**：
- 正则是软约束，training loss 大时正则可能被"压制"，gain 仍然越界
- hard clamp 保证模型物理上不可能解释超出曝光范围的差异
- 如果数据的曝光差异超过这个范围，说明需要先做预处理，不应该靠这个模块修

### 5.2 完整实现

```python
# scene/luminance_model.py

import torch
import torch.nn as nn


class PerImageExposureModel(nn.Module):
    """
    Per-image 标量曝光校正。
    
    职责：修正手机自动曝光导致的跨帧亮度不一致。
    边界：只修整体亮度，不修颜色，不修模糊。
    
    变换公式：
        corrected = clamp(image * exp(clamp(log_gain, -0.30, 0.30))
                          + clamp(bias, -0.05, 0.05),
                          0.0, 1.0)
    
    参数量：每张图 2 个（log_gain + bias）
    初始状态：log_gain=0（gain=1），bias=0，等效 identity（不做任何修改）
    
    使用位置：只在 loss 计算端
        image_for_loss = exposure_model(rendered_image, image_id)
        loss = l1_loss(image_for_loss, gt_image)
    
    不使用的地方：
        推理时不调用（推理只渲染 canonical 3DGS）
        不修改 Gaussian 的 SH 或其他属性
    """

    LOG_GAIN_MIN = -0.30   # 对应 gain ≈ 0.74（最暗允许校正到原来的 0.74 倍）
    LOG_GAIN_MAX =  0.30   # 对应 gain ≈ 1.35（最亮允许校正到原来的 1.35 倍）
    BIAS_MIN     = -0.05   # 绝对亮度偏移范围（pixel value，[0,1] 域）
    BIAS_MAX     =  0.05

    def __init__(self, num_images: int):
        super().__init__()
        # shape [N, 1, 1]：方便直接 broadcast 到 [3, H, W]
        self.log_gain = nn.Parameter(torch.zeros(num_images, 1, 1))
        self.bias     = nn.Parameter(torch.zeros(num_images, 1, 1))

    def forward(self, image: torch.Tensor, image_id: int) -> torch.Tensor:
        """
        image:    [3, H, W]，float，[0, 1]
        image_id: int
        返回:     [3, H, W]，校正后图像，仅用于 loss
        """
        log_g = self.log_gain[image_id]   # [1, 1]
        b     = self.bias[image_id]        # [1, 1]

        # hard clamp：防止模型解释超出曝光范围的差异
        log_g = torch.clamp(log_g, self.LOG_GAIN_MIN, self.LOG_GAIN_MAX)
        b     = torch.clamp(b,     self.BIAS_MIN,     self.BIAS_MAX)

        gain = torch.exp(log_g)           # [1, 1]，broadcast 到 [3, H, W]
        corrected = image * gain + b
        return torch.clamp(corrected, 0.0, 1.0)

    def regularization_loss(self,
                             lambda_gain: float = 5e-3,
                             lambda_bias: float = 1e-2) -> torch.Tensor:
        """
        正则项：惩罚偏离 identity 的校正量。
        lambda_bias > lambda_gain：bias 更容易无意中解释模糊残差，惩罚更强。
        """
        return (lambda_gain * (self.log_gain ** 2).mean()
              + lambda_bias * (self.bias     ** 2).mean())

    def get_stats(self) -> dict:
        """每 500 iter 打印，监控校正量是否在合理范围。"""
        with torch.no_grad():
            gains  = torch.exp(self.log_gain).squeeze()
            biases = self.bias.squeeze()
            return {
                "gain_mean": gains.mean().item(),
                "gain_std":  gains.std().item(),
                "gain_min":  gains.min().item(),
                "gain_max":  gains.max().item(),
                "bias_mean": biases.mean().item(),
                "bias_max":  biases.abs().max().item(),
            }
```

### 5.3 接入 train.py 的要点

```python
# 初始化
if opt.luminance_enable:
    exposure_model = PerImageExposureModel(num_images=len(train_cameras)).cuda()
    # 独立 optimizer，学习率比主训练小一个数量级
    exp_optimizer = torch.optim.Adam(exposure_model.parameters(), lr=opt.luminance_lr)

# 每个 iteration（在主 optimizer.step() + zero_grad() 之后）
if opt.luminance_enable and iteration >= opt.luminance_start_iter:
    image_for_loss = exposure_model(rendered_image, image_id)
    # ... 计算 loss ...
    exp_optimizer.step()
    exp_optimizer.zero_grad(set_to_none=True)
```

**关键顺序**：主 optimizer.step() → 主 zero_grad() → 计算曝光校正后的 loss → exp_optimizer.step()

---

## 6. 模块设计：EAP 初始化增强（Phase 3）

### 6.1 为什么需要这个

COLMAP 的稀疏点云重建依赖特征点匹配。手机图的以下情况会导致点云稀疏：
- 弱纹理区域（墙面、地板、天空）：特征点少，COLMAP 生成的点很少
- 少视角：同一区域只有少数图片能看到，三角化精度低

初始点云稀疏的区域，3DGS 训练时只能靠 densification 逐步补充，但 densification 依赖梯度信号，在几何完全没有 Gaussian 的区域，梯度也很弱，难以自我修复。

EAP 的思路：在 COLMAP 后、Gaussian 初始化前，用**两视角的高置信特征匹配轨迹**生成额外的 3D 点，只在 COLMAP 点云稀疏的区域增补，不替换原有点。

### 6.2 接口设计

```python
# scene/point_augmentation/eap_init.py

def eap_augment_pointcloud(
    colmap_cameras: dict,
    colmap_images: dict,
    colmap_points3D: dict,
    input_images_dir: str,
    opt,
) -> dict:
    """
    EAP-style 点云初始化增强。
    
    执行时机：COLMAP 完成后、GaussianModel 初始化前。
    
    流程：
    1. 找 COLMAP 点云中局部密度低的区域（sparse region）
    2. 在 sparse region 附近找高质量的两视角特征匹配轨迹
    3. 三角化得到候选 3D 点
    4. 过滤掉质量差的候选点
    5. Append 到 colmap_points3D
    
    质量过滤标准（全部满足才接受）：
    - reprojection_error < eap_min_reproj_error
    - baseline_deg ∈ [eap_min_baseline_deg, eap_max_baseline_deg]
    - 局部点密度低于第 eap_low_density_percentile 百分位
    - 图像 patch 不过曝（mean > 0.95 的 patch 跳过）
    - 图像 patch 不严重模糊（sharpness > threshold）
    
    返回：扩充后的 points3D（原始点保留，新点 append）
    """
    ...
```

**注意事项**：
- 不要改变相机位姿
- 新点 append，不替换原始点
- 如果 `eap_accepted_points > original_points * 0.5`，截断，防止初始点爆炸
- 打印详细日志：original / candidate / accepted / rejected_reproj / rejected_density / rejected_quality

---

## 7. 模块设计：gsplat Rasterizer Backend（Phase 4）

### 7.1 为什么要引入 gsplat

`diff_gaussian_rasterization`（原始使用的 CUDA 扩展）在某些 GPU/CUDA 版本组合下速度较慢，或不支持某些高级功能。

`gsplat` 是一个独立实现的 Gaussian rasterizer，接口更现代，支持 `packed=True`（内存优化）、`absgrad`（绝对梯度统计）等特性，未来可以为 GDAGS 统计提供更好的梯度信息。

### 7.2 使用边界（非常重要）

**允许**：
- 替换 rasterize() 调用
- 返回 image / alpha / radii / visibility / depth

**禁止**：
- 不使用 gsplat 自带的 strategy（gsplat 有内置的 densification 策略）
- 不使用 gsplat 的 pruning
- 不让 gsplat 接管 motion/defocus 的多 moment 渲染循环

### 7.3 第一版只对 sharp 图启用

原因：
- motion 分支需要循环调用 M+1 次 rasterizer，gsplat 的坐标约定和梯度累积方式与原始不同，需要单独验证
- 第一版先保证 sharp 路径正确，deblur 路径暂时维持 original

用两个独立开关控制：

```python
--renderer_backend        original  # sharp 图（blur_type=0）用
--renderer_backend_deblur original  # motion/defocus 图用（第一版固定 original）
```

### 7.4 接口设计

```python
# gaussian_renderer/backends/gsplat_backend.py

def gsplat_rasterize(
    means3D, means2D, shs, colors_precomp,
    opacity, scales, rotations, cov3D_precomp,
    raster_settings,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    用 gsplat.rasterization 替换 diff_gaussian_rasterization。
    返回 (rendered_image [3,H,W], radii [N]) 格式与原始一致。
    
    坐标约定对齐：
    - gsplat 使用 OpenCV 坐标系，原始用 OpenGL
    - 需要处理 y 轴翻转
    - 输出 shape 转换回 [C, H, W]
    """
    from gsplat import rasterization
    ...
```

### 7.5 验收标准

同一场景，同一随机种子，对比 `renderer_backend=original` 和 `renderer_backend=gsplat`：
- iter 1 的渲染图 max diff < 0.01
- iter 1000 的 PSNR diff < 0.5 dB
- 无 NaN，无黑图，颜色方向正常
- radii 分布基本一致（允许数值误差）

---

## 8. 模块设计：Deblur-aware GDAGS（Phase 5）

### 8.1 为什么原始 densify_and_prune 不够好

原始 3DGS 的密度控制逻辑：
- viewspace gradient 大 → split 或 clone
- opacity 低 → prune

这在普通 3DGS 里是合理的，但在 Deblurring-3DGS 里，viewspace gradient 来自 deblur render 路径，混合了多种信号：
- 真实几何误差（应该 split）
- GTnet 还没收敛（不该 split）
- 相机位姿误差（不该 split）
- 曝光差异（不该 split）

如果直接用 deblur gradient 控制几何，会产生"把模糊学不好的区域当成几何缺失"的问题，在拖影区域、过曝区域产生大量错误的 split，形成 floaters。

### 8.2 GDAGS 的核心思想

GDAGS（Gradient-Direction-Aware Density Control）的核心指标是 **GCR（Gradient Coherence Ratio）**：

```
normalized_grad = grad / ||grad||
gcr = ||sum(normalized_grad) / count||

取值 [0, 1]：
  gcr ≈ 1：多次梯度方向一致，Gaussian 在稳定拟合某个几何细节
  gcr ≈ 0：多次梯度方向冲突，Gaussian 可能需要 split 来表达更复杂的结构
```

**Deblur-aware 改造**：

不用 deblur render 的梯度做 GCR 统计，而是用 sharp 图的 canonical probe 的梯度：

```
canonical_stats（主导决策） ← 来自 sharp 图的 canonical render probe
blur_stats（辅助参考）     ← 来自训练 pass 的 deblur render（主要是 base_pose_moment）
```

### 8.3 canonical probe 的正确执行方式

**错误做法**（会污染参数梯度）：
```python
# 错误：在主 backward 之后直接再 backward
loss.backward()
proxy_loss.backward()   # ← 这里的梯度会叠加到所有参数上
optimizer.step()        # ← 把 proxy 梯度也更新进去了
```

**正确做法**：
```python
# Step 1：主训练 pass（正常）
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)   # ← 必须先清零

# Step 2：canonical probe（完全隔离）
with torch.enable_grad():
    probe_means2d = torch.zeros(N, 3, device="cuda", requires_grad=True)
    probe_pkg = render_canonical_probe(probe_cam, gaussians, ...)
    proxy_loss = canonical_proxy_loss(probe_pkg)
    
    # 只对 probe_means2d 求梯度，不对任何模型参数求梯度
    canon_grads = torch.autograd.grad(
        outputs=proxy_loss,
        inputs=probe_means2d,   # ← 只求这个输入的梯度
        create_graph=False,
    )[0]   # [N, 3]，普通 tensor，不在计算图里

# 此时 gaussians.parameters() 的所有 .grad 均为 None（已在 zero_grad 清零）
gdags.update_canonical_stats(canon_grads, probe_pkg["visibility_filter"])
# 不调用 optimizer.step()，probe 只用于统计
```

### 8.4 GDAGS Buffer 同步（非常重要）

3DGS 训练中会持续 clone/split/prune，Gaussian 数量不断变化。GDAGS 的所有统计 buffer 长度必须始终等于 `gaussians.get_xyz.shape[0]`，否则 shape mismatch 会导致统计错位或运行时报错。

**必须实现的三个同步方法**：

```python
class DeblurAwareGDAGS:
    
    def on_clone(self, parent_mask: torch.Tensor):
        """
        clone 在数组末尾 append 子点，父点保留。
        parent_mask: bool [N_before]，True 的位置被 clone。
        
        子点初始化：
        - 梯度统计清零（从头开始积累）
        - canonical 统计继承父点的 30%（有参考但不完全继承）
        - age 清零
        - source_type = 3（clone）
        - protect_until_iter 继承父点
        """
        ...

    def on_split(self, parent_mask: torch.Tensor, n_split: int = 2):
        """
        split 在数组末尾 append n_split 个子点。
        父点由后续的 on_prune 删除（调用方负责）。
        
        调用顺序必须是：on_split() → on_prune()
        不能反过来，因为 on_prune 要作用于 append 后的长数组。
        
        子点初始化：
        - 所有统计清零
        - age 清零
        - source_type = 4（split）
        """
        ...

    def on_prune(self, keep_mask: torch.Tensor):
        """
        keep_mask: bool [N_current]，True 的位置保留。
        必须在所有 on_clone/on_split 完成后调用。
        
        带 shape 断言：
        assert keep_mask.shape[0] == self.blur_grad_dir_sum.shape[0]
        """
        ...

    def assert_consistent(self, gaussians, tag: str = ""):
        """调试：检查 buffer 和 Gaussian 数量一致。"""
        N_gs  = gaussians.get_xyz.shape[0]
        N_buf = self.blur_grad_dir_sum.shape[0]
        assert N_gs == N_buf, f"[GDAGS] {tag}: buffer={N_buf} != N_GS={N_gs}"
```

### 8.5 完整 GDAGS 实现

```python
# scene/density_control/gdags_deblur_aware.py

import torch
import torch.nn.functional as F


class DeblurAwareGDAGS:
    """
    Deblur-aware GDAGS 密度控制器。
    
    设计原则：
    1. canonical 统计（来自 sharp 图 probe）主导 split/clone/prune 决策
    2. blur 统计（来自训练 pass 的 base_pose_moment）只作辅助
    3. 所有 buffer 必须和 gaussians.get_xyz 数量保持同步
    4. prune 极保守（gdags_prune_ratio <= 0.01，每次最多 1%）
    5. EAP/Nexus 新点在保护期内绝对不 prune
    
    source_type 编码：
    0 = sfm/original（COLMAP 原始点）
    1 = eap（EAP 初始化增补点）
    2 = nexus（NexusGS-lite 补点）
    3 = clone
    4 = split
    """

    def __init__(self, num_gaussians: int, device: str = "cuda"):
        self.device = device
        self._init_buffers(num_gaussians)

    def _init_buffers(self, N: int):
        """初始化所有统计 buffer。"""
        # blur 统计（来自 base_pose_moment，辅助）
        self.blur_grad_dir_sum      = torch.zeros(N, 3,  device=self.device)
        self.blur_grad_count        = torch.zeros(N,     device=self.device)
        # canonical 统计（来自 sharp probe，主导）
        self.canonical_grad_dir_sum = torch.zeros(N, 3,  device=self.device)
        self.canonical_grad_count   = torch.zeros(N,     device=self.device)
        # 元信息
        self.gaussian_age           = torch.zeros(N,     device=self.device, dtype=torch.long)
        self.source_type            = torch.zeros(N,     device=self.device, dtype=torch.uint8)
        self.protect_until_iter     = torch.zeros(N,     device=self.device, dtype=torch.long)

    # ---- Buffer 同步 ----

    def on_clone(self, parent_mask: torch.Tensor):
        n_new = parent_mask.sum().item()
        if n_new == 0:
            return
        self.blur_grad_dir_sum      = torch.cat([self.blur_grad_dir_sum,      torch.zeros(n_new, 3, device=self.device)], 0)
        self.blur_grad_count        = torch.cat([self.blur_grad_count,        torch.zeros(n_new,    device=self.device)], 0)
        self.canonical_grad_dir_sum = torch.cat([self.canonical_grad_dir_sum, self.canonical_grad_dir_sum[parent_mask] * 0.3], 0)
        self.canonical_grad_count   = torch.cat([self.canonical_grad_count,   self.canonical_grad_count[parent_mask]   * 0.3], 0)
        self.gaussian_age           = torch.cat([self.gaussian_age,           torch.zeros(n_new, device=self.device, dtype=torch.long)], 0)
        self.source_type            = torch.cat([self.source_type,            torch.full((n_new,), 3, device=self.device, dtype=torch.uint8)], 0)
        self.protect_until_iter     = torch.cat([self.protect_until_iter,     self.protect_until_iter[parent_mask].clone()], 0)

    def on_split(self, parent_mask: torch.Tensor, n_split: int = 2):
        n_new = parent_mask.sum().item() * n_split
        if n_new == 0:
            return
        self.blur_grad_dir_sum      = torch.cat([self.blur_grad_dir_sum,      torch.zeros(n_new, 3, device=self.device)], 0)
        self.blur_grad_count        = torch.cat([self.blur_grad_count,        torch.zeros(n_new,    device=self.device)], 0)
        self.canonical_grad_dir_sum = torch.cat([self.canonical_grad_dir_sum, torch.zeros(n_new, 3, device=self.device)], 0)
        self.canonical_grad_count   = torch.cat([self.canonical_grad_count,   torch.zeros(n_new,    device=self.device)], 0)
        self.gaussian_age           = torch.cat([self.gaussian_age,           torch.zeros(n_new, device=self.device, dtype=torch.long)], 0)
        self.source_type            = torch.cat([self.source_type,            torch.full((n_new,), 4, device=self.device, dtype=torch.uint8)], 0)
        self.protect_until_iter     = torch.cat([self.protect_until_iter,     self.protect_until_iter[parent_mask].repeat_interleave(n_split)], 0)

    def on_prune(self, keep_mask: torch.Tensor):
        N_buf = self.blur_grad_dir_sum.shape[0]
        assert keep_mask.shape[0] == N_buf, \
            f"[GDAGS] on_prune: keep_mask.shape={keep_mask.shape[0]} != buffer={N_buf}. " \
            f"请确认 on_clone/on_split 已在 on_prune 之前调用。"
        self.blur_grad_dir_sum      = self.blur_grad_dir_sum[keep_mask]
        self.blur_grad_count        = self.blur_grad_count[keep_mask]
        self.canonical_grad_dir_sum = self.canonical_grad_dir_sum[keep_mask]
        self.canonical_grad_count   = self.canonical_grad_count[keep_mask]
        self.gaussian_age           = self.gaussian_age[keep_mask]
        self.source_type            = self.source_type[keep_mask]
        self.protect_until_iter     = self.protect_until_iter[keep_mask]

    def assert_consistent(self, gaussians, tag: str = ""):
        N_gs  = gaussians.get_xyz.shape[0]
        N_buf = self.blur_grad_dir_sum.shape[0]
        assert N_gs == N_buf, f"[GDAGS] {tag} buffer={N_buf} != N_GS={N_gs}"

    def register_new_points(self, n_new: int, source: int, protect_until: int):
        """
        EAP/Nexus 新点加入时调用。
        source: 1=eap, 2=nexus
        protect_until: 保护期结束的 iteration
        """
        self.blur_grad_dir_sum      = torch.cat([self.blur_grad_dir_sum,      torch.zeros(n_new, 3, device=self.device)], 0)
        self.blur_grad_count        = torch.cat([self.blur_grad_count,        torch.zeros(n_new,    device=self.device)], 0)
        self.canonical_grad_dir_sum = torch.cat([self.canonical_grad_dir_sum, torch.zeros(n_new, 3, device=self.device)], 0)
        self.canonical_grad_count   = torch.cat([self.canonical_grad_count,   torch.zeros(n_new,    device=self.device)], 0)
        self.gaussian_age           = torch.cat([self.gaussian_age,           torch.zeros(n_new, device=self.device, dtype=torch.long)], 0)
        self.source_type            = torch.cat([self.source_type,            torch.full((n_new,), source, device=self.device, dtype=torch.uint8)], 0)
        self.protect_until_iter     = torch.cat([self.protect_until_iter,     torch.full((n_new,), protect_until, device=self.device, dtype=torch.long)], 0)

    # ---- 统计更新 ----

    def update_blur_stats(self, viewspace_points: torch.Tensor,
                           visibility_filter: torch.Tensor, blur_type: int):
        """
        收集 blur 统计。
        
        调用条件：
        - viewspace_points 必须是 base_pose_means2d（motion 图的原始位置 moment）
        - 不是 all_viewspace 列表，不是 sharp render 的 screenspace_points
        
        原因：motion 图有 M+1 组梯度，只取 base_pose_moment 的梯度，
        避免偏移 moment 的梯度人为放大 GCR 信号。
        """
        if not isinstance(viewspace_points, torch.Tensor):
            return
        if viewspace_points.grad is None:
            return
        grad = viewspace_points.grad[visibility_filter]
        if grad.shape[0] == 0:
            return
        grad_norm = F.normalize(grad, dim=-1, eps=1e-8)
        self.blur_grad_dir_sum[visibility_filter] += grad_norm
        self.blur_grad_count[visibility_filter]   += 1

    def update_canonical_stats(self, canonical_grads: torch.Tensor,
                                 visibility_filter: torch.Tensor):
        """
        收集 canonical 统计。
        
        调用条件：只由 sharp 图的 canonical probe 调用。
        canonical_grads 来自 torch.autograd.grad，是普通 tensor，不在计算图里。
        """
        if canonical_grads is None:
            return
        visible_grads = canonical_grads[visibility_filter]
        if visible_grads.shape[0] == 0:
            return
        grad_norm = F.normalize(visible_grads, dim=-1, eps=1e-8)
        self.canonical_grad_dir_sum[visibility_filter] += grad_norm
        self.canonical_grad_count[visibility_filter]   += 1

    def update_canonical_stats_no_grad(self, radii: torch.Tensor,
                                         visibility_filter: torch.Tensor):
        """
        无梯度版 canonical 统计（sharp 图 < 5 张时的备选方案）。
        用 visibility 代替梯度方向，只统计哪些 Gaussian 被看到。
        """
        self.canonical_grad_count[visibility_filter] += 1

    # ---- GCR 计算 ----

    def compute_gcr(self, use_canonical: bool = True) -> torch.Tensor:
        """
        GCR = ||sum(normalized_grad)|| / count
        
        取值 [0, 1]：
          接近 1：梯度方向一致，Gaussian 正在稳定拟合某个细节
          接近 0：梯度方向冲突，该 Gaussian 可能需要 split 来表达更复杂的结构
        """
        if use_canonical:
            count = self.canonical_grad_count.clamp(min=1.0)
            return self.canonical_grad_dir_sum.norm(dim=-1) / count
        else:
            count = self.blur_grad_count.clamp(min=1.0)
            return self.blur_grad_dir_sum.norm(dim=-1) / count

    # ---- 密度决策 ----

    def decide_densify_prune(self, gaussians, iteration: int, opt):
        """
        返回 (split_mask, clone_mask, prune_mask)，bool tensor [N]。
        
        决策原则：
        - canonical 统计主导
        - prune 极保守（min_age=2000, prune_ratio=0.01）
        - EAP 点（source=1）前 5000 iter 绝对不 prune
        - Nexus 点（source=2）保护期内不 prune
        - 不因 deblur gradient 高低来决定 split/prune
        """
        gcr     = self.compute_gcr(use_canonical=True)
        opacity = gaussians.get_opacity.squeeze()
        N       = gaussians.get_xyz.shape[0]

        not_protected     = (iteration > self.protect_until_iter)
        age_ok            = (self.gaussian_age > 500)
        has_canon_stats   = (self.canonical_grad_count >= 3)

        # split：GCR 低（梯度冲突）
        split_mask = (
            (gcr < opt.gdags_gcr_split_threshold)
            & (opacity > 0.1)
            & age_ok & not_protected & has_canon_stats
        )

        # clone：GCR 高（梯度一致）、Gaussian 较小
        max_scale = gaussians.get_scaling.max(dim=-1).values
        clone_mask = (
            (gcr > opt.gdags_gcr_clone_threshold)
            & (opacity > 0.05)
            & (max_scale < gaussians.scene_extent * 0.01)
            & age_ok & has_canon_stats
        )

        # prune：极保守
        eap_protected   = (self.source_type == 1) & (iteration < 5000)
        nexus_protected = (self.source_type == 2) & (~not_protected)

        prune_mask = (
            (opacity < opt.opacity_threshold)
            & (self.canonical_grad_count >= 1)
            & (self.gaussian_age > opt.gdags_min_age_for_prune)
            & not_protected & ~eap_protected & ~nexus_protected
        )

        # 限制每次 prune 的比例（最多 gdags_prune_ratio * N）
        if prune_mask.sum() > N * opt.gdags_prune_ratio:
            opacity_vals = opacity.clone()
            opacity_vals[~prune_mask] = 1.0
            k = int(N * opt.gdags_prune_ratio)
            threshold = opacity_vals.topk(k, largest=False).values[-1]
            prune_mask = prune_mask & (opacity < threshold)

        return split_mask, clone_mask, prune_mask

    def tick_age(self):
        """每个 iteration 结束时调用，递增所有 Gaussian 的 age。"""
        self.gaussian_age += 1
```

### 8.6 densify/prune 包装函数（在 gdags_integration.py 里）

```python
# scene/density_control/gdags_integration.py

def gdags_aware_densify_and_prune(gaussians, gdags, iteration, opt):
    """
    替代原始 densify_and_prune 的包装函数。
    在每次 clone/split/prune 后立即同步 GDAGS buffer。
    
    执行顺序（不能调换）：
    1. clone → 末尾 append → on_clone
    2. split → 末尾 append 子点 → on_split
    3. prune → 删除父点（包括 split 的父点）→ on_prune
    4. assert_consistent 验证
    5. tick_age
    """
    split_mask, clone_mask, prune_mask = gdags.decide_densify_prune(
        gaussians, iteration, opt
    )

    # 1. Clone
    if clone_mask.sum() > 0:
        gaussians.densification_postfix_clone(clone_mask)
        gdags.on_clone(clone_mask)
        gdags.assert_consistent(gaussians, f"after_clone_iter{iteration}")

    # 2. Split（分两步：先 append 子点，再 prune 父点）
    if split_mask.sum() > 0:
        gaussians.densification_postfix_split(split_mask)
        gdags.on_split(split_mask, n_split=2)
        # split 父点删除
        n_new = split_mask.sum().item() * 2
        N_old = gaussians.get_xyz.shape[0] - n_new
        split_keep = torch.ones(N_old, dtype=torch.bool, device="cuda")
        split_keep[split_mask] = False
        keep_extended = torch.cat([split_keep, torch.ones(n_new, dtype=torch.bool, device="cuda")])
        gaussians.prune_points(split_keep)
        gdags.on_prune(keep_extended)
        gdags.assert_consistent(gaussians, f"after_split_iter{iteration}")

    # 3. Prune
    if prune_mask.sum() > 0:
        # prune_mask 对应当前 N，需要重新计算（split 后 N 变了）
        _, _, prune_mask = gdags.decide_densify_prune(gaussians, iteration, opt)
        keep = ~prune_mask
        gaussians.prune_points(keep)
        gdags.on_prune(keep)
        gdags.assert_consistent(gaussians, f"after_prune_iter{iteration}")

    gdags.tick_age()
```

---

## 9. 模块设计：NexusGS-lite 稀疏补点（Phase 6）

### 9.1 为什么需要这个

EAP 在初始化阶段增补点，但训练过程中可能仍然出现新的稀疏区域（例如：某些区域的 Gaussian 被 prune 掉后没有有效补充）。NexusGS-lite 在训练中后期，用高置信度的几何线索为这些区域补充 Gaussian。

**为什么叫 "lite"**：
完整的 NexusGS 使用全局 dense depth 重建，计算量很大。本项目只做轻量版：
- 只在 EAP 后仍然稀疏的区域工作
- 只使用高置信度图片对（排除模糊、过曝、低重叠的对）
- 只在 flow + epipolar + depth 三重一致的候选点上添加

### 9.2 接入时机

```python
nexus_start_iter = int(opt.iterations * opt.nexus_start_iter_ratio)
# 默认 0.35 * total_iters，等 GDAGS 和主 3DGS 都稳定后再补点
```

不要在 0.12T 前的早期大量加点（几何还不稳定，加点可能乱飘）。

### 9.3 高置信度评分

只有总置信度 > `nexus_conf_add_thresh` 的候选点才加入：

```
conf = pair_conf × corr_conf × depth_conf × need_conf × deblur_safe_conf

pair_conf:        图片对的质量（清晰度、曝光、baseline、overlap）
corr_conf:        匹配点的质量（flow fb-check、epipolar距离、NCC）
depth_conf:       深度的可靠性（正深度、reprojection error、多视角一致）
need_conf:        该区域是否真的需要补点（局部稀疏、canonical coverage 不足）
deblur_safe_conf: 该区域的 deblur 状态是否稳定（pos_delta 不异常）
```

### 9.4 新点初始化

```python
# 新点属性
source_type     = "nexus"
opacity_init    = 0.02    # 低初始 opacity，让 Gaussian 先稳定
protect_until   = current_iter + 1500  # 保护 1500 iter 不被 prune
xyz_lr_scale    = 0.3     # 初期位置学习率较低，慢慢收敛
```

---

## 10. 关键问题修正（6 项工程细节）

### 10.1 问题：Canonical probe 污染主优化梯度

**现象**：如果在主 `loss.backward()` 之后、`optimizer.step()` 之前执行 `proxy_loss.backward()`，probe 的梯度会累加到所有参数的 `.grad` 上，optimizer.step() 会把这些梯度一起更新，导致 canonical probe "意外地"修改了 Gaussian 的几何。

**修正**：参见第 8.3 节的正确做法。关键：
1. 主训练：backward → step → **zero_grad**（先清零）
2. probe：torch.autograd.grad（只取输入的梯度，不流回参数）
3. probe 后：不调用 optimizer.step()

### 10.2 问题：GDAGS buffer 随 densify/prune 不同步

**现象**：clone/split/prune 操作改变了 Gaussian 数量，但 GDAGS 的 buffer 还是旧长度，indexing 错位导致梯度统计张冠李戴，或运行时 shape mismatch 报错。

**修正**：参见第 8.5 节的 `on_clone / on_split / on_prune` 方法，以及第 8.6 节的 `gdags_aware_densify_and_prune` 调用顺序。

### 10.3 问题：motion 分支的 "base_pose_moment" 命名混乱

**现象**：motion 分支的循环 `for i in range(M+1)` 中，原始位置渲染是 `i==M`（最后一个），但文档里容易错误地称为 "moment-0"，导致实现时取了 `all_viewspace[0]`（第一个偏移 moment）的梯度用于 GDAGS。

**修正**：
- 代码中用 `is_base_pose = (i == M)` 明确标注
- 返回字段命名为 `base_pose_means2d / base_pose_visibility / base_pose_radii`
- 禁止使用 "moment-0" 这个名称

```python
# 正确：
is_base_pose = (i == M)   # M 是最后一个索引
if is_base_pose:
    t_pos = means3D          # 无偏移，原始位置
    base_pose_means2d = sp   # 供 GDAGS 使用

# 错误的命名（禁止）：
# moment_0 / first_moment / moment-0 ← 这些名字和代码语义不符
```

### 10.4 问题：defocus 分支四元数乘法不合法

**现象**：`new_rotations = rotations * r_delta` 逐元素乘后，结果不再是单位四元数（模不等于 1），rasterizer 内部计算旋转矩阵时会出错，可能导致渲染崩溃或 NaN。

**修正**：
```python
# defocus 分支：默认关闭 rotation delta（只学 scale）
# 如果开启 rotation delta，必须归一化：
if r_delta is not None:
    new_rotations = torch.nn.functional.normalize(
        rotations * r_delta, dim=-1
    )
else:
    new_rotations = rotations  # 不改 rotation

# motion 分支：每个 moment 都要归一化
t_r = torch.nn.functional.normalize(rotations * r_delta_3d[..., i], dim=-1)
```

### 10.5 问题：Luminance 参数过多导致抢解释权

**现象**：Per-channel RGB scale+bias（6参数）、3x3 matrix（12参数）等配置自由度过强，可能把颜色误差、局部模糊残差都"吸收"进来，让 GTnet 的 blur_code 学不到干净的模糊特征。

**修正**：只用 2 个标量（scalar gain + scalar bias）。参见第 5 章 `PerImageExposureModel`。
- 禁止 per-channel
- 禁止 color matrix
- 禁止 tone curve
- hard clamp 防止越界

### 10.6 问题：loss 权重改变了正则项的相对强度

**现象**：原始写法是：
```python
loss = photo_loss * sharp_weight + lambda_code * z_reg + lambda_delta * delta_reg
```
对 sharp 图，`sharp_weight=2.0` 放大了 `photo_loss`，但 `lambda_code`、`lambda_delta` 没有跟着放大。等价于：sharp 图训练时，正则项相对弱了一半，可能导致 blur_code 在 sharp 图上过拟合（因为正则约束相对变弱了）。

**修正**：数据项和正则项严格分离，blur_weight 只作用于数据项：

```python
photo_loss_raw      = compute_photo_loss(image_for_loss, gt_image)
photo_loss_weighted = photo_loss_raw * blur_weight    # 只放大数据项

reg_loss = (lambda_code * z_reg
          + lambda_delta * delta_reg
          + exposure_reg)                              # 正则项不乘 blur_weight

loss_total = photo_loss_weighted + reg_loss
```

同时分别记录 `photo_loss_raw` 和 `photo_loss_weighted`，方便排查训练是否被某种图主导。

---

## 11. Render 函数完整实现

```python
# gaussian_renderer/__init__.py
# 在原始代码基础上修改，保留原始 render() 签名，内部按 blur_type 路由

import math
import torch
import torch.nn.functional as F


def render(
    viewpoint_camera,
    pc,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    deblur: bool = False,
    blur_type: int = 0,           # 新增：0=sharp, 1=motion, 2=defocus
    image_id: int = None,         # 新增：当前图的全局 ID
    lambda_s: float = 0.01,
    lambda_p: float = 0.01,
    max_clamp: float = 1.1,
    backend: str = "original",    # 新增："original" | "gsplat"
):
    """
    统一 render 入口。
    
    路由逻辑：
    - deblur=False 或 blur_type=0：sharp render（不调用任何 GTnet）
    - blur_type=1：motion render（调用 motion_GTnet，多 moment 平均）
    - blur_type=2：defocus render（调用 defocus_GTnet，scale 变换）
    
    backend 说明：
    - motion/defocus 分支强制使用 original（第一版不对 deblur 路径用 gsplat）
    - sharp 分支可以用 gsplat
    
    返回字段（所有路径都有）：
    - render:                渲染图 [3, H, W]
    - viewspace_points:      screenspace_points（sharp/defocus 是 tensor，motion 是 list）
    - visibility_filter:     radii > 0（同上）
    - radii:                 Gaussian 在屏幕上的半径
    - blur_code:             GTnet 的 blur embedding（sharp 时为 None）
    - delta_reg:             GTnet 输出的正则项（sharp 时为 0）
    - base_pose_means2d:     motion 分支的原始位置 moment（供 GDAGS 使用）
    - base_pose_visibility:  同上
    """

    # ---- 公共前处理（与原始完全相同，不改动）----
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    ) + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
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
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D    = pc.get_xyz
    means2D    = screenspace_points
    opacity    = pc.get_opacity
    scales     = pc.get_scaling
    rotations  = pc.get_rotation

    shs            = None
    colors_precomp = None
    cov3D_precomp  = None

    if pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp   = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb   = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        shs = pc.get_features

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)

    # motion/defocus 分支强制使用 original backend（第一版）
    actual_backend = backend if (not deblur or blur_type == 0) else "original"

    # 判断是否使用双分支模式
    use_dual_branch = hasattr(pc, 'motion_GTnet') and hasattr(pc, 'defocus_GTnet')

    # ---- 路由 ----
    if not deblur or blur_type == 0:
        return _render_sharp(
            means3D, means2D, shs, colors_precomp, opacity,
            scales, rotations, cov3D_precomp, rasterizer, actual_backend
        )
    elif blur_type == 1:
        gtnet = pc.motion_GTnet if use_dual_branch else pc.GTerr
        return _render_motion(
            pc, gtnet, means3D, means2D, shs, colors_precomp, opacity,
            scales, rotations, cov3D_precomp, rasterizer, "original",
            viewpoint_camera, image_id, lambda_s, lambda_p, max_clamp
        )
    elif blur_type == 2:
        gtnet = pc.defocus_GTnet if use_dual_branch else pc.GTerr
        return _render_defocus(
            pc, gtnet, means3D, means2D, shs, colors_precomp, opacity,
            scales, rotations, cov3D_precomp, rasterizer, "original",
            viewpoint_camera, image_id, lambda_s, max_clamp
        )
    else:
        raise ValueError(f"Unknown blur_type={blur_type}")


def _render_sharp(means3D, means2D, shs, colors_precomp, opacity,
                   scales, rotations, cov3D_precomp, rasterizer, backend):
    """
    Sharp render：原始 not-deblur 分支，完全不动。
    不调用任何 GTnet，直接 rasterize。
    """
    img, radii = _call_rasterizer(
        rasterizer, backend,
        means3D, means2D, shs, colors_precomp, opacity,
        scales, rotations, cov3D_precomp
    )
    return {
        "render":             img,
        "viewspace_points":   means2D,
        "visibility_filter":  radii > 0,
        "radii":              radii,
        "blur_code":          None,
        "delta_reg":          torch.tensor(0.0, device=means3D.device),
        "base_pose_means2d":  means2D,
        "base_pose_visibility": radii > 0,
    }


def _render_defocus(pc, gtnet, means3D, means2D, shs, colors_precomp, opacity,
                     scales, rotations, cov3D_precomp, rasterizer, backend,
                     viewpoint_camera, image_id, lambda_s, max_clamp):
    """
    Defocus blur render：复用原始 use_pos=False 分支逻辑。
    只修改 scale（rotation 默认关闭，见问题 10.4 修正）。
    """
    dir_pp   = means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1)
    viewdirs = dir_pp / dir_pp.norm(dim=1, keepdim=True)

    s_delta, r_delta, _, z = gtnet(
        means3D.detach(), scales.detach(), rotations.detach(), viewdirs, image_id
    )

    # scale 变换
    s_clamped  = torch.clamp(lambda_s * s_delta + (1 - lambda_s), min=1.0, max=max_clamp)
    new_scales = scales * s_clamped

    # rotation 变换（默认 r_delta=None，即不修改 rotation）
    if r_delta is not None:
        # 修正 10.4：四元数乘法后必须归一化
        new_rotations = F.normalize(rotations * r_delta, dim=-1)
    else:
        new_rotations = rotations

    img, radii = _call_rasterizer(
        rasterizer, backend,
        means3D, means2D, shs, colors_precomp, opacity,
        new_scales, new_rotations, cov3D_precomp
    )

    delta_reg = (s_clamped - 1.0).abs().mean()
    if r_delta is not None:
        delta_reg = delta_reg + (r_delta - 1.0).abs().mean()

    return {
        "render":             img,
        "viewspace_points":   means2D,
        "visibility_filter":  radii > 0,
        "radii":              radii,
        "blur_code":          z,
        "delta_reg":          delta_reg,
        "base_pose_means2d":  means2D,    # defocus 只有一次渲染，直接作为 base_pose
        "base_pose_visibility": radii > 0,
    }


def _render_motion(pc, gtnet, means3D, means2D, shs, colors_precomp, opacity,
                    scales, rotations, cov3D_precomp, rasterizer, backend,
                    viewpoint_camera, image_id, lambda_s, lambda_p, max_clamp):
    """
    Motion blur render：复用原始 use_pos=True 分支逻辑。
    
    重要命名规则（见问题 10.3 修正）：
      i = 0 .. M-1  →  offset_moment（有位置偏移）
      i = M         →  base_pose_moment（原始位置，无偏移）
    
    GDAGS 只使用 base_pose_moment（i==M）的 viewspace gradient。
    不使用 offset_moment 的梯度（避免偏移方向影响 GCR 统计）。
    """
    M = gtnet.num_moments

    dir_pp   = means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1)
    viewdirs = dir_pp / dir_pp.norm(dim=1, keepdim=True)

    s_delta, r_delta, p_delta, z = gtnet(
        means3D.detach(), scales.detach(), rotations.detach(), viewdirs, image_id
    )

    s_delta = torch.clamp(lambda_s * s_delta + (1 - lambda_s), min=1.0, max=max_clamp)
    r_delta = torch.clamp(lambda_s * r_delta + (1 - lambda_s), min=1.0, max=max_clamp)
    p_delta = lambda_p * p_delta

    p_delta_3d = p_delta.view(-1, 3, M)
    s_delta_3d = s_delta.view(-1, 3, M + 1)
    r_delta_3d = r_delta.view(-1, 4, M + 1)

    renders           = []
    all_viewspace     = []
    all_visibility    = []
    all_radii         = []
    base_pose_means2d    = None
    base_pose_visibility = None
    base_pose_radii      = None

    for i in range(M + 1):
        sp = torch.zeros_like(means3D, requires_grad=True, device="cuda")
        try:
            sp.retain_grad()
        except Exception:
            pass

        # base_pose_moment：原始位置，无偏移（i == M）
        is_base_pose = (i == M)

        if is_base_pose:
            t_pos = means3D
            t_s   = scales * s_delta_3d[..., -1]
            t_r   = F.normalize(rotations * r_delta_3d[..., -1], dim=-1)  # 修正 10.4
        else:
            t_pos = means3D + p_delta_3d[..., i]
            t_s   = scales * s_delta_3d[..., i]
            t_r   = F.normalize(rotations * r_delta_3d[..., i], dim=-1)   # 修正 10.4

        img, radii = _call_rasterizer(
            rasterizer, backend,
            t_pos, sp, shs, colors_precomp, opacity, t_s, t_r, cov3D_precomp
        )

        if is_base_pose:
            base_pose_means2d    = sp
            base_pose_visibility = radii > 0
            base_pose_radii      = radii

        renders.append(img)
        all_viewspace.append(sp)
        all_visibility.append(radii > 0)
        all_radii.append(radii)

    render_blur = sum(renders) / len(renders)
    delta_reg   = (p_delta.abs().mean()
                   + (s_delta - 1.0).abs().mean()
                   + (r_delta - 1.0).abs().mean())

    return {
        "render":             render_blur,
        "viewspace_points":   all_viewspace,
        "visibility_filter":  all_visibility,
        "radii":              all_radii,
        "blur_code":          z,
        "delta_reg":          delta_reg,
        # GDAGS 专用：base_pose_moment（原始位置，i==M）
        "base_pose_means2d":    base_pose_means2d,
        "base_pose_visibility": base_pose_visibility,
        "base_pose_radii":      base_pose_radii,
    }


def _call_rasterizer(rasterizer, backend, means3D, means2D, shs, colors_precomp,
                      opacity, scales, rotations, cov3D_precomp):
    """统一 rasterizer 调用层。"""
    if backend == "original":
        return rasterizer(
            means3D=means3D, means2D=means2D, shs=shs,
            colors_precomp=colors_precomp, opacities=opacity,
            scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
        )
    elif backend == "gsplat":
        from gaussian_renderer.backends.gsplat_backend import gsplat_rasterize
        return gsplat_rasterize(
            means3D, means2D, shs, colors_precomp, opacity,
            scales, rotations, cov3D_precomp, rasterizer.raster_settings
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def render_canonical_probe(viewpoint_camera, pc, pipe, bg_color, probe_means2d):
    """
    GDAGS canonical probe 专用的 render 函数。
    
    和主 render 的区别：
    - 使用调用方传入的 probe_means2d（不是自己创建的 screenspace_points）
    - 永远不调用 GTnet（deblur=False，blur_type=0）
    - 永远使用 original backend
    - 只能用 sharp 图（调用方保证）
    
    probe_means2d 由外部创建（requires_grad=True），
    torch.autograd.grad 对它求梯度，不流回任何模型参数。
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=bg_color, scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False, debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    img, radii = rasterizer(
        means3D=pc.get_xyz, means2D=probe_means2d, shs=pc.get_features,
        colors_precomp=None, opacities=pc.get_opacity,
        scales=pc.get_scaling, rotations=pc.get_rotation, cov3D_precomp=None,
    )
    return {"render": img, "visibility_filter": radii > 0, "radii": radii}


def canonical_proxy_loss(probe_pkg: dict) -> torch.Tensor:
    """
    Canonical probe 的 proxy loss。
    
    第一版：用 coverage（alpha sum 均值）作为 proxy。
    目的：鼓励 Gaussian 覆盖视野，抑制空洞区域，让 coverage 高的 Gaussian 有梯度信号。
    
    不使用真实 GT 图像监督，原因：
    - GT 图可能是 blurry 的（motion/defocus 图作为 probe）
    - 把 blurry 信号引入 canonical 统计会污染 GDAGS 的密度决策
    """
    return -probe_pkg["render"].mean()   # 最大化 coverage
```

---

## 12. train.py 主循环完整实现

```python
# train.py（仅展示改动部分，原始代码未改动的部分以注释标注保留位置）

import os, json, random, math
import torch
import torch.nn.functional as F
from utils.auto_blur_detect import auto_detect_blur_type
from scene.luminance_model import PerImageExposureModel
from scene.density_control.gdags_deblur_aware import DeblurAwareGDAGS
from scene.density_control.gdags_integration import gdags_aware_densify_and_prune
from gaussian_renderer import render, render_canonical_probe, canonical_proxy_loss


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from):

    # ================================================================
    # [原始代码] 基础初始化（保留，不改动）
    # ================================================================
    # first_iter = 0
    # tb_writer = prepare_output_and_logger(dataset)
    # gaussians = GaussianModel(dataset.sh_degree)
    # scene = Scene(dataset, gaussians)
    # gaussians.training_setup(opt)
    # ... 原始代码 ...

    # ================================================================
    # [新增] Step 1：加载或自动检测 blur 标签
    # ================================================================
    blur_map = {}
    train_cameras = scene.getTrainCameras()

    if opt.mixed_blur:
        label_path = os.path.join(dataset.source_path, "blur_labels.json")
        if os.path.exists(label_path):
            with open(label_path) as f:
                data = json.load(f)
            lbl2int = {"sharp": 0, "motion": 1, "defocus": 2}
            blur_map = {k: lbl2int[v["blur_type"]] for k, v in data["images"].items()}
            print(f"[Blur Labels] loaded {len(blur_map)} from {label_path}")
        else:
            print(f"[Blur Labels] not found at {label_path}, using auto-detection")

    # ================================================================
    # [新增] Step 2：给每个 camera 设置 image_id 和 blur_type
    # ================================================================
    for idx, cam in enumerate(train_cameras):
        cam.image_id = idx   # 连续 ID，从 0 开始
        if cam.image_name in blur_map:
            cam.blur_type = blur_map[cam.image_name]
            cam.sharpness_score = 1.0  # 来自标注，不需要检测
        elif opt.mixed_blur:
            bt, score = auto_detect_blur_type(cam.original_image)
            cam.blur_type      = bt
            cam.sharpness_score = score
        else:
            # 兼容原始模式：全局统一 blur_type
            cam.blur_type      = 1 if opt.use_pos else 2
            cam.sharpness_score = 1.0
        cam.image_id = idx

    # 按 blur_type 分组
    sharp_cameras   = [c for c in train_cameras if c.blur_type == 0]
    motion_cameras  = [c for c in train_cameras if c.blur_type == 1]
    defocus_cameras = [c for c in train_cameras if c.blur_type == 2]
    print(f"[Dataset Split] N={len(train_cameras)} | "
          f"sharp={len(sharp_cameras)} | motion={len(motion_cameras)} | "
          f"defocus={len(defocus_cameras)}")

    if len(sharp_cameras) == 0 and opt.gdags_enable:
        print("[WARNING] 没有 sharp 图，GDAGS canonical probe 将使用 no_grad 模式")

    # ================================================================
    # [新增] Step 3：创建 GTnet（在原始 gaussians.training_setup 之后调用）
    # ================================================================
    gaussians.create_deblur_nets(len(train_cameras), opt)
    # 注意：需要在 gaussians.training_setup(opt) 里把
    # gaussians.get_deblur_optimizer_params(opt.gtnet_lr) 加入 optimizer
    # 具体见 scene/gaussian_model.py 的 training_setup 修改

    # ================================================================
    # [新增] Step 4：曝光校正模型
    # ================================================================
    exposure_model = None
    exp_optimizer  = None
    if opt.luminance_enable:
        exposure_model = PerImageExposureModel(num_images=len(train_cameras)).cuda()
        exp_optimizer  = torch.optim.Adam(
            exposure_model.parameters(),
            lr=opt.luminance_lr, betas=(0.9, 0.999)
        )
        print(f"[Exposure] PerImageExposureModel enabled, "
              f"start_iter={opt.luminance_start_iter}")

    # ================================================================
    # [新增] Step 5：GDAGS
    # ================================================================
    gdags = None
    gdags_probe_use_grad = False
    if opt.gdags_enable:
        gdags = DeblurAwareGDAGS(
            num_gaussians=gaussians.get_xyz.shape[0], device="cuda"
        )
        gdags_probe_use_grad = len(sharp_cameras) >= 5
        print(f"[GDAGS] enabled, probe_use_grad={gdags_probe_use_grad}")

    # ================================================================
    # 训练循环
    # ================================================================
    for iteration in range(first_iter + 1, opt.iterations + 1):

        # [原始代码] 学习率调度、SH 升级等保留

        # ---- 采样 ----
        viewpoint_cam = sample_camera(
            iteration, train_cameras, sharp_cameras,
            motion_cameras, defocus_cameras, opt
        )

        blur_type  = viewpoint_cam.blur_type if opt.mixed_blur else (
            1 if (deblur and opt.use_pos) else (2 if deblur else 0)
        )
        image_id   = viewpoint_cam.image_id
        use_deblur = deblur and (blur_type != 0)

        # ---- Render ----
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, background,
            deblur=use_deblur,
            blur_type=blur_type,
            image_id=image_id,
            lambda_s=opt.lambda_s,
            lambda_p=opt.lambda_p,
            max_clamp=opt.max_clamp,
            backend=opt.renderer_backend if blur_type == 0 else opt.renderer_backend_deblur,
        )
        image = render_pkg["render"]

        # ---- Loss 计算（问题 10.6 修正：数据项和正则项严格分离）----
        loss, loss_components = compute_loss(
            render_pkg, viewpoint_cam, exposure_model, opt, iteration
        )

        # ---- 主 backward + step ----
        loss.backward()

        with torch.no_grad():
            # [原始代码] 记录 visibility 统计，更新 max_radii2D 等

            # GDAGS blur 统计（在 step 之前收集，梯度还在）
            if opt.gdags_enable and gdags is not None and iteration >= opt.gdags_start_iter:
                vsp = render_pkg.get("base_pose_means2d")
                vis = render_pkg.get("base_pose_visibility")
                if vsp is not None and vsp.grad is not None:
                    gdags.update_blur_stats(vsp, vis, blur_type)

        # optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)   # ← 必须在 probe 前清零（问题 10.1 修正）

        # 曝光校正独立 step
        if exp_optimizer is not None and iteration >= opt.luminance_start_iter:
            exp_optimizer.step()
            exp_optimizer.zero_grad(set_to_none=True)

        # ================================================================
        # [新增] GDAGS canonical probe（在主 zero_grad 之后）
        # 问题 10.1 修正：完全隔离，不污染任何参数梯度
        # ================================================================
        if (opt.gdags_enable and gdags is not None
                and iteration >= opt.gdags_start_iter
                and iteration % opt.gdags_probe_interval == 0):

            if gdags_probe_use_grad and len(sharp_cameras) > 0:
                # 方案 A：用 autograd.grad 精确取 viewspace 梯度
                probe_cam = random.choice(sharp_cameras)
                with torch.enable_grad():
                    probe_means2d = torch.zeros(
                        gaussians.get_xyz.shape[0], 3,
                        device="cuda", dtype=torch.float32, requires_grad=True
                    )
                    probe_pkg = render_canonical_probe(
                        probe_cam, gaussians, pipe, background, probe_means2d
                    )
                    proxy_loss = canonical_proxy_loss(probe_pkg)
                    # 只对 probe_means2d 求梯度，不流回模型参数
                    canon_grads = torch.autograd.grad(
                        outputs=proxy_loss,
                        inputs=probe_means2d,
                        create_graph=False, retain_graph=False,
                    )[0]
                # 验证参数梯度确实为 None
                assert gaussians.get_xyz.grad is None, \
                    "[GDAGS] probe 污染了 xyz 梯度，检查 zero_grad 调用顺序"
                gdags.update_canonical_stats(canon_grads, probe_pkg["visibility_filter"])
            else:
                # 方案 B：no_grad 模式（sharp 图不足时）
                if len(sharp_cameras) > 0:
                    probe_cam = random.choice(sharp_cameras)
                    with torch.no_grad():
                        probe_means2d = torch.zeros(gaussians.get_xyz.shape[0], 3, device="cuda")
                        probe_pkg = render_canonical_probe(
                            probe_cam, gaussians, pipe, background, probe_means2d
                        )
                    gdags.update_canonical_stats_no_grad(
                        probe_pkg["radii"], probe_pkg["visibility_filter"]
                    )

        # ================================================================
        # 密度控制
        # ================================================================
        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:

            if not opt.gdags_enable:
                # [原始代码] 原始 densify_and_prune 逻辑，完全保留
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune(
                    opt.densify_grad_threshold, 0.005,
                    scene.cameras_extent, size_threshold
                )
            else:
                # GDAGS 密度控制
                gdags_aware_densify_and_prune(gaussians, gdags, iteration, opt)
                gdags.assert_consistent(gaussians, tag=f"iter{iteration}")

        # ================================================================
        # 日志
        # ================================================================
        if iteration % 100 == 0:
            log_training(iteration, gaussians, loss_components, gdags, opt)

        if iteration % 500 == 0 and exposure_model is not None:
            stats = exposure_model.get_stats()
            print(f"[Exposure iter {iteration}] "
                  f"gain={stats['gain_mean']:.3f}±{stats['gain_std']:.3f} "
                  f"bias_max={stats['bias_max']:.4f}")
            if stats['gain_max'] > 1.30 or stats['gain_min'] < 0.77:
                print("[WARNING] gain 接近 clamp 边界，检查是否有强曝光差异图片")
            if stats['bias_max'] > 0.04:
                print("[WARNING] bias 偏大，检查是否有过曝或欠曝图片")

        # [原始代码] 保存 checkpoint、evaluation 等逻辑保留


def compute_loss(render_pkg, viewpoint_cam, exposure_model, opt, iteration):
    """
    标准化 loss 计算。
    
    重要：数据项（photo_loss）和正则项（reg_loss）严格分离。
    blur_weight 只作用于 photo_loss，不放大正则项（问题 10.6 修正）。
    """
    image    = render_pkg["render"]
    gt_image = viewpoint_cam.original_image.cuda()
    blur_type = viewpoint_cam.blur_type
    image_id  = viewpoint_cam.image_id

    # 1. 曝光校正（只修 loss 端，不改 Gaussian）
    if exposure_model is not None and iteration >= opt.luminance_start_iter:
        image_for_loss = exposure_model(image, image_id)
        exp_reg = exposure_model.regularization_loss(
            lambda_gain=opt.luminance_lambda_gain,
            lambda_bias=opt.luminance_lambda_bias,
        )
    else:
        image_for_loss = image
        exp_reg = torch.tensor(0.0, device=image.device)

    # 2. 数据项（photo loss）
    Ll1   = l1_loss(image_for_loss, gt_image)
    Lssim = 1.0 - ssim(image_for_loss, gt_image)
    photo_loss_raw = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim

    # blur_type 权重：只放大数据项，不放大正则项
    weight_map = {0: opt.sharp_weight, 1: opt.motion_weight, 2: opt.defocus_weight}
    blur_weight = weight_map.get(blur_type, 1.0)
    photo_loss_weighted = photo_loss_raw * blur_weight

    # 3. 正则项（不乘 blur_weight）
    reg_loss = exp_reg

    if render_pkg.get("blur_code") is not None:
        z = render_pkg["blur_code"]
        reg_loss = reg_loss + opt.lambda_code * (z ** 2).mean()

    if render_pkg.get("delta_reg") is not None:
        reg_loss = reg_loss + opt.lambda_delta * render_pkg["delta_reg"]

    # 4. 合并
    loss_total = photo_loss_weighted + reg_loss

    # 5. 分量记录（方便排查哪种图主导了训练）
    components = {
        "photo_raw":      photo_loss_raw.item(),
        "photo_weighted": photo_loss_weighted.item(),
        "blur_type":      blur_type,
        "blur_weight":    blur_weight,
        "reg_total":      reg_loss.item(),
        "exp_reg":        exp_reg.item() if torch.is_tensor(exp_reg) else 0.0,
    }
    return loss_total, components


def log_training(iteration, gaussians, loss_components, gdags, opt):
    """每 100 iter 打印关键指标。"""
    N = gaussians.get_xyz.shape[0]
    print(
        f"[{iteration:06d}] N={N} | "
        f"photo_raw={loss_components['photo_raw']:.4f} | "
        f"photo_w={loss_components['photo_weighted']:.4f} | "
        f"type={loss_components['blur_type']} w={loss_components['blur_weight']:.1f} | "
        f"reg={loss_components['reg_total']:.5f}"
    )
    if gdags is not None and opt.gdags_enable:
        gcr = gdags.compute_gcr()
        print(
            f"         gcr={gcr.mean():.3f} | "
            f"canon_n={gdags.canonical_grad_count.mean():.1f} | "
            f"protected={(gdags.protect_until_iter > iteration).sum().item()}"
        )
    # 异常检测
    if loss_components["photo_raw"] > 0.5:
        print(f"[WARNING] photo_loss 偏大，检查渲染是否正常")
    delta_reg = loss_components["reg_total"] - loss_components["exp_reg"]
    if delta_reg > 0.5:
        print(f"[WARNING] delta_reg 偏大（{delta_reg:.3f}），"
              f"检查 lambda_delta 或 GTnet 初始化")
```

---

## 13. 实施阶段与验收标准

### Phase 0：Baseline Freeze

**目的**：建立基准，确认原始仓库可复现，所有后续 Phase 的改动都要和这个基准比较。

**任务**：
```bash
python train.py \
  --mixed_blur False \
  --luminance_enable False \
  --eap_enable False \
  --gdags_enable False \
  --nexus_enable False \
  --renderer_backend original

# 保存基准结果
cp output/metrics.json baseline_ref/baseline_metrics.json
cp output/train_log.txt baseline_ref/baseline_train_log.txt
```

**验收**：能完整跑完训练，记录 PSNR/SSIM/N_GS，作为后续对比基准。

---

### Phase 1：Per-image 混合模糊路由

**改动文件**：
- `utils/auto_blur_detect.py`：新增
- `scene/blur_kernel.py`：新增 `ConditionalGTnet`（保留原始 `GTnet`）
- `scene/cameras.py`：新增 `image_id`, `blur_type`, `sharpness_score` 字段
- `scene/gaussian_model.py`：新增 `create_deblur_nets`, `get_deblur_optimizer_params`
- `gaussian_renderer/__init__.py`：新增路由逻辑
- `train.py`：接入采样、blur_type 传递、compute_loss
- `arguments/__init__.py`：新增参数

**验收标准**：
```
1. mixed_blur=False 时，PSNR 与 baseline 差异 < 0.2 dB（兼容性）
2. mixed_blur=True 时：
   - sharp 图的 canonical render 不比 baseline 更糊
   - blur_code L2 norm < 1.0（不过拟合）
   - motion 图 delta_reg < 0.5（GTnet 不过大变形）
   - 无 NaN，无 shape mismatch
3. 打印出 sharp/motion/defocus 分组数量
```

---

### Phase 2：Per-image 曝光校正

**改动文件**：
- `scene/luminance_model.py`：新增 `PerImageExposureModel`
- `train.py`：接入 `exposure_model` 和 `exp_optimizer`
- `arguments/__init__.py`：新增参数

**验收标准**：
```
1. gain_mean ∈ [0.85, 1.15]（不接近 clamp 边界）
2. bias_max < 0.04
3. sharp 图 canonical render 亮度不偏离 baseline
4. motion 图 delta_reg 不因 exposure 校正而增大
5. 关闭 luminance_enable，结果等同于 Phase 1
6. 曝光差异大的手机图，训练 loss 更稳定
```

---

### Phase 3：EAP-GS 初始化增强

**改动文件**：
- `scene/point_augmentation/eap_init.py`：新增
- `scene/__init__.py` 或 `train.py`：在 COLMAP 后、GaussianModel 初始化前调用

**验收标准**：
```
打印日志：
  original_points
  eap_accepted_points  （必须 < original_points * 0.5）
  eap_rejected_reproj / rejected_density / rejected_quality

结果检查：
  COLMAP 原始高置信点全部保留
  floaters 不明显增加
  初始化后的点云在弱纹理区域更均匀
```

---

### Phase 4：gsplat Rasterizer Backend

**改动文件**：
- `gaussian_renderer/backends/__init__.py`：新增
- `gaussian_renderer/backends/original_backend.py`：新增（封装原始 rasterizer）
- `gaussian_renderer/backends/gsplat_backend.py`：新增
- `gaussian_renderer/__init__.py`：接入 `_call_rasterizer`

**注意**：`renderer_backend_deblur` 永远保持 `original`（第一版不对 deblur 路径启用 gsplat）。

**验收标准**：
```
同场景、同随机种子，比较 original vs gsplat（只对 blur_type=0）：
  iter 1 渲染图 max pixel diff < 0.01
  iter 1000 PSNR diff < 0.5 dB
  无 NaN，无黑图，颜色方向正常
  radii 分布基本一致
```

---

### Phase 5：Deblur-aware GDAGS

**改动文件**：
- `scene/density_control/__init__.py`：新增
- `scene/density_control/gdags_deblur_aware.py`：新增 `DeblurAwareGDAGS`
- `scene/density_control/gdags_integration.py`：新增 `gdags_aware_densify_and_prune`
- `train.py`：接入 probe 逻辑和密度控制

**验收标准**：
```
1. canonical probe 后：gaussians.get_xyz.grad is None（打印确认）
2. 每次 densify/prune 后：gdags.assert_consistent 不报错
3. N_GS 不快速下降（prune 过强）或爆炸（split/clone 过强）
4. EAP 点（source_type=1）在 5000 iter 前 prune_count == 0
5. canonical_grad_count.mean() > 0（有实际统计）
6. gcr_mean 日志有意义（不全是 0 或 1）
```

---

### Phase 6：NexusGS-lite

**改动文件**：
- `scene/point_augmentation/nexus_lite.py`：新增
- `scene/point_augmentation/confidence.py`：新增
- `train.py`：在 `nexus_start_iter` 后触发

**验收标准**：
```
1. accepted/rejected 各类别日志清晰
2. accepted_points < N_GS * 0.20（不能大量加点）
3. 保护期内 prune_count == 0
4. canonical render 细节在稀疏区域有提升
5. floaters 不增加
```

---

## 14. 参数配置完整清单

```python
# arguments/__init__.py
# 在 OptimizationParams 和 ModelParams 里新增以下参数

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # ============================================================
        # 原始参数（一律保留，不删除）
        # ============================================================
        # self.iterations      = 30000
        # self.use_pos         = 0        ← 兼容旧模式，mixed_blur=False 时生效
        # self.num_moments     = 4
        # self.lambda_s        = 0.01
        # self.lambda_p        = 0.01
        # self.gtnet_lr        = 1e-3
        # self.hidden          = 3
        # self.width           = 64
        # self.max_clamp       = 1.1
        # self.lambda_dssim    = 0.2
        # self.densify_from_iter    = 500
        # self.densification_interval = 100
        # self.opacity_reset_interval = 3000
        # self.densify_grad_threshold = 0.0002

        # ============================================================
        # Phase 1：混合模糊路由
        # ============================================================
        self.mixed_blur          = True
        # True  → 双分支模式，每张图独立选 blur 路径
        # False → 兼容原始模式，全局 use_pos 开关

        self.blur_code_dim       = 8
        # blur embedding 维度，8 维在表达力和抢解释权风险之间平衡
        # 不要超过 16

        self.warmup_iters        = 3000
        # 前 3000 iter 优先采样 sharp 图（70% 概率）
        # 目的：先建立基本几何，再引入模糊图

        self.sharp_sample_ratio  = 0.35
        self.motion_sample_ratio = 0.35
        # defocus_sample_ratio = 1 - sharp - motion = 0.30
        # sharp_sample_ratio 不能低于 0.30

        self.sharp_weight        = 2.0
        # sharp 图 photo_loss 的权重倍增
        # 只作用于 photo_loss，不放大正则项

        self.motion_weight       = 1.0
        self.defocus_weight      = 1.0
        self.lambda_code         = 1e-4   # blur_code L2 正则
        self.lambda_delta        = 1e-3   # GTnet delta 正则
        self.defocus_enable_rotation_delta = False
        # defocus 分支 rotation delta 默认关闭
        # 原因：四元数乘法需要 normalize，且 scale 版先稳定再开 rotation

        # ============================================================
        # Phase 2：曝光校正
        # ============================================================
        self.luminance_enable       = False
        self.luminance_start_iter   = 1000
        # 先让 GTnet 收敛 1000 步，再开曝光校正
        # 避免训练初期 GTnet 还没收敛时曝光校正抢信号

        self.luminance_lr           = 5e-4
        # 独立 optimizer 学习率，比主训练（1e-3）小一个数量级

        self.luminance_lambda_gain  = 5e-3
        self.luminance_lambda_bias  = 1e-2
        # bias 正则比 gain 更强，因为 bias 更容易无意中解释模糊残差

        # 注意：没有 luminance_stage / luminance_per_channel / luminance_matrix
        # 只有一种配置：scalar gain + scalar bias（hard clamp 已在模型内部）

        # ============================================================
        # Phase 3：EAP 初始化
        # ============================================================
        self.eap_enable                  = False
        self.eap_min_reproj_error        = 2.0
        self.eap_min_baseline_deg        = 1.0
        self.eap_max_baseline_deg        = 30.0
        self.eap_low_density_percentile  = 35
        self.eap_max_added_points_ratio  = 0.5
        # 安全上限：EAP 加的点不超过原始点的 50%

        # ============================================================
        # Phase 4：gsplat backend
        # ============================================================
        self.renderer_backend        = "original"
        # "original" | "gsplat"
        # 作用于 blur_type=0（sharp 图）的渲染

        self.renderer_backend_deblur = "original"
        # 第一版固定 original，不对 motion/defocus 启用 gsplat
        # 原因：多 moment 循环的梯度累积需要单独验证

        # ============================================================
        # Phase 5：GDAGS
        # ============================================================
        self.gdags_enable              = False
        self.gdags_start_iter          = 3000
        # 在 GTnet 和几何都基本稳定后才开始 GDAGS 统计

        self.gdags_probe_interval      = 200
        # 每 200 iter 做一次 canonical probe
        # 过于频繁会浪费计算，过于稀疏会导致统计不够准确

        self.gdags_gcr_split_threshold = 0.35
        # GCR < 0.35 才考虑 split
        # 越低越保守（split 越少）

        self.gdags_gcr_clone_threshold = 0.75
        # GCR > 0.75 才考虑 clone

        self.gdags_min_age_for_prune   = 2000
        # Gaussian 至少存在 2000 iter 才可能被 prune
        # 比原方案（1500）更保守

        self.gdags_prune_ratio         = 0.01
        # 每次最多 prune 1% 的 Gaussian
        # 极保守，防止意外清除有效点

        self.opacity_threshold         = 0.005
        # prune 的 opacity 阈值（与原始相同）

        # ============================================================
        # Phase 6：NexusGS-lite
        # ============================================================
        self.nexus_enable                  = False
        self.nexus_start_iter_ratio        = 0.35
        # 在 35% 进度后才开始补点
        # 等 GDAGS 和主 3DGS 都稳定

        self.nexus_pair_min_inliers        = 120
        self.nexus_flow_fb_thresh_px       = 1.5
        self.nexus_epipolar_thresh_px      = 1.5
        self.nexus_reproj_thresh_px        = 2.0
        self.nexus_conf_add_thresh         = 0.70
        self.nexus_max_added_points_ratio  = 0.20
        # 每次最多加 N_GS * 20% 的点

        self.nexus_protect_new_points_iter = 1500
        # Nexus 新点保护 1500 iter（< GDAGS min_age 2000，有重叠保护）

        super().__init__(parser, "Optimization Parameters")
```

---

## 15. 附录

### A. 推荐实验配置

#### A.1 最稳健起步（验证 Phase 1+2+3）

```bash
python train.py \
  --mixed_blur True \
  --blur_code_dim 8 \
  --warmup_iters 3000 \
  --sharp_weight 2.0 \
  --lambda_code 1e-4 \
  --lambda_delta 1e-3 \
  --luminance_enable True \
  --luminance_start_iter 1000 \
  --luminance_lr 5e-4 \
  --eap_enable True \
  --gdags_enable False \
  --nexus_enable False \
  --renderer_backend original \
  --renderer_backend_deblur original
```

#### A.2 完整融合（所有模块）

```bash
python train.py \
  --mixed_blur True \
  --blur_code_dim 8 \
  --warmup_iters 3000 \
  --sharp_weight 2.0 \
  --luminance_enable True \
  --luminance_start_iter 1000 \
  --eap_enable True \
  --renderer_backend gsplat \
  --renderer_backend_deblur original \
  --gdags_enable True \
  --gdags_start_iter 3000 \
  --gdags_min_age_for_prune 2000 \
  --gdags_prune_ratio 0.01 \
  --nexus_enable True \
  --nexus_start_iter_ratio 0.35 \
  --nexus_protect_new_points_iter 1500
```

#### A.3 兼容原始仓库（验证 baseline 不变）

```bash
python train.py \
  --mixed_blur False \
  --luminance_enable False \
  --eap_enable False \
  --gdags_enable False \
  --nexus_enable False \
  --renderer_backend original
```

---

### B. 新增文件清单

```
scene/
    luminance_model.py              # PerImageExposureModel
    blur_kernel.py                  # 新增 ConditionalGTnet（保留原始 GTnet）
    density_control/
        __init__.py
        gdags_deblur_aware.py       # DeblurAwareGDAGS
        gdags_integration.py        # gdags_aware_densify_and_prune
    point_augmentation/
        __init__.py
        eap_init.py                 # EAP 初始化增强
        nexus_lite.py               # NexusGS-lite 补点
        confidence.py               # 置信度计算

gaussian_renderer/
    __init__.py                     # 修改：路由逻辑
    backends/
        __init__.py
        original_backend.py         # 封装原始 rasterizer
        gsplat_backend.py           # gsplat 封装

utils/
    auto_blur_detect.py             # blur_type 自动检测
    geometry_utils.py               # 几何辅助函数
    flow_utils.py                   # optical flow 接口
    debug_dump.py                   # 调试输出工具

arguments/
    __init__.py                     # 修改：新增所有参数

train.py                            # 修改：主循环
scene/gaussian_model.py             # 修改：create_deblur_nets
scene/cameras.py                    # 修改：新增 image_id/blur_type 字段
```

---

### C. 最重要的检查点（开始实施前必读）

| 检查点 | 验证方式 | 如果失败 |
|---|---|---|
| canonical probe 后 `.grad is None` | `assert gaussians.get_xyz.grad is None` | 检查 zero_grad 调用时机 |
| GDAGS buffer 数量一致 | `gdags.assert_consistent(gaussians)` | 检查 on_clone/on_split/on_prune 调用顺序 |
| base_pose_moment 是 `i==M` 不是 `i==0` | 检查 render 函数的 `is_base_pose = (i == M)` | 重命名，禁止 "moment-0" |
| defocus rotation 已归一化 | `F.normalize(rotations * r_delta, dim=-1)` | 加归一化或关闭 rotation delta |
| Luminance 无 per-channel | `PerImageExposureModel` 只有 `log_gain [N,1,1]` 和 `bias [N,1,1]` | 禁止添加 per-channel |
| photo_loss 和 reg_loss 分离 | 日志里 `photo_weighted` 和 `reg_total` 分别记录 | 重写 compute_loss |
| EAP 点 5000 iter 前不 prune | `gdags.source_type==1 & iter<5000 → not pruned` | 检查 decide_densify_prune |
| 推理时 deblur=False | 推理脚本不传 blur_type，不调用 GTnet | 检查 render_sets.py |
| mixed_blur=False 结果等同 baseline | PSNR diff < 0.2 dB | 检查兼容路径 |
| 所有开关默认关闭 | `opt.luminance_enable=False` 等 | 检查 arguments/__init__.py |
