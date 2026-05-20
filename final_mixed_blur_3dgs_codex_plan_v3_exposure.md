# 混合模糊 3DGS 融合方案：Codex 实施版 v3

> 目标：把原“融合方案综合评估与完整优化版”改成一份可直接交给 Codex 落代码的工程规格。
>
> 本版重点修复 6 个会导致实现错误或训练偏移的问题：
>
> 1. GDAGS canonical probe 梯度必须隔离，不能污染正常 optimizer step。
> 2. GDAGS 统计 buffer 必须随 clone / split / prune 同步扩容、复制、裁剪。
> 3. motion 分支不要再叫 moment-0；原始位置渲染统一命名为 `identity_moment` / `canonical_moment`。
> 4. defocus 分支默认只学习 scale；rotation delta 默认关闭，开启时必须保证 quaternion 合法。
> 5. Luminance 模型第一版只用 per-image scalar exposure gain + scalar brightness bias，禁止一开始启用 per-channel RGB、完整 3x3 matrix 和 tone curve。
> 6. loss 拆成 `photo_loss_raw`、`photo_loss_weighted`、`reg_loss`，避免 blur type 权重不透明地改变正则相对比例。

---

## 0. 最终工程结论

### 0.1 推荐实施顺序

必须按下面顺序实现，不要一次性打开所有模块：

```text
Phase 0: baseline freeze
    - 不改任何训练逻辑
    - 记录原仓库 PSNR / SSIM / LPIPS / Gaussian 数量曲线

Phase 1: mixed blur routing
    - 加 blur_type / image_id
    - sharp / motion / defocus 分支可跑通
    - 不开 GDAGS、不开发光照校正、不改 densify/prune

Phase 2: safe luminance
    - 只启用 per-image scalar exposure gain + scalar brightness bias
    - 只作用在 loss 端
    - 记录 raw / weighted photo loss

Phase 3: GDAGS stats only
    - 只统计，不改变 densify / prune 决策
    - 完成 canonical probe 梯度隔离
    - 完成所有 buffer 同步逻辑

Phase 4: GDAGS density control
    - 开启 clone / split / prune 决策
    - prune 必须极保守
    - 新生点必须保护

Phase 5: optional advanced features
    - EAP init
    - NexusGS-lite
    - gsplat backend
    - defocus rotation delta
    - Luminance 3x3 matrix / tone curve
```

### 0.2 第一版默认配置

第一版目标是“正确、稳定、可调试”，不是一次性追最高指标。

```bash
--mixed_blur
--renderer_backend original
--renderer_backend_deblur original

--luminance_enable
--luminance_mode exposure_gain_bias
--luminance_matrix_enable false
--luminance_curve_enable false
--luminance_per_channel false

--gdags_enable false
--gdags_stats_enable false

--defocus_learn_rotation false
```

GDAGS 应在 Phase 3 后再打开：

```bash
--gdags_stats_enable true
--gdags_enable false
--gdags_probe_use_autograd_grad true
```

确认 stats 正常后再进入 Phase 4：

```bash
--gdags_enable true
--gdags_prune_enable true
--gdags_clone_enable true
--gdags_split_enable true
```

---

## 1. 核心数据结构

### 1.1 blur_type 约定

所有代码必须使用同一套 enum，不允许散落 magic number。

```python
BLUR_SHARP = 0
BLUR_MOTION = 1
BLUR_DEFOCUS = 2
```

建议放在：

```text
utils/blur_types.py
```

内容：

```python
from enum import IntEnum

class BlurType(IntEnum):
    SHARP = 0
    MOTION = 1
    DEFOCUS = 2
```

### 1.2 Camera 新增字段

每个训练相机必须有：

```python
camera.image_id: int
camera.blur_type: int
camera.sharpness_score: Optional[float]
```

约束：

```python
assert hasattr(viewpoint_cam, "image_id")
assert hasattr(viewpoint_cam, "blur_type")
assert viewpoint_cam.blur_type in (0, 1, 2)
```

### 1.3 blur label 文件

推荐格式：

```json
{
  "images": {
    "00001.png": {"blur_type": "sharp"},
    "00002.png": {"blur_type": "motion"},
    "00003.png": {"blur_type": "defocus"}
  }
}
```

加载逻辑：

```python
BLUR_NAME_TO_ID = {
    "sharp": 0,
    "motion": 1,
    "defocus": 2,
}
```

如果没有 label 文件，可以 fallback 到自动检测，但自动检测结果必须写日志：

```text
[BlurLabel] 00003.png auto_detect -> defocus, score=...
```

---

## 2. Render 入口与命名规范

### 2.1 统一 render 签名

```python
def render(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    scaling_modifier=1.0,
    deblur=False,
    blur_type=0,
    image_id=None,
    lambda_s=0.01,
    lambda_p=0.01,
    max_clamp=1.1,
    backend="original",
    return_identity_moment_only=False,
):
    """
    blur_type:
        0 = sharp
        1 = motion
        2 = defocus

    return_identity_moment_only:
        只允许 GDAGS stats / probe 调用。
        motion 分支下只渲染原始位置对应的 identity_moment。
        不要叫 moment-0。
    """
```

### 2.2 禁止继续使用 moment-0 这个名字

原方案中“moment-0”命名错误，因为代码实际逻辑是：

```python
if i == M:
    t_pos = means3D
else:
    t_pos = means3D + p_delta_3d[..., i]
```

也就是说，原始位置渲染在数组最后一个位置 `i == M`，不是第 0 个。

最终统一命名：

| 概念 | 正确命名 | 含义 |
|---|---|---|
| 原始 Gaussian 位置渲染 | `identity_moment` | `t_pos = means3D` |
| canonical / sharp-like 渲染 | `canonical_moment` | 供 GDAGS canonical stats 使用 |
| 运动采样渲染 | `motion_moments` | `t_pos = means3D + p_delta[..., k]` |
| motion 平均图 | `blurred_render` | 多个 motion moments + identity_moment 平均 |

不要在代码、日志、注释里写 `moment0`、`moment-0`。

### 2.3 render 返回结构

sharp 返回：

```python
{
    "render": rendered_image,
    "viewspace_points": means2D,
    "visibility_filter": radii > 0,
    "radii": radii,
    "blur_code": None,
    "delta_reg": zero,
    "gdags_stats_viewspace_points": means2D,
    "gdags_stats_visibility": radii > 0,
}
```

defocus 返回：

```python
{
    "render": rendered_image,
    "viewspace_points": means2D,
    "visibility_filter": radii > 0,
    "radii": radii,
    "blur_code": z,
    "delta_reg": delta_reg,
    "gdags_stats_viewspace_points": means2D,
    "gdags_stats_visibility": radii > 0,
}
```

motion 返回：

```python
{
    "render": blurred_render,
    "viewspace_points_all": [sp_0, sp_1, ..., sp_M],
    "visibility_all": [vis_0, vis_1, ..., vis_M],
    "radii_all": [r_0, r_1, ..., r_M],
    "identity_moment": {
        "render": identity_img,
        "viewspace_points": identity_sp,
        "visibility_filter": identity_vis,
        "radii": identity_radii,
    },
    "blur_code": z,
    "delta_reg": delta_reg,
    "gdags_stats_viewspace_points": identity_sp,
    "gdags_stats_visibility": identity_vis,
}
```

关键点：

```python
# GDAGS 统计永远使用显式字段，不靠 [-1] 这种隐含约定。
viewspace_for_stats = render_pkg["gdags_stats_viewspace_points"]
visibility_for_stats = render_pkg["gdags_stats_visibility"]
```

### 2.4 motion 分支伪代码

```python
def _render_motion(..., return_identity_moment_only=False):
    M = pc.motion_GTnet.num_moments

    s_delta, r_delta, p_delta, z = pc.motion_GTnet(
        _pos, _scales, _rotations, _viewdirs, image_id
    )

    s_delta = torch.clamp(lambda_s * s_delta + (1 - lambda_s), min=1.0, max=max_clamp)
    p_delta = lambda_p * p_delta

    p_delta_3d = p_delta.view(-1, 3, M)
    s_delta_3d = s_delta.view(-1, 3, M + 1)

    motion_renders = []
    viewspace_all = []
    visibility_all = []
    radii_all = []

    # 1) offset motion moments
    if not return_identity_moment_only:
        for k in range(M):
            sp = torch.zeros_like(means3D, requires_grad=True, device=means3D.device)
            sp.retain_grad()

            t_pos = means3D + p_delta_3d[..., k]
            t_s = scales * s_delta_3d[..., k]
            t_r = rotations

            img, radii = _call_rasterizer(..., t_pos, sp, ..., t_s, t_r, ...)
            motion_renders.append(img)
            viewspace_all.append(sp)
            visibility_all.append(radii > 0)
            radii_all.append(radii)

    # 2) identity moment: original Gaussian positions
    identity_sp = torch.zeros_like(means3D, requires_grad=True, device=means3D.device)
    identity_sp.retain_grad()

    identity_s = scales * s_delta_3d[..., M]
    identity_r = rotations

    identity_img, identity_radii = _call_rasterizer(
        ..., means3D, identity_sp, ..., identity_s, identity_r, ...
    )
    identity_vis = identity_radii > 0

    if return_identity_moment_only:
        return {
            "render": identity_img,
            "identity_moment": {
                "render": identity_img,
                "viewspace_points": identity_sp,
                "visibility_filter": identity_vis,
                "radii": identity_radii,
            },
            "blur_code": z,
            "delta_reg": torch.zeros((), device=means3D.device),
            "gdags_stats_viewspace_points": identity_sp,
            "gdags_stats_visibility": identity_vis,
        }

    motion_renders.append(identity_img)
    viewspace_all.append(identity_sp)
    visibility_all.append(identity_vis)
    radii_all.append(identity_radii)

    blurred_render = sum(motion_renders) / len(motion_renders)

    delta_reg = (
        p_delta.abs().mean()
        + (s_delta - 1.0).abs().mean()
    )

    return {
        "render": blurred_render,
        "viewspace_points_all": viewspace_all,
        "visibility_all": visibility_all,
        "radii_all": radii_all,
        "identity_moment": {
            "render": identity_img,
            "viewspace_points": identity_sp,
            "visibility_filter": identity_vis,
            "radii": identity_radii,
        },
        "blur_code": z,
        "delta_reg": delta_reg,
        "gdags_stats_viewspace_points": identity_sp,
        "gdags_stats_visibility": identity_vis,
    }
```

---

## 3. Defocus rotation 处理

### 3.1 第一版默认：不学习 rotation

defocus 分支第一版只学习 scale delta：

```python
--defocus_learn_rotation false
```

实现：

```python
s_delta, r_delta, _, z = pc.defocus_GTnet(...)

s_delta = torch.clamp(lambda_s * s_delta + (1 - lambda_s), min=1.0, max=max_clamp)
new_scales = scales * s_delta

if opt.defocus_learn_rotation:
    new_rotations = apply_safe_rotation_delta(rotations, r_delta)
else:
    new_rotations = rotations
```

`delta_reg` 第一版只统计 scale：

```python
delta_reg = (s_delta - 1.0).abs().mean()
```

### 3.2 如果必须打开 rotation

如果 `rotations` 是 quaternion，不能只做裸的逐元素乘法：

```python
# 禁止作为最终实现
new_rotations = rotations * r_delta
```

最低限度必须 normalize：

```python
new_rotations = torch.nn.functional.normalize(rotations * r_delta, dim=-1)
```

更稳的做法是让网络输出一个 delta quaternion，然后做 quaternion composition：

```python
def quat_mul(q, r):
    # q, r: [..., 4], convention must match existing codebase
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)

def apply_safe_rotation_delta(rotations, r_delta):
    delta_q = torch.nn.functional.normalize(r_delta, dim=-1)
    new_rot = quat_mul(rotations, delta_q)
    return torch.nn.functional.normalize(new_rot, dim=-1)
```

验收条件：

```python
norm = new_rotations.norm(dim=-1)
assert torch.allclose(norm.mean(), torch.ones_like(norm.mean()), atol=1e-3)
```

---

## 4. Luminance 模型：第一版只做曝光归一化

### 4.1 第一版定位

你的手机数据里主要问题是曝光和整体亮度不一致，不是色偏。因此 Luminance 第一版不应该作为“颜色校正模块”，而应该只是一个曝光归一化模块。

第一版只允许每张图学习两个标量：

```python
image_corrected = image * exposure_gain[image_id] + brightness_bias[image_id]
```

其中：

```text
exposure_gain: 乘法项，用于修曝光
brightness_bias: 加法项，用于修整体亮度 / 黑电平偏移
```

禁止第一版启用：

```text
per-channel RGB scale
per-channel RGB bias
3x3 color matrix
tone curve
```

原因：

- `blur_code` 和 Luminance 都是 per-image 低频适应。
- 手机数据没有明显色偏时，per-channel RGB scale / bias 会引入不必要颜色自由度。
- 3x3 matrix + bias + tone curve 容量过强，容易解释 blur residual。
- 这会导致 GTnet 学不到干净的 motion / defocus compensation。

### 4.2 推荐实现

```python
class PerImageExposureModel(nn.Module):
    def __init__(self, num_images):
        super().__init__()

        # 每张图一个曝光增益，初始化为 0，exp(0)=1
        self.log_gain = nn.Parameter(torch.zeros(num_images, 1, 1, 1))

        # 每张图一个亮度偏置，初始化为 0
        self.bias = nn.Parameter(torch.zeros(num_images, 1, 1, 1))

    def forward(self, image, image_id):
        """
        image: [3, H, W]
        image_id: int
        """

        log_gain = self.log_gain[image_id]
        bias = self.bias[image_id]

        # 限制范围，防止 luminance 解释模糊残差
        gain = torch.exp(torch.clamp(log_gain, min=-0.30, max=0.30))
        bias = torch.clamp(bias, min=-0.05, max=0.05)

        corrected = image * gain + bias
        return torch.clamp(corrected, 0.0, 1.0)

    def regularization_loss(self, lambda_gain=5e-3, lambda_bias=1e-2):
        return (
            lambda_gain * (self.log_gain ** 2).mean()
            + lambda_bias * (self.bias ** 2).mean()
        )
```

### 4.3 与 scalar exposure gain + brightness bias 的区别

| 方案 | 参数量 | 能修什么 | 风险 |
|---|---:|---|---|
| scalar exposure gain + brightness bias | 每图 6 个参数 | 曝光、亮度、轻微色偏 | 可能引入不必要颜色自由度 |
| scalar gain + bias | 每图 2 个参数 | 曝光、整体亮度 | 最安全，最不容易干扰 deblur |
| 3x3 matrix + bias | 每图 12 个参数 | 色彩混合、白平衡、复杂颜色偏差 | 对当前场景过强 |
| tone curve | 更多 | 非线性响应 / HDR 差异 | 最容易抢解释权 |

### 4.4 调度建议

```bash
--luminance_start_iter 1000
--luminance_lr 1e-3
--luminance_lambda_gain 5e-3
--luminance_lambda_bias 1e-2
```

第一版默认配置：

```bash
--luminance_enable
--luminance_mode exposure_gain_bias
--luminance_matrix_enable false
--luminance_curve_enable false
--luminance_per_channel false
```

范围建议：

```text
log_gain clamp: [-0.30, 0.30]
gain range: approximately [0.74, 1.35]

bias clamp: [-0.05, 0.05]
```

如果 motion / defocus 的 `delta_reg` 持续变小但图像仍糊，且 luminance gain / bias 变大，说明 Luminance 在抢解释权。应降低 luminance lr 或延后 start iter。

### 4.5 后续可选升级

只有当 scalar exposure gain + brightness bias 稳定后，并且确认数据确实存在色偏，才允许升级到 per-channel scalar exposure gain + brightness bias。

3x3 matrix 必须更晚再开，且必须是 residual identity：

```python
M = I + matrix_delta
```

tone curve 必须最后再开，不要与 GTnet 一起从头训练。

---

## 5. Loss 拆分与日志

### 5.1 正确 loss 结构

不要把主 loss 乘权重后直接继续往上加正则，导致日志难以解释。

必须拆成：

```python
photo_loss_raw = compute_photo_loss(image_for_loss, gt_image)

weight = get_blur_type_weight(
    blur_type,
    sharp_weight=opt.sharp_weight,
    motion_weight=opt.motion_weight,
    defocus_weight=opt.defocus_weight,
)

photo_loss_weighted = weight * photo_loss_raw

code_reg = compute_code_reg(render_pkg, opt)
delta_reg = compute_delta_reg(render_pkg, opt)
lum_reg = compute_lum_reg(luminance_model, opt)

reg_loss = code_reg + delta_reg + lum_reg

loss = photo_loss_weighted + reg_loss
```

其中：

```python
def compute_photo_loss(pred, gt):
    Ll1 = l1_loss(pred, gt)
    ssim_loss = 1.0 - ssim(pred, gt)
    return (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
```

### 5.2 正则项不乘 blur_type 权重

`sharp_weight`、`motion_weight`、`defocus_weight` 只作用于 data term：

```python
photo_loss_weighted = blur_type_weight * photo_loss_raw
```

正则项单独加：

```python
loss = photo_loss_weighted + reg_loss
```

这样可以明确回答：

- sharp 图是不是主导了 data term？
- GTnet delta 是不是被 reg 压住？
- Luminance 是不是在抢解释权？

### 5.3 必须记录的日志

每 100 或 500 iter 记录：

```python
log = {
    "loss/total": loss.item(),
    "loss/photo_raw": photo_loss_raw.item(),
    "loss/photo_weighted": photo_loss_weighted.item(),
    "loss/reg_total": reg_loss.item(),
    "loss/code_reg": code_reg.item(),
    "loss/delta_reg": delta_reg.item(),
    "loss/lum_reg": lum_reg.item(),
    "blur/type": blur_type,
    "blur/weight": weight,
}
```

按 blur_type 分桶的滑动平均也要记录：

```text
photo_raw/sharp
photo_raw/motion
photo_raw/defocus
photo_weighted/sharp
photo_weighted/motion
photo_weighted/defocus
```

### 5.4 sharp_weight 建议

如果 `sharp_weight=2.0`，则 sharp 图 data term 会显著变大，但正则项不变，这本身不是错误，只是必须可见、可控。

第一版建议：

```bash
--sharp_weight 1.0
--motion_weight 1.0
--defocus_weight 1.0
```

如果 sharp 图较少，再逐步调到：

```bash
--sharp_weight 1.2
```

不要一开始就用 2.0。

---

## 6. GDAGS canonical probe：梯度隔离是硬约束

### 6.1 禁止的实现

禁止在正常训练 `loss.backward()` 后直接做：

```python
proxy_loss.backward()
gaussians.update_canonical_stats(...)
optimizer.step()
```

这会让 canonical probe 的梯度叠加到模型参数 `.grad`，污染下一次 `optimizer.step()` 或当前 step。

### 6.2 推荐实现：torch.autograd.grad

canonical probe 的目标是获取 `viewspace_points` 的梯度方向，不需要更新 Gaussian 参数。

默认实现必须用：

```python
grad_vsp = torch.autograd.grad(
    outputs=proxy_loss,
    inputs=probe_pkg["viewspace_points"],
    retain_graph=False,
    create_graph=False,
    allow_unused=True,
)[0]
```

然后把 `grad_vsp` 显式传给 stats 更新函数。

### 6.3 canonical probe 函数

```python
def run_gdags_canonical_probe(
    gaussians,
    probe_cam,
    pipe,
    background,
    opt,
    optimizer,
    luminance_optimizer=None,
):
    """
    只统计 canonical gradient，不更新任何可学习参数。
    默认只用 sharp 图。
    """

    # 保险：probe 前清空所有已有梯度
    optimizer.zero_grad(set_to_none=True)
    if luminance_optimizer is not None:
        luminance_optimizer.zero_grad(set_to_none=True)

    probe_pkg = render(
        probe_cam,
        gaussians,
        pipe,
        background,
        deblur=False,
        blur_type=0,
        image_id=probe_cam.image_id,
        backend="original",
    )

    pred = probe_pkg["render"]
    gt = probe_cam.original_image.cuda()

    # canonical proxy loss 可以用 photo loss，也可以只用 L1
    proxy_loss = l1_loss(pred, gt)

    vsp = probe_pkg["viewspace_points"]
    vis = probe_pkg["visibility_filter"]

    grad_vsp = torch.autograd.grad(
        proxy_loss,
        vsp,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )[0]

    gaussians.gdags.update_canonical_stats(
        viewspace_points=vsp,
        visibility=vis,
        grad=grad_vsp,
    )

    # 保险：probe 后再次清空，确保不会影响正常训练
    optimizer.zero_grad(set_to_none=True)
    if luminance_optimizer is not None:
        luminance_optimizer.zero_grad(set_to_none=True)
```

### 6.4 fallback：如果必须用 backward

如果某些 rasterizer 对 `autograd.grad` 不兼容，才允许 fallback：

```python
optimizer.zero_grad(set_to_none=True)
proxy_loss.backward()
gaussians.gdags.update_canonical_stats(
    viewspace_points=vsp,
    visibility=vis,
    grad=vsp.grad,
)
optimizer.zero_grad(set_to_none=True)
```

注意：

- fallback 后绝对不能 `optimizer.step()`。
- fallback 前后都必须 `zero_grad(set_to_none=True)`。
- 必须有单元测试验证 probe 不改变参数。

### 6.5 canonical probe 只用 sharp 图

```python
if len(sharp_cameras) > 0:
    probe_cam = random.choice(sharp_cameras)
else:
    # 可选 fallback：从全体中选 sharpness_score 最高的少量视角
    # 但必须在日志中标明不是人工 sharp label
    probe_cam = select_highest_sharpness_camera(train_cameras)
```

禁止使用 motion / defocus 图做 canonical probe。

---

## 7. GDAGS buffer 同步

### 7.1 GDAGS buffers

这些 buffer 长度必须永远等于当前 Gaussian 数量 `N`：

```python
blur_grad_dir_sum:        [N, 3]
blur_grad_count:          [N]
canonical_grad_dir_sum:   [N, 3]
canonical_grad_count:     [N]
protect_until_iter:       [N]
gaussian_age:             [N]
source_type:              [N]
```

`source_type` 约定：

```python
SOURCE_SFM = 0
SOURCE_EAP = 1
SOURCE_NEXUS = 2
SOURCE_CLONE = 3
SOURCE_SPLIT = 4
```

### 7.2 shape invariant

每次 densify / prune 后必须调用：

```python
gaussians.gdags.assert_shape(gaussians.get_xyz.shape[0])
```

实现：

```python
def assert_shape(self, n):
    assert self.blur_grad_dir_sum.shape[0] == n
    assert self.blur_grad_count.shape[0] == n
    assert self.canonical_grad_dir_sum.shape[0] == n
    assert self.canonical_grad_count.shape[0] == n
    assert self.protect_until_iter.shape[0] == n
    assert self.gaussian_age.shape[0] == n
    assert self.source_type.shape[0] == n
```

### 7.3 stats update 函数必须接收显式 grad

不要在函数内部只读 `viewspace_points.grad`，因为 canonical probe 推荐用 `autograd.grad`，不会填充 `.grad`。

```python
def update_blur_stats(self, viewspace_points, visibility, blur_type, grad=None):
    if grad is None:
        grad = viewspace_points.grad
    if grad is None:
        return

    grad_visible = grad[visibility]
    grad_dir = grad_visible / (grad_visible.norm(dim=-1, keepdim=True) + 1e-8)

    self.blur_grad_dir_sum[visibility] += grad_dir
    self.blur_grad_count[visibility] += 1


def update_canonical_stats(self, viewspace_points, visibility, grad=None):
    if grad is None:
        grad = viewspace_points.grad
    if grad is None:
        return

    grad_visible = grad[visibility]
    grad_dir = grad_visible / (grad_visible.norm(dim=-1, keepdim=True) + 1e-8)

    self.canonical_grad_dir_sum[visibility] += grad_dir
    self.canonical_grad_count[visibility] += 1
```

### 7.4 on_clone

clone 时，新点来自 parent Gaussian。新点应有保护期，统计可以继承一部分，但 age 必须归零。

```python
def on_clone(self, parent_idx, current_iter, protect_iters):
    """
    parent_idx: LongTensor [K]
    append K new buffer rows.
    """
    new_blur_dir = 0.5 * self.blur_grad_dir_sum[parent_idx].clone()
    new_blur_cnt = 0.5 * self.blur_grad_count[parent_idx].clone()

    new_can_dir = 0.5 * self.canonical_grad_dir_sum[parent_idx].clone()
    new_can_cnt = 0.5 * self.canonical_grad_count[parent_idx].clone()

    K = parent_idx.numel()
    new_age = torch.zeros(K, dtype=self.gaussian_age.dtype, device=self.gaussian_age.device)
    new_protect = torch.full(
        (K,),
        current_iter + protect_iters,
        dtype=self.protect_until_iter.dtype,
        device=self.protect_until_iter.device,
    )
    new_source = torch.full(
        (K,),
        SOURCE_CLONE,
        dtype=self.source_type.dtype,
        device=self.source_type.device,
    )

    self._append(
        new_blur_dir,
        new_blur_cnt,
        new_can_dir,
        new_can_cnt,
        new_protect,
        new_age,
        new_source,
    )
```

### 7.5 on_split

split 时，一个 parent 产生多个 children。children 的统计建议更保守，继承更少，防止 parent 的旧统计误导新点。

```python
def on_split(self, parent_idx_repeated, current_iter, protect_iters):
    """
    parent_idx_repeated: LongTensor [K * num_split]
    """
    new_blur_dir = 0.25 * self.blur_grad_dir_sum[parent_idx_repeated].clone()
    new_blur_cnt = 0.25 * self.blur_grad_count[parent_idx_repeated].clone()

    new_can_dir = 0.25 * self.canonical_grad_dir_sum[parent_idx_repeated].clone()
    new_can_cnt = 0.25 * self.canonical_grad_count[parent_idx_repeated].clone()

    K = parent_idx_repeated.numel()
    new_age = torch.zeros(K, dtype=self.gaussian_age.dtype, device=self.gaussian_age.device)
    new_protect = torch.full(
        (K,),
        current_iter + protect_iters,
        dtype=self.protect_until_iter.dtype,
        device=self.protect_until_iter.device,
    )
    new_source = torch.full(
        (K,),
        SOURCE_SPLIT,
        dtype=self.source_type.dtype,
        device=self.source_type.device,
    )

    self._append(
        new_blur_dir,
        new_blur_cnt,
        new_can_dir,
        new_can_cnt,
        new_protect,
        new_age,
        new_source,
    )
```

### 7.6 on_prune

prune 必须和 GaussianModel 使用同一个 keep mask。

```python
def on_prune(self, keep_mask):
    self.blur_grad_dir_sum = self.blur_grad_dir_sum[keep_mask]
    self.blur_grad_count = self.blur_grad_count[keep_mask]

    self.canonical_grad_dir_sum = self.canonical_grad_dir_sum[keep_mask]
    self.canonical_grad_count = self.canonical_grad_count[keep_mask]

    self.protect_until_iter = self.protect_until_iter[keep_mask]
    self.gaussian_age = self.gaussian_age[keep_mask]
    self.source_type = self.source_type[keep_mask]
```

如果原始代码里传的是 `prune_mask`，需要转换：

```python
keep_mask = ~prune_mask
gaussians.gdags.on_prune(keep_mask)
```

### 7.7 EAP / Nexus 新点

外部模块新增点时，不一定有 parent。此时用 `on_external_add`：

```python
def on_external_add(self, num_new, source_type, current_iter, protect_iters):
    device = self.blur_grad_count.device

    self._append(
        torch.zeros(num_new, 3, device=device),
        torch.zeros(num_new, device=device),
        torch.zeros(num_new, 3, device=device),
        torch.zeros(num_new, device=device),
        torch.full((num_new,), current_iter + protect_iters, dtype=torch.long, device=device),
        torch.zeros(num_new, dtype=torch.long, device=device),
        torch.full((num_new,), source_type, dtype=torch.uint8, device=device),
    )
```

EAP 建议保护：

```bash
--gdags_eap_protect_iters 5000
```

Nexus 建议保护：

```bash
--gdags_nexus_protect_iters 1500
```

### 7.8 GaussianModel 接入点

必须在这些位置接入 GDAGS 同步：

```text
GaussianModel.densify_and_clone
    - 找到 selected parent indices
    - append Gaussian 参数
    - gdags.on_clone(parent_idx, current_iter, protect_iters)

GaussianModel.densify_and_split
    - 找到 selected parent indices
    - 构造 parent_idx_repeated
    - append children Gaussian 参数
    - gdags.on_split(parent_idx_repeated, current_iter, protect_iters)
    - prune old parent 时再调用 gdags.on_prune(keep_mask)

GaussianModel.prune_points
    - 使用同一个 prune mask 裁剪 optimizer tensors
    - 同步调用 gdags.on_prune(keep_mask)

EAP / Nexus external add
    - append Gaussian 参数后
    - gdags.on_external_add(...)
```

任何新增 / 删除 Gaussian 的函数，如果没有同步 GDAGS buffer，都视为 bug。

---

## 8. GDAGS 决策逻辑

### 8.1 age 更新

每个训练 iter 后：

```python
if gaussians.gdags is not None:
    gaussians.gdags.gaussian_age += 1
```

新生点 age 在 `on_clone` / `on_split` / `on_external_add` 中置 0。

### 8.2 GCR 计算

```python
def compute_gcr(self, use_canonical=True):
    if use_canonical:
        count = self.canonical_grad_count.clamp(min=1)
        return self.canonical_grad_dir_sum.norm(dim=-1) / count
    count = self.blur_grad_count.clamp(min=1)
    return self.blur_grad_dir_sum.norm(dim=-1) / count
```

### 8.3 prune 必须极保守

建议第一版 prune 条件：

```python
prune_mask = (
    (opacity < opt.opacity_threshold)
    & (self.canonical_grad_count >= opt.gdags_min_canonical_count_for_prune)
    & (self.gaussian_age > opt.gdags_min_age_for_prune)
    & (iteration > self.protect_until_iter)
)
```

EAP 特殊保护：

```python
if iteration < opt.gdags_eap_global_protect_until:
    prune_mask = prune_mask & (self.source_type != SOURCE_EAP)
```

不要用 defocus scale delta 作为 prune 依据。

### 8.4 blur stats 与 canonical stats 的关系

- `canonical_grad_*` 是主决策依据。
- `blur_grad_*` 是辅助诊断。
- motion 图只用 `identity_moment` 的 viewspace gradient 做 stats。
- defocus 图可以收集 blur stats，但不要用 defocus 的 scale delta 推断几何缺失。
- canonical probe 只来自 sharp 图。

---

## 9. 训练循环最终结构

### 9.1 单次训练 step 伪代码

```python
for iteration in range(1, opt.iterations + 1):
    viewpoint_cam = sample_camera(iteration)
    blur_type = int(viewpoint_cam.blur_type)
    image_id = int(viewpoint_cam.image_id)

    use_deblur = bool(opt.deblur and blur_type != BLUR_SHARP)

    optimizer.zero_grad(set_to_none=True)
    if luminance_optimizer is not None:
        luminance_optimizer.zero_grad(set_to_none=True)

    render_pkg = render(
        viewpoint_cam,
        gaussians,
        pipe,
        background,
        deblur=use_deblur,
        blur_type=blur_type,
        image_id=image_id,
        lambda_s=opt.lambda_s,
        lambda_p=opt.lambda_p,
        max_clamp=opt.max_clamp,
        backend=opt.renderer_backend if blur_type == BLUR_SHARP else opt.renderer_backend_deblur,
    )

    image = render_pkg["render"]
    gt_image = viewpoint_cam.original_image.cuda()

    if opt.luminance_enable and iteration >= opt.luminance_start_iter:
        image_for_loss = luminance_model(image, image_id)
        lum_reg = luminance_model.regularization_loss(
            lambda_gain=opt.luminance_lambda_gain,
            lambda_bias=opt.luminance_lambda_bias,
        )
    else:
        image_for_loss = image
        lum_reg = torch.zeros((), device=image.device)

    photo_loss_raw = compute_photo_loss(image_for_loss, gt_image, opt)
    blur_weight = get_blur_type_weight(blur_type, opt)
    photo_loss_weighted = blur_weight * photo_loss_raw

    code_reg = compute_code_reg(render_pkg, opt)
    delta_reg = opt.lambda_delta * render_pkg.get("delta_reg", torch.zeros((), device=image.device))

    reg_loss = code_reg + delta_reg + lum_reg
    loss = photo_loss_weighted + reg_loss

    loss.backward()

    # 原始 3DGS densification stats 或 GDAGS blur stats
    if opt.gdags_stats_enable:
        gaussians.gdags.update_blur_stats(
            viewspace_points=render_pkg["gdags_stats_viewspace_points"],
            visibility=render_pkg["gdags_stats_visibility"],
            blur_type=blur_type,
        )
    else:
        # 原始 add_densification_stats 逻辑
        pass

    optimizer.step()
    if luminance_optimizer is not None and opt.luminance_enable and iteration >= opt.luminance_start_iter:
        luminance_optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    if luminance_optimizer is not None:
        luminance_optimizer.zero_grad(set_to_none=True)

    # GDAGS canonical probe：必须在 step 之后、zero_grad 之后执行
    if (
        opt.gdags_stats_enable
        and iteration >= opt.gdags_start_iter
        and iteration % opt.gdags_probe_interval == 0
        and len(sharp_cameras) > 0
    ):
        probe_cam = random.choice(sharp_cameras)
        run_gdags_canonical_probe(
            gaussians=gaussians,
            probe_cam=probe_cam,
            pipe=pipe,
            background=background,
            opt=opt,
            optimizer=optimizer,
            luminance_optimizer=luminance_optimizer,
        )

    # densify / prune
    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
        if opt.gdags_enable:
            split_mask, clone_mask, prune_mask = gaussians.gdags.decide(gaussians, iteration, opt)
            gaussians.gdags_densify_and_prune(
                split_mask=split_mask,
                clone_mask=clone_mask,
                prune_mask=prune_mask,
                iteration=iteration,
                opt=opt,
            )
        else:
            gaussians.densify_and_prune(...)

    if gaussians.gdags is not None:
        gaussians.gdags.gaussian_age += 1
        gaussians.gdags.assert_shape(gaussians.get_xyz.shape[0])

    log_training_metrics(...)
```

### 9.2 optimizer 顺序

推荐顺序：

```text
zero_grad
forward
loss
backward
stats update
optimizer.step
luminance_optimizer.step
zero_grad
canonical probe
zero_grad
densify/prune
assert_shape
```

canonical probe 放在 step 后的好处：

- 不影响当前 step。
- probe 后再 zero_grad，保证不影响下一次 step。
- 与正常训练梯度生命周期完全隔离。

---

## 10. 参数配置

### 10.1 mixed blur

```python
parser.add_argument("--mixed_blur", action="store_true")
parser.add_argument("--blur_label_path", type=str, default="")
parser.add_argument("--sharp_sample_ratio", type=float, default=0.34)
parser.add_argument("--motion_sample_ratio", type=float, default=0.33)
parser.add_argument("--defocus_sample_ratio", type=float, default=0.33)
```

### 10.2 blur loss weight

```python
parser.add_argument("--sharp_weight", type=float, default=1.0)
parser.add_argument("--motion_weight", type=float, default=1.0)
parser.add_argument("--defocus_weight", type=float, default=1.0)
```

### 10.3 luminance

```python
parser.add_argument("--luminance_enable", action="store_true")
parser.add_argument("--luminance_start_iter", type=int, default=1000)
parser.add_argument("--luminance_lr", type=float, default=1e-3)
parser.add_argument("--luminance_lambda_gain", type=float, default=5e-3)
parser.add_argument("--luminance_lambda_bias", type=float, default=1e-2)

parser.add_argument("--luminance_mode", type=str, default="exposure_gain_bias",
                    choices=["exposure_gain_bias", "rgb_scale_bias", "matrix", "tone_curve"])
parser.add_argument("--luminance_per_channel", action="store_true")
parser.add_argument("--luminance_matrix_enable", action="store_true")
parser.add_argument("--luminance_curve_enable", action="store_true")
```

第一版必须保持：

```text
luminance_mode = exposure_gain_bias
luminance_per_channel = false
luminance_matrix_enable = false
luminance_curve_enable = false
```

### 10.4 defocus

```python
parser.add_argument("--defocus_learn_rotation", action="store_true")
```

默认 false。

### 10.5 GDAGS

```python
parser.add_argument("--gdags_stats_enable", action="store_true")
parser.add_argument("--gdags_enable", action="store_true")
parser.add_argument("--gdags_start_iter", type=int, default=3000)
parser.add_argument("--gdags_probe_interval", type=int, default=100)
parser.add_argument("--gdags_probe_use_autograd_grad", action="store_true", default=True)

parser.add_argument("--gdags_clone_enable", action="store_true")
parser.add_argument("--gdags_split_enable", action="store_true")
parser.add_argument("--gdags_prune_enable", action="store_true")

parser.add_argument("--gdags_gcr_split_threshold", type=float, default=0.30)
parser.add_argument("--gdags_gcr_clone_threshold", type=float, default=0.85)
parser.add_argument("--gdags_min_canonical_count_for_prune", type=int, default=3)
parser.add_argument("--gdags_min_age_for_prune", type=int, default=1000)
parser.add_argument("--gdags_newborn_protect_iters", type=int, default=1000)
parser.add_argument("--gdags_eap_protect_iters", type=int, default=5000)
parser.add_argument("--gdags_nexus_protect_iters", type=int, default=1500)
parser.add_argument("--gdags_eap_global_protect_until", type=int, default=5000)
```

### 10.6 backend

```python
parser.add_argument("--renderer_backend", type=str, default="original",
                    choices=["original", "gsplat"])
parser.add_argument("--renderer_backend_deblur", type=str, default="original",
                    choices=["original", "gsplat"])
```

第一版：

```text
renderer_backend_deblur = original
```

---

## 11. Codex 实施任务拆分

### Task 1: 加 blur type 基础设施

文件：

```text
utils/blur_types.py
scene/cameras.py 或 camera 构造处
train.py
```

实现：

- `BlurType` enum。
- load `blur_labels.json`。
- 给所有 train camera 设置 `image_id`、`blur_type`。
- 打印分布统计。

验收：

```text
[Dataset] sharp=..., motion=..., defocus=...
```

### Task 2: 改 render routing

文件：

```text
gaussian_renderer/__init__.py
scene/gaussian_model.py
```

实现：

- `render(..., blur_type, image_id, return_identity_moment_only)`。
- sharp 不走 GTnet。
- motion 返回 `identity_moment` 和 `gdags_stats_viewspace_points`。
- defocus 默认只改 scale。

验收：

- sharp 路径结果与原始 not-deblur 一致。
- motion 不再出现 `moment0` 命名。
- defocus rotation 默认不变。

### Task 3: 改 Luminance

文件：

```text
scene/luminance_model.py
train.py
```

实现：

- `PerImageExposureModel` 只用 scalar log_gain + scalar brightness bias。
- loss 端调用。
- 记录 luminance reg、gain/bias norm。

验收：

- 初始化时等价 identity。
- 关闭 Luminance 时训练结果完全不受影响。
- 开启后 gain / bias 范围受限。

### Task 4: 改 loss 结构

文件：

```text
train.py
utils/loss_utils.py 可选
```

实现：

- `photo_loss_raw`
- `photo_loss_weighted`
- `reg_loss`
- `loss = photo_loss_weighted + reg_loss`
- 分 blur_type 日志。

验收：

- 日志能看到 raw 和 weighted 两套 loss。
- 正则项不随 blur type weight 改变。

### Task 5: GDAGS stats manager

文件：

```text
scene/gdags.py
scene/gaussian_model.py
train.py
```

实现：

- `DeblurAwareGDAGS`。
- `update_blur_stats(..., grad=None)`。
- `update_canonical_stats(..., grad=None)`。
- `run_gdags_canonical_probe` 用 `torch.autograd.grad`。
- `assert_shape`。

验收：

- canonical probe 后 Gaussian 参数不变。
- canonical probe 后 optimizer param grad 为空或全 0。
- stats buffer shape 始终等于 `gaussians.get_xyz.shape[0]`。

### Task 6: densify/prune buffer 同步

文件：

```text
scene/gaussian_model.py
scene/gdags.py
```

实现：

- `on_clone(parent_idx, current_iter, protect_iters)`。
- `on_split(parent_idx_repeated, current_iter, protect_iters)`。
- `on_prune(keep_mask)`。
- `on_external_add(num_new, source_type, current_iter, protect_iters)`。

验收：

- clone 后所有 GDAGS buffer 增加 K 行。
- split 后所有 GDAGS buffer 增加 children 行。
- prune 后所有 GDAGS buffer 和 Gaussian 数量一致。
- 没有 shape mismatch。

### Task 7: GDAGS density control

实现：

- stats-only 模式先跑。
- 再接入 clone / split / prune。
- prune 需要 canonical count、age、protect 三重限制。

验收：

- EAP / Nexus 新生点不会马上被 prune。
- Gaussian 数量曲线没有突然断崖式下降。
- canonical_grad_count 不足时不 prune。

---

## 12. 必须写的测试

### 12.1 probe 梯度隔离测试

伪代码：

```python
params_before = [p.detach().clone() for p in gaussians.parameters() if p.requires_grad]

run_gdags_canonical_probe(...)

params_after = [p.detach().clone() for p in gaussians.parameters() if p.requires_grad]

for a, b in zip(params_before, params_after):
    assert torch.allclose(a, b)

for group in optimizer.param_groups:
    for p in group["params"]:
        assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))
```

### 12.2 buffer shape 测试

```python
N0 = gaussians.get_xyz.shape[0]
gaussians.gdags.assert_shape(N0)

gaussians.densify_and_clone(...)
gaussians.gdags.assert_shape(gaussians.get_xyz.shape[0])

gaussians.densify_and_split(...)
gaussians.gdags.assert_shape(gaussians.get_xyz.shape[0])

gaussians.prune_points(...)
gaussians.gdags.assert_shape(gaussians.get_xyz.shape[0])
```

### 12.3 motion identity_moment 测试

```python
pkg = render(..., blur_type=BLUR_MOTION, return_identity_moment_only=True)
assert "identity_moment" in pkg
assert "gdags_stats_viewspace_points" in pkg
```

代码中不得出现：

```text
moment0
moment_0
moment-0
```

### 12.4 defocus quaternion 测试

仅在 `defocus_learn_rotation=true` 时运行：

```python
new_rotations = apply_safe_rotation_delta(rotations, r_delta)
assert torch.allclose(
    new_rotations.norm(dim=-1),
    torch.ones_like(new_rotations[..., 0]),
    atol=1e-3,
)
```

### 12.5 luminance identity 测试

```python
model = PerImageExposureModel(num_images=10).cuda()
img = torch.rand(3, 64, 64, device="cuda")
out = model(img, image_id=0)
assert torch.allclose(out, img, atol=1e-6)
```

如果因为 clamp 导致极端值不完全一致，需要保证输入在 `[0, 1]` 且初始化输出一致。

### 12.6 loss 分解测试

```python
assert torch.allclose(loss, photo_loss_weighted + reg_loss)
assert reg_loss does not change when only sharp_weight changes
```

---

## 13. 日志与调试指标

### 13.1 每 500 iter 打印

```text
[Iter 05000]
blur_type=motion
photo_raw=...
photo_weighted=...
reg/code=...
reg/delta=...
reg/lum=...
z_norm=...
delta_pos_mean=...
delta_scale_mean=...
lum_gain_max=...
lum_bias_abs_mean=...
gdags/canonical_count_mean=...
gdags/gcr_mean=...
gaussians/N=...
```

### 13.2 必须报警的情况

```python
if abs(luminance.log_gain).max() > 0.30:
    warn("Luminance gain reached clamp boundary; may be absorbing blur residual.")

if render_pkg["delta_reg"].item() > opt.delta_reg_warn_threshold:
    warn("GTnet delta too large.")

if gaussians.gdags is not None:
    gaussians.gdags.assert_shape(gaussians.get_xyz.shape[0])
```

### 13.3 canonical probe 日志

```text
[GDAGS Probe] cam=00012.png blur_type=sharp grad_source=autograd.grad canonical_visible=...
```

如果不是 sharp 图：

```text
[GDAGS Probe][WARN] using fallback high-sharpness camera because no sharp label exists
```

---

## 14. 最终验收标准

### 14.1 Phase 1 验收

- sharp 分支与原始 render 对齐。
- motion / defocus 不崩溃。
- `blur_type` 采样分布符合日志。
- 不开 Luminance / GDAGS 时，训练曲线可复现。

### 14.2 Phase 2 验收

- Luminance 初始化等价 identity。
- `log_gain` 和 `bias` 不长期顶到 clamp 边界。
- `photo_loss_raw` 与 `photo_loss_weighted` 都可见。
- GTnet delta 没有因为 Luminance 开启而明显塌缩。

### 14.3 Phase 3 验收

- canonical probe 不改变参数。
- canonical probe 不留下 optimizer grad。
- `canonical_grad_count` 正常增长。
- 所有 GDAGS buffer shape 与 Gaussian 数量一致。

### 14.4 Phase 4 验收

- clone / split / prune 后无 shape mismatch。
- 新生点在保护期内不会被 prune。
- Gaussian 数量没有异常断崖式下降。
- EAP 点前 5000 iter 不被大量 prune。
- prune 只在 canonical stats 足够时发生。

### 14.5 最终指标验收

至少比较：

```text
baseline original
+ mixed blur routing
+ luminance exposure gain bias
+ GDAGS stats only
+ GDAGS density control
```

每阶段记录：

```text
PSNR
SSIM
LPIPS
训练耗时
Gaussian 数量
显存峰值
sharp/motion/defocus 分桶 photo_loss
```

---

## 15. 实现红线

Codex 写代码时必须遵守：

1. canonical probe 默认用 `torch.autograd.grad`，不允许污染 optimizer grad。
2. probe fallback backward 前后必须 `optimizer.zero_grad(set_to_none=True)`，且不能 step。
3. GDAGS 所有 buffer 必须随 Gaussian clone / split / prune 同步。
4. motion 原始位置渲染只能叫 `identity_moment` 或 `canonical_moment`，禁止叫 `moment0`。
5. defocus 第一版不学 rotation；如果学 rotation，必须 normalize quaternion。
6. Luminance 第一版只用 per-image scalar exposure gain + scalar brightness bias，禁止 per-channel RGB 自由度。
7. blur_type 权重只乘 photo loss，不乘正则项。
8. 日志必须同时记录 `photo_loss_raw` 和 `photo_loss_weighted`。
9. `gdags_stats_viewspace_points` 必须是显式返回字段，不能依赖 list 的 `[-1]`。
10. 每次 densify / prune 后必须 `assert_shape`。

---

## 16. 给 Codex 的最短执行提示

可以把下面这段直接贴给 Codex：

```text
请按 final_mixed_blur_3dgs_codex_plan.md 实现，不要自行发挥。

优先级：
1. blur_type / image_id / mixed blur routing
2. render 返回 identity_moment 与 gdags_stats_viewspace_points
3. Luminance 只做 per-image scalar log_gain + brightness bias
4. loss 拆成 photo_loss_raw / photo_loss_weighted / reg_loss
5. GDAGS canonical probe 用 torch.autograd.grad，禁止污染 optimizer grad
6. GDAGS buffers 实现 on_clone / on_split / on_prune / on_external_add
7. 最后才启用 GDAGS density control

硬约束：
- 不要写 moment0；原始位置渲染叫 identity_moment。
- defocus rotation 默认关闭。
- 3x3 luminance matrix 和 tone curve 默认关闭。
- 每次 Gaussian 数量变化后 assert GDAGS buffer shape。
- probe 后不得 optimizer.step。
```
