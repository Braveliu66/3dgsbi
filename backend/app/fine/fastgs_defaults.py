from __future__ import annotations


# ============================================================
# Quality defaults for official FastGS-Big image fine reconstruction.
# 这是 FastGS-Big 精修阶段的默认质量/速度参数。
# 建议把这里当成唯一配置源，避免多个地方参数不一致。
# ============================================================


# 不同场景 profile 对应的最大输入图像边长。
# mixed_balanced: 通用平衡模式，默认 1080，速度更快。
# indoor_full: 室内高质量模式，1500，细节更好但更慢。
# outdoor_fast_clean: 室外快速干净模式，1080，适合大场景快速出结果。
FINE_SCENE_PROFILE_MAX_SIDES = {
    "mixed_balanced": 1080,
    "indoor_full": 1500,
    "outdoor_fast_clean": 1080,
}

# 默认场景 profile。
# 如果没有手动指定，就用 mixed_balanced。
# 如果你追求室内小场景质量，建议运行时切到 indoor_full。
DEFAULT_FINE_SCENE_PROFILE = "mixed_balanced"
FASTGS_SCENE_TYPE = "auto"

# 精修输入图像最大边长。
# 1500 质量较好，但比 1080 慢。
# 对 Deblur/室内细节建议 1500；预览可降到 1080。
FINE_IMAGE_MAX_SIDE = 1500

# 总训练轮数。
# 30000 是质量优先设置，比 15000 更稳、更接近官方 3DGS/Deblurring-3DGS 训练强度。
FINE_ITERATIONS = 40_000

# FastGS 实际训练轮数，保持和 FINE_ITERATIONS 一致。
FASTGS_ITERATIONS = FINE_ITERATIONS


# ============================================================
# COLMAP / SfM 参数
# ============================================================

# COLMAP SIFT 每张图最多提取的特征点数量。
# 20000 对模糊图/少图场景比较有帮助，可以提升注册和 sparse points 数量。
COLMAP_SIFT_MAX_NUM_FEATURES = 65_536

# COLMAP 处理图像时的最大边长。
# 和训练图像边长一致，保证位姿/点云和训练分辨率匹配。
COLMAP_MAX_IMAGE_SIZE = 3_200
COLMAP_SIFT_PEAK_THRESHOLD = 0.002
COLMAP_SIFT_EDGE_THRESHOLD = 12
COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE = True
COLMAP_SIFT_DOMAIN_SIZE_POOLING = True
COLMAP_GUIDED_MATCHING = True
COLMAP_SIFT_MATCH_MAX_RATIO = 0.85
COLMAP_MIN_SPARSE_POINTS = 15_000
COLMAP_TARGET_SPARSE_POINTS = 30_000

# COLMAP 使用线程数。
# 8 是比较稳的默认值，机器 CPU 多可以提高。
COLMAP_THREADS = 16

# COLMAP 匹配器。
# auto: 根据图片数量/场景自动选择。
# exhaustive: 少图场景更稳但慢。
# sequential: 视频抽帧/顺序拍摄更快。
COLMAP_MATCHER = "exhaustive"

# 最小注册成功比例。
# None 表示使用系统默认。
# 例如 0.7 表示至少 70% 图片注册成功才继续。
COLMAP_MIN_REGISTERED_RATIO = None


# ============================================================
# FastGS 基础渲染/优化参数
# ============================================================

# 球谐函数阶数。
# 3 是 3DGS 常用值，颜色表现更好；降低会快但颜色细节差。
FASTGS_SH_DEGREE = 3

# FastGS 训练分辨率。
# 和 FINE_IMAGE_MAX_SIDE 保持一致。
FASTGS_RESOLUTION = FINE_IMAGE_MAX_SIDE

# 数据放置设备。
# cuda 表示图像/训练数据放 GPU，速度快但显存占用高。
FASTGS_DATA_DEVICE = "cuda"

# optimizer 类型。
# default 是官方 FastGS 默认优化器；Deblur 目前建议保持 default。
FASTGS_OPTIMIZER_TYPE = "default"

# 是否使用随机背景。
# False 更适合真实图像重建；随机背景更多用于带 alpha/mask 的对象重建。
FASTGS_RANDOM_BACKGROUND = False


# ============================================================
# Gaussian 参数学习率
# ============================================================

# Gaussian xyz 初始学习率。
# 影响点的位置移动速度；太大会漂，太小收敛慢。
FASTGS_POSITION_LR_INIT = 0.00016

# Gaussian xyz 最终学习率。
# 训练后期位置微调用，通常比初始小很多。
FASTGS_POSITION_LR_FINAL = 0.0000016

# xyz 学习率延迟倍率。
# 0.01 表示前期更保守，避免点云一开始剧烈漂移。
FASTGS_POSITION_LR_DELAY_MULT = 0.01

# xyz 学习率衰减总步数。
# 必须和总训练轮数同步，否则 30000 轮时后半段学习率调度不合理。
FASTGS_POSITION_LR_MAX_STEPS = FINE_ITERATIONS

# DC/color feature 学习率。
# 控制基础颜色更新速度。
FASTGS_FEATURE_LR = 0.0025

# SH feature 学习率。
# 控制高阶颜色/视角相关颜色学习。
FASTGS_SHFEATURE_LR = 0.005

# FastGS-Big 里的 high feature 学习率。
# 对细节和颜色表达有影响，太高可能有彩色噪点。
FASTGS_HIGHFEATURE_LR = 0.02

# FastGS-Big 里的 low feature 学习率。
# 控制低频颜色/特征，通常保持默认。
FASTGS_LOWFEATURE_LR = 0.0025

# opacity 学习率。
# 控制透明度收敛速度；太高容易 opacity 变化剧烈。
FASTGS_OPACITY_LR = 0.025

# scale 学习率。
# 控制 Gaussian 尺度更新；太大会出现大 splat / 糊 / 飞点。
FASTGS_SCALING_LR = 0.005

# rotation 学习率。
# 控制 Gaussian 旋转更新。
FASTGS_ROTATION_LR = 0.001


# ============================================================
# FastGS densify / VCD / VCP 相关参数
# ============================================================

# percent_dense 是 3DGS/FastGS densify 判断中的场景比例参数。
# 越大越容易把较大 Gaussian 也纳入 densify。
FASTGS_PERCENT_DENSE = 0.004

# dense 控制超过场景 extent 一定比例的 Gaussian 是否强制 densify。
# 0.005 是较保守的质量设置。
FASTGS_DENSE = 0.01

# compact box / splat tile 相关倍率。
# 影响 FastGS compact 策略；越小越紧，可能更快但风险更高。
FASTGS_MULT = 0.7

# DSSIM loss 权重。
# 总 loss 通常是 L1 + lambda_dssim * DSSIM。
# 0.2 是常用默认，提升结构相似度。
FASTGS_LAMBDA_DSSIM = 0.2

# FastGS VCD/VCP 的 loss 阈值。
# 越低，更多区域会被认为有误差，可能保留/加密更多 Gaussians。
# 质量优先可试 0.05；默认 0.1 比较平衡。
FASTGS_LOSS_THRESH = 0.05

# Deblur-aware VCD blends FastGS multi-view error with base Gaussian gradients.
FASTGS_VCD_BLEND_ALPHA = 0.6
FASTGS_VCD_SCORE_THRESH = 0.3

# Deblur-aware VCP lowers score-based prune pressure in regions where GTnet
# predicts stronger training-time blur. Opacity/scale prune are not protected.
FASTGS_VCP_BLUR_PROTECT_WEIGHT = 0.65

# clone 类型 densify 梯度阈值。
# 越低越容易 clone/加密，点更多，细节可能更好但更慢/更易飞点。
FASTGS_GRAD_THRESH = 0.0005

# split 类型 densify 的绝对梯度阈值。
# 越低越容易 split；0.00035 是较常用平衡值。
FASTGS_GRAD_ABS_THRESH = 0.0002

# FastGS densify_grad_threshold。
# 质量优先建议 0.0002；motion blur 场景可以试 0.0005。
FASTGS_DENSIFY_GRAD_THRESHOLD = 0.0001

# 每隔多少 iter 做一次 densify 检查。
# 100 表示每 100 轮加密/剪枝判断一次。
FASTGS_DENSIFICATION_INTERVAL = 100

# opacity reset 间隔。
# 100000 基本等于 30000 训练中不 reset。
# Deblur 场景建议禁用中后期 reset，避免 loss 暴涨或模型变糊。
FASTGS_OPACITY_RESET_INTERVAL = 100_000

# Decouple large-splat size pruning from opacity reset.  Opacity reset stays
# effectively disabled for deblur runs, but large blobs still need pruning.
FASTGS_SIZE_PRUNE_FROM_ITER = 12_000
FASTGS_SIZE_PRUNE_MAX_SCREEN_SIZE = 12
FASTGS_SIZE_PRUNE_MAX_WORLD_SCALE_RATIO = 0.18

# densify 从第几轮开始。
# 500 是标准设置，让初始点先稳定一点再加密。
FASTGS_DENSIFY_FROM_ITER = 500

# densify 到第几轮结束。
# 15000 对 30000 轮训练是质量优先设置。
# 点云少/细节不足时可以到 18000 或 20000。
FASTGS_DENSIFY_UNTIL_ITER = 34_000

# VCD/VCP 采样多少个相机计算多视角 score。
# 12 对你这种少图场景等于基本全采样，比较稳。
FASTGS_SAMPLE_CAMERAS = 10

# VCD 使用的误差百分位。
# 0.60 表示以较高误差区域作为加密判断依据。
# 太低会加很多点，太高可能加不够。
FASTGS_VCD_PERCENTILE = 0.60

# compact box 倍率，和 FASTGS_MULT 保持一致。
FASTGS_COMPACT_BOX_MULT = FASTGS_MULT


# ============================================================
# FastGS late prune / final prune
# ============================================================

# 是否启用训练后期 late prune。
# Deblur 质量优先建议先 False，避免把细节剪掉。
FASTGS_LATE_PRUNE_ENABLED = False

# late prune 间隔。
# 如果启用，则每 3000 iter 做一次 late prune。
FASTGS_LATE_PRUNE_INTERVAL = 4_000

# late prune 从第几轮开始。
# 27000 表示只在最后 10% 左右开始轻剪。
FASTGS_LATE_PRUNE_FROM_ITER = 28_000

# late prune 到第几轮结束。

FASTGS_LATE_PRUNE_UNTIL_ITER = 30_000

# late prune 的 opacity 阈值。
# 0.005 是轻剪；0.02 以上会更强，可能剪掉远端细节。
FASTGS_LATE_PRUNE_MIN_OPACITY = 0.001

# late prune 的 score 阈值。
# 1.0 基本最保守；0.95/0.98 会更积极剪。
FASTGS_LATE_PRUNE_SCORE_THRESH = 0.97
FASTGS_LATE_PRUNE_MAX_WORLD_SCALE_RATIO = 0.18
FASTGS_LATE_PRUNE_MAX_FRACTION = 0.02

# final prune 的 opacity 阈值。
# 0.001 很轻，质量优先推荐。
# 如果飞点很多可试 0.003 或 0.005，但可能损失细节。
FASTGS_FINAL_PRUNE_ENABLED = False
FASTGS_FINAL_PRUNE_MIN_OPACITY = 0.001
# final prune 的 score 阈值。
# 1.0 最保守，尽量不按 score 剪。
# 如果后处理飞点很多，可试 0.98 或 0.95。
FASTGS_FINAL_PRUNE_SCORE_THRESH = 0.95
FASTGS_FINAL_PRUNE_MAX_WORLD_SCALE_RATIO = 0.15


# ============================================================
# Deblurring-3DGS settings
# ============================================================

# Deblur 开关。
# 默认统一启用 GTnet；清晰图像由 transform 正则压向恒等变换。
# false 表示显式关闭。
FASTGS_DEBLUR_ENABLED = "true"

# Deblur 模式。
# defocus: 失焦/景深糊，最稳，不使用 position moments。
# motion: 运动模糊/拖影，使用位置扰动。
# mixed: defocus + motion 都开；默认不再按检测结果切换模式。
FASTGS_DEBLUR_MODE = "mixed"

# blur registry 路径。
# 空字符串表示由 pipeline 自动生成/传入。
# registry 记录每张图是否 blurred、kind、quality 等。
FASTGS_DEBLUR_BLUR_REGISTRY = ""

# 是否启用 Deblur 自动调度。
# true 表示用调度器控制 warmup、deblur loss、densify、prune。
FASTGS_DEBLUR_AUTO_SCHEDULE = "true"

# 调度 profile。
# quality: 质量优先，Deblur loss 不应中途关闭。
# balanced: 质量/速度折中。
# fast: 速度优先。
FASTGS_DEBLUR_SCHEDULE_PROFILE = "quality"

# Deblur warmup 轮数。
# 7000 表示前 7000 轮不走 Deblur，先用普通 FastGS 稳定几何和加密。
# 对 30000 轮来说 7000 偏保守但稳定；想更早去模糊可用 3000~5000。
FASTGS_DEBLUR_WARMUP_ITERS = 3_000

# Deblur motion/mixed 的 moments 数量。
# motion/mixed 模式下通常会多次 rasterize，5 质量高但慢。
# defocus 模式可设 0，因为不需要 motion moments。
FASTGS_DEBLUR_NUM_MOMENTS = 5

# GTnet 学习率。
# 官方风格通常 0.001。
FASTGS_DEBLUR_GTNET_LR = 0.001

# GTnet MLP 隐藏层数量。
# 3 是官方常用设置。
FASTGS_DEBLUR_HIDDEN = 3

# GTnet MLP 宽度。
# 64 是官方常用设置。
FASTGS_DEBLUR_WIDTH = 64

# scale/covariance blur 强度。
# 0.01 是官方常用值。
# defocus 主要靠这个。
FASTGS_DEBLUR_LAMBDA_S = 0.01

# position/motion blur 强度。
# motion/mixed 用 0.01。
FASTGS_DEBLUR_LAMBDA_P = 0.006

# GTnet 对 scale/rotation delta 的最大 clamp。
# 1.1 表示训练时最多把 Gaussian 扩到 1.1 倍左右。
# 太大会变糊/飞点，太小去模糊能力弱。
FASTGS_DEBLUR_MAX_CLAMP = 1.4

# position delta 最大位移。
# motion/mixed 用 0.02。
FASTGS_DEBLUR_MAX_POSITION_DELTA = 0.008

# GTnet transform 正则权重。
# 0.0: GTnet 自由度最大，去模糊强，但可能更容易飞点。
# 1e-6: 稍微约束 GTnet，通常更稳。
# 你现在设 1e-6，适合减少外部彩色长条/飞点。
FASTGS_DEBLUR_TRANSFORM_REG_WEIGHT = 0.006

# Deblur 阶段 xyz 学习率缩放。
# 0.5 表示位置更新减半，减少点被 Deblur 拉飞。
# 飞点多可试 0.3；细节长不出来可试 0.7。
FASTGS_DEBLUR_XYZ_LR_SCALE = 0.1

# 是否只有 blurred 图像走 Deblur。
# 默认 false：所有训练图像都走 Deblur，不再用 registry 预判模糊类型。
FASTGS_DEBLUR_BLURRED_VIEWS_ONLY = "true"

# Keep GTnet as a training-time blur renderer only.  The final stage disables
# deblur and fine-tunes ordinary Gaussians from clear frames for PLY/SPZ export.
FASTGS_DEBLUR_SHARP_REFINE_ENABLED = "true"
FASTGS_DEBLUR_SHARP_REFINE_FROM_ITER = 36_000
FASTGS_DEBLUR_SHARP_REFINE_CLEAR_ONLY = "false"

# Extra points are risky on mixed sharp/blurred captures because they clone from
# blurred-view gradients without the usual multiview score filter.
FASTGS_DEBLUR_EXTRA_POINTS_ENABLED = "true"
FASTGS_DEBLUR_EXTRA_POINTS_MANDATORY = "true"
FASTGS_DEBLUR_EXTRA_POINTS_WEAK_TARGET = 200_000
FASTGS_DEBLUR_EXTRA_POINTS_TARGET = 100_000

# VCD should follow the active GTnet blur renderer by default so densification
# sees the same blurred observation model as the training loss.
FASTGS_DEBLUR_TOPOLOGY_SHARP_ONLY = "false"

# 是否启用 Deblur 后期二次 densify。
# false: 中后期不再额外加密，更稳，飞点更少。
# true: 中后期再加密一段，细节可能更多，但更容易长飞点。
# 注意：这里不能有逗号。
FASTGS_DEBLUR_LATE_DENSIFY_ENABLED = "true"
