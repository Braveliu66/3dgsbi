import math

import torch

from utils.general_utils import get_expon_lr_func


class TrainingScheduler:
    """Dash-style frequency scheduler adapted for Deblur training."""

    def __init__(self, opt, pipe, gaussians, original_images):
        self.max_steps = opt.iterations
        self.init_n_gaussian = int(gaussians.get_xyz.shape[0])
        self.densify_mode = getattr(pipe, "densify_mode", getattr(opt, "densify_mode", "free"))
        self.densify_until_iter = int(opt.densify_until_iter)
        self.densification_interval = int(opt.densification_interval)
        self.resolution_mode = getattr(pipe, "resolution_mode", getattr(opt, "resolution_mode", "const"))

        self.start_significance_factor = float(getattr(opt, "dash_start_significance_factor", 4))
        self.max_reso_scale = max(1, int(getattr(opt, "dash_max_reso_scale", 4)))
        self.reso_sample_num = 32
        self.max_densify_rate_per_step = float(getattr(opt, "dash_max_densify_rate_per_step", 0.10))
        self.reso_scales = None
        self.reso_level_significance = None
        self.reso_level_begin = None
        self.increase_reso_until = self.densify_until_iter
        self.next_i = 2

        max_n_gaussian = int(getattr(pipe, "max_n_gaussian", getattr(opt, "max_n_gaussian", -1)))
        if max_n_gaussian > 0:
            self.max_n_gaussian = max_n_gaussian
            self.momentum = -1
        else:
            self.momentum = 5 * self.init_n_gaussian
            self.max_n_gaussian = self.init_n_gaussian + self.momentum
            self.integrate_factor = 0.98
            self.momentum_step_cap = 1_000_000

        self.init_reso_scheduler(original_images)
        if self.resolution_mode == "freq":
            gaussians.xyz_scheduler_args = get_expon_lr_func(
                lr_init=opt.position_lr_init * gaussians.spatial_lr_scale,
                lr_final=opt.position_lr_final * gaussians.spatial_lr_scale,
                lr_delay_mult=opt.position_lr_delay_mult,
                max_steps=opt.position_lr_max_steps,
                decay_from_iter=self.lr_decay_from_iter(),
            )

    def update_momentum(self, momentum_step):
        if self.momentum == -1:
            return
        self.momentum = max(
            self.momentum,
            int(self.integrate_factor * self.momentum + min(self.momentum_step_cap, int(momentum_step))),
        )
        self.max_n_gaussian = self.init_n_gaussian + self.momentum

    def get_res_scale(self, iteration):
        if self.resolution_mode == "const":
            return 1
        if self.resolution_mode != "freq":
            raise NotImplementedError(f"Resolution mode '{self.resolution_mode}' is not implemented.")
        if iteration >= self.increase_reso_until:
            return 1
        if iteration < self.reso_level_begin[1]:
            return int(self.reso_scales[0])
        while iteration >= self.reso_level_begin[self.next_i]:
            self.next_i += 1
        i = self.next_i - 1
        i_now, i_nxt = self.reso_level_begin[i : i + 2]
        s_lst, s_now = self.reso_scales[i - 1 : i + 1]
        scale = (1 / ((iteration - i_now) / (i_nxt - i_now) * (1 / s_now**2 - 1 / s_lst**2) + 1 / s_lst**2)) ** 0.5
        return max(1, int(scale))

    def get_densify_rate(self, iteration, cur_n_gaussian, cur_scale=None):
        if self.densify_mode == "free":
            return 1.0
        if self.densify_mode != "freq":
            raise NotImplementedError(f"Densify mode '{self.densify_mode}' is not implemented.")
        if cur_scale is None:
            raise ValueError("cur_scale is required when densify_mode='freq'")
        if self.densification_interval + iteration < self.increase_reso_until:
            next_n_gaussian = int(
                (self.max_n_gaussian - self.init_n_gaussian)
                / max(float(cur_scale), 1.0) ** (2 - iteration / self.densify_until_iter)
            ) + self.init_n_gaussian
        else:
            next_n_gaussian = self.max_n_gaussian
        return min(max((next_n_gaussian - cur_n_gaussian) / max(cur_n_gaussian, 1), 0.0), self.max_densify_rate_per_step)

    def lr_decay_from_iter(self):
        if self.resolution_mode == "const":
            return 1
        for iteration, scale in zip(self.reso_level_begin, self.reso_scales):
            if scale < 2:
                return iteration
        return self.increase_reso_until

    def init_reso_scheduler(self, original_images):
        if self.resolution_mode != "freq":
            print(f"[INFO] Skipped resolution scheduler initialization, resolution_mode={self.resolution_mode}")
            return

        def compute_win_significance(significance_map, scale):
            h, w = significance_map.shape[-2:]
            c = ((h + 1) // 2, (w + 1) // 2)
            win_size = (max(1, int(h / scale)), max(1, int(w / scale)))
            return significance_map[..., c[0] - win_size[0] // 2 : c[0] + win_size[0] // 2, c[1] - win_size[1] // 2 : c[1] + win_size[1] // 2].sum().item()

        def scale_solver(significance_map, target_significance):
            left, right = 0.0, 1.0
            mid = 1.0
            for _ in range(64):
                mid = (left + right) / 2
                if compute_win_significance(significance_map, 1 / max(mid, 1e-6)) < target_significance:
                    left = mid
                else:
                    right = mid
            return 1 / max(mid, 1e-6)

        print("[INFO] Initializing Dash frequency resolution scheduler")
        self.next_i = 2
        scene_freq_image = None
        max_reso_scale = float(self.max_reso_scale)
        for img in original_images:
            img_fft_centered = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
            img_fft_mod = (img_fft_centered.real.square() + img_fft_centered.imag.square()).sqrt()
            scene_freq_image = img_fft_mod if scene_freq_image is None else scene_freq_image + img_fft_mod
            e_total = img_fft_mod.sum().item()
            e_min = e_total / max(self.start_significance_factor, 1.0)
            max_reso_scale = min(max_reso_scale, scale_solver(img_fft_mod, e_min))

        self.max_reso_scale = max(1.0, max_reso_scale)
        self.reso_scales = []
        self.reso_level_significance = []
        self.reso_level_begin = []
        scene_freq_image /= max(len(original_images), 1)
        e_total = scene_freq_image.sum().item()
        e_min = max(compute_win_significance(scene_freq_image, self.max_reso_scale), 1e-6)
        modulation_total = max(math.log(max(e_total / e_min, 1.000001)), 1e-6)

        self.reso_level_significance.append(e_min)
        self.reso_scales.append(self.max_reso_scale)
        self.reso_level_begin.append(0)
        for i in range(1, self.reso_sample_num - 1):
            sig = (e_total - e_min) * i / (self.reso_sample_num - 1) + e_min
            self.reso_scales.append(scale_solver(scene_freq_image, sig))
            self.reso_level_significance.append(math.log(max(self.reso_level_significance[-1] / e_min, 1.000001)))
            self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-1] / modulation_total))
        self.reso_level_significance.append(modulation_total)
        self.reso_scales.append(1.0)
        self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-2] / modulation_total))
        self.reso_level_begin.append(self.increase_reso_until)


class DeblurDashScheduler(TrainingScheduler):
    def enabled_for_resolution(self, iteration, start_iter):
        return iteration >= start_iter
