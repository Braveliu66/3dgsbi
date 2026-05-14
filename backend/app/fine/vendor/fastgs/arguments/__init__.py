#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[5]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.fine.fastgs_defaults import (
    FASTGS_DATA_DEVICE,
    FASTGS_DEBLUR_BLURRED_VIEWS_ONLY,
    FASTGS_DEBLUR_BLUR_REGISTRY,
    FASTGS_DEBLUR_ENABLED,
    FASTGS_DEBLUR_GTNET_LR,
    FASTGS_DEBLUR_HIDDEN,
    FASTGS_DEBLUR_LAMBDA_P,
    FASTGS_DEBLUR_LAMBDA_S,
    FASTGS_DEBLUR_MAX_CLAMP,
    FASTGS_DEBLUR_MAX_POSITION_DELTA,
    FASTGS_DEBLUR_MODE,
    FASTGS_DEBLUR_NUM_MOMENTS,
    FASTGS_DEBLUR_TRANSFORM_REG_WEIGHT,
    FASTGS_DEBLUR_WARMUP_ITERS,
    FASTGS_DEBLUR_WIDTH,
    FASTGS_DEBLUR_XYZ_LR_SCALE,
    FASTGS_DENSE,
    FASTGS_DENSIFICATION_INTERVAL,
    FASTGS_DENSIFY_FROM_ITER,
    FASTGS_DENSIFY_GRAD_THRESHOLD,
    FASTGS_DENSIFY_UNTIL_ITER,
    FASTGS_FEATURE_LR,
    FASTGS_FINAL_PRUNE_MIN_OPACITY,
    FASTGS_FINAL_PRUNE_SCORE_THRESH,
    FASTGS_GRAD_ABS_THRESH,
    FASTGS_GRAD_THRESH,
    FASTGS_HIGHFEATURE_LR,
    FASTGS_ITERATIONS,
    FASTGS_LAMBDA_DSSIM,
    FASTGS_LATE_PRUNE_ENABLED,
    FASTGS_LATE_PRUNE_FROM_ITER,
    FASTGS_LATE_PRUNE_INTERVAL,
    FASTGS_LATE_PRUNE_MIN_OPACITY,
    FASTGS_LATE_PRUNE_SCORE_THRESH,
    FASTGS_LATE_PRUNE_UNTIL_ITER,
    FASTGS_LOSS_THRESH,
    FASTGS_LOWFEATURE_LR,
    FASTGS_MULT,
    FASTGS_OPACITY_LR,
    FASTGS_OPACITY_RESET_INTERVAL,
    FASTGS_OPTIMIZER_TYPE,
    FASTGS_PERCENT_DENSE,
    FASTGS_POSITION_LR_DELAY_MULT,
    FASTGS_POSITION_LR_FINAL,
    FASTGS_POSITION_LR_INIT,
    FASTGS_POSITION_LR_MAX_STEPS,
    FASTGS_RANDOM_BACKGROUND,
    FASTGS_RESOLUTION,
    FASTGS_ROTATION_LR,
    FASTGS_SAMPLE_CAMERAS,
    FASTGS_SCALING_LR,
    FASTGS_SH_DEGREE,
    FASTGS_SHFEATURE_LR,
)

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = FASTGS_SH_DEGREE
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = FASTGS_RESOLUTION
        self._white_background = False
        self.data_device = FASTGS_DATA_DEVICE
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = FASTGS_ITERATIONS
        self.position_lr_init = FASTGS_POSITION_LR_INIT
        self.position_lr_final = FASTGS_POSITION_LR_FINAL
        self.position_lr_delay_mult = FASTGS_POSITION_LR_DELAY_MULT
        self.position_lr_max_steps = FASTGS_POSITION_LR_MAX_STEPS
        self.feature_lr = FASTGS_FEATURE_LR
        self.shfeature_lr = FASTGS_SHFEATURE_LR
        self.opacity_lr = FASTGS_OPACITY_LR
        self.scaling_lr = FASTGS_SCALING_LR
        self.rotation_lr = FASTGS_ROTATION_LR
        self.percent_dense = FASTGS_PERCENT_DENSE
        self.lambda_dssim = FASTGS_LAMBDA_DSSIM
        self.densification_interval = FASTGS_DENSIFICATION_INTERVAL
        self.opacity_reset_interval = FASTGS_OPACITY_RESET_INTERVAL
        self.densify_from_iter = FASTGS_DENSIFY_FROM_ITER
        self.densify_until_iter = FASTGS_DENSIFY_UNTIL_ITER
        self.densify_grad_threshold = FASTGS_DENSIFY_GRAD_THRESHOLD
        
        # fastgs parameters
        self.loss_thresh = FASTGS_LOSS_THRESH
        self.fastgs_sample_cameras = FASTGS_SAMPLE_CAMERAS
        self.grad_abs_thresh = FASTGS_GRAD_ABS_THRESH
        self.highfeature_lr = FASTGS_HIGHFEATURE_LR
        self.lowfeature_lr = FASTGS_LOWFEATURE_LR
        self.grad_thresh = FASTGS_GRAD_THRESH
        self.dense = FASTGS_DENSE
        self.mult = FASTGS_MULT

        # Deblurring-3DGS GTnet training-time blur model, adapted to FastGS.
        self.deblur_enabled = FASTGS_DEBLUR_ENABLED
        self.deblur_mode = FASTGS_DEBLUR_MODE
        self.deblur_blur_registry = FASTGS_DEBLUR_BLUR_REGISTRY
        self.deblur_warmup_iters = FASTGS_DEBLUR_WARMUP_ITERS
        self.deblur_num_moments = FASTGS_DEBLUR_NUM_MOMENTS
        self.deblur_gtnet_lr = FASTGS_DEBLUR_GTNET_LR
        self.deblur_hidden = FASTGS_DEBLUR_HIDDEN
        self.deblur_width = FASTGS_DEBLUR_WIDTH
        self.deblur_lambda_s = FASTGS_DEBLUR_LAMBDA_S
        self.deblur_lambda_p = FASTGS_DEBLUR_LAMBDA_P
        self.deblur_max_clamp = FASTGS_DEBLUR_MAX_CLAMP
        self.deblur_max_position_delta = FASTGS_DEBLUR_MAX_POSITION_DELTA
        self.deblur_transform_reg_weight = FASTGS_DEBLUR_TRANSFORM_REG_WEIGHT
        self.deblur_xyz_lr_scale = FASTGS_DEBLUR_XYZ_LR_SCALE
        self.deblur_blurred_views_only = FASTGS_DEBLUR_BLURRED_VIEWS_ONLY
        self.fastgs_final_prune_min_opacity = FASTGS_FINAL_PRUNE_MIN_OPACITY
        self.fastgs_final_prune_score_thresh = FASTGS_FINAL_PRUNE_SCORE_THRESH
        self.fastgs_late_prune_enabled = "true" if FASTGS_LATE_PRUNE_ENABLED else "false"
        self.fastgs_late_prune_interval = FASTGS_LATE_PRUNE_INTERVAL
        self.fastgs_late_prune_from_iter = FASTGS_LATE_PRUNE_FROM_ITER
        self.fastgs_late_prune_until_iter = FASTGS_LATE_PRUNE_UNTIL_ITER
        self.fastgs_late_prune_min_opacity = FASTGS_LATE_PRUNE_MIN_OPACITY
        self.fastgs_late_prune_score_thresh = FASTGS_LATE_PRUNE_SCORE_THRESH

        self.random_background = FASTGS_RANDOM_BACKGROUND
        self.optimizer_type = FASTGS_OPTIMIZER_TYPE
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
