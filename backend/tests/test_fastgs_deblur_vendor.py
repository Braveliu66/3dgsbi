from __future__ import annotations

import sys
import unittest
import importlib.util
from argparse import ArgumentParser
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
FASTGS_ROOT = BACKEND_ROOT / "app" / "fine" / "vendor" / "fastgs"
sys.path.insert(0, str(FASTGS_ROOT))

try:
    import torch

    spec = importlib.util.spec_from_file_location("fastgs_blur_kernel_test", FASTGS_ROOT / "scene" / "blur_kernel.py")
    if spec is None or spec.loader is None:
        raise ImportError("failed to load FastGS blur_kernel.py")
    blur_kernel = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = blur_kernel
    spec.loader.exec_module(blur_kernel)
    GTnet = blur_kernel.GTnet
    DeblurConfig = blur_kernel.DeblurConfig
    DeblurState = blur_kernel.DeblurState
    compute_blur_indicator = blur_kernel.compute_blur_indicator
    deblur_transform_regularization = blur_kernel.deblur_transform_regularization
    predict_deblur_transforms = blur_kernel.predict_deblur_transforms
except Exception as exc:  # pragma: no cover - depends on local torch availability
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class FastGSDeblurSourceTests(unittest.TestCase):
    def test_deblur_renderer_uses_fastgs_rasterizer_abi(self) -> None:
        source = (FASTGS_ROOT / "gaussian_renderer" / "deblur.py").read_text(encoding="utf-8")

        self.assertIn("diff_gaussian_rasterization_fastgs", source)
        self.assertNotIn("from diff_gaussian_rasterization import", source)
        self.assertIn("torch.zeros((pc.get_xyz.shape[0], 4)", source)
        self.assertIn("dc = pc.get_features_dc", source)
        self.assertIn("shs = pc.get_features_rest", source)
        self.assertIn("mult = mult", source)
        self.assertIn("metric_map = metric_map", source)
        self.assertIn("get_flag=get_flag", source)
        self.assertIn("scales=scales * scale_delta", source)
        self.assertIn("scales=scales * scale_delta[..., transform_index]", source)
        self.assertIn("metric_counts_accum + accum_metric_counts", source)

    def test_train_loop_routes_deblur_and_samples_score_views(self) -> None:
        source = (FASTGS_ROOT / "train.py").read_text(encoding="utf-8")

        self.assertIn("deblur_view_active = bool(deblur_loss_active and is_deblur_view", source)
        self.assertIn("render_fastgs_deblur", source)
        self.assertIn("compute_blur_indicator", source)
        self.assertIn("sample_sharp_score_cameras(scene, blur_registry, opt)", source)
        self.assertIn("cameras = scene.getTrainCameras().copy()", source)
        self.assertIn("compute_gaussian_score_fastgs(", source)
        self.assertIn("deblur_loss_active = schedule_deblur_loss_active", source)
        self.assertIn("vcd_score_renderer = (", source)
        self.assertIn('score_renderer=vcd_score_renderer', source)
        self.assertIn('score_purpose="vcd"', source)
        self.assertIn('deblur_state=deblur_state if vcd_score_renderer == "deblur" else None', source)
        self.assertIn('if vcd_score_renderer == "deblur":', source)
        self.assertIn('score_renderer="sharp"', source)
        self.assertIn('score_purpose="vcp"', source)
        self.assertIn("sharp_score_skipped_steps", source)
        self.assertIn("densify_deblur_extra_points", source)
        self.assertIn("fastgs_final_prune_min_opacity", source)
        self.assertIn("fastgs_final_prune_score_thresh", source)
        self.assertIn("score_thresh = opt.fastgs_final_prune_score_thresh", source)
        self.assertIn("blur_indicator = compute_blur_indicator", source)
        self.assertIn("fastgs_vcp_blur_protect_weight", source)
        self.assertIn("fastgs_late_prune_enabled", source)
        self.assertIn("fastgs_late_prune_min_opacity", source)
        self.assertIn("score_thresh = opt.fastgs_late_prune_score_thresh", source)
        self.assertIn("collect_final_metrics", source)
        self.assertIn('"final_psnr"', source)
        self.assertIn('"final_ssim"', source)
        self.assertIn('"final_lpips"', source)
        self.assertIn('"final_render_fps"', source)
        self.assertIn('"training_time_seconds"', source)
        self.assertIn('"final_model_dir_bytes"', source)
        self.assertIn("and not deblur_loss_active", source)
        self.assertIn('use_score = scene_profile.name != "indoor"', source)
        self.assertIn('use_scale = scene_profile.name != "indoor"', source)
        self.assertNotIn("and not deblur_state.enabled", source)
        self.assertNotIn('opt.deblur_blurred_views_only = "false"', source)
        self.assertIn('"deblur_final_prune_uses_sharp_score": True', source)

        schedule_source = (BACKEND_ROOT / "app" / "fine" / "deblur_schedule.py").read_text(encoding="utf-8")
        self.assertNotIn('if prune_mode == "conservative":\n            return raw_prune_mask', schedule_source)
        self.assertIn("max_prune_fraction_per_step=0.02", schedule_source)

    def test_fastgs_score_supports_deblur_renderer_for_vcd(self) -> None:
        source = (FASTGS_ROOT / "utils" / "fast_utils.py").read_text(encoding="utf-8")

        self.assertIn("from gaussian_renderer.deblur import render_fastgs_deblur", source)
        self.assertIn("def _render_score_image", source)
        self.assertIn('if score_renderer == "deblur"', source)
        self.assertIn("render_fastgs_deblur(", source)
        self.assertIn("score_renderer = \"sharp\"", source)
        self.assertIn("score_purpose = None", source)
        self.assertIn("deblur score renderer requires an enabled DeblurState", source)

    def test_fastgs_argparse_uses_central_defaults(self) -> None:
        sys.path.insert(0, str(BACKEND_ROOT))
        from app.fine.fastgs_defaults import (
            FASTGS_DATA_DEVICE,
            FASTGS_DENSIFICATION_INTERVAL,
            FASTGS_FINAL_PRUNE_MIN_OPACITY,
            FASTGS_ITERATIONS,
            FASTGS_LATE_PRUNE_INTERVAL,
            FASTGS_RESOLUTION,
            FASTGS_SAMPLE_CAMERAS,
        )

        spec = importlib.util.spec_from_file_location("fastgs_arguments_test", FASTGS_ROOT / "arguments" / "__init__.py")
        if spec is None or spec.loader is None:
            raise ImportError("failed to load FastGS arguments")
        arguments = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = arguments
        spec.loader.exec_module(arguments)

        parser = ArgumentParser()
        arguments.ModelParams(parser)
        arguments.OptimizationParams(parser)
        parsed = parser.parse_args([])

        self.assertEqual(parsed.iterations, FASTGS_ITERATIONS)
        self.assertEqual(parsed.resolution, FASTGS_RESOLUTION)
        self.assertEqual(parsed.data_device, FASTGS_DATA_DEVICE)
        self.assertEqual(parsed.densification_interval, FASTGS_DENSIFICATION_INTERVAL)
        self.assertEqual(parsed.fastgs_sample_cameras, FASTGS_SAMPLE_CAMERAS)
        self.assertEqual(parsed.fastgs_final_prune_min_opacity, FASTGS_FINAL_PRUNE_MIN_OPACITY)
        self.assertEqual(parsed.fastgs_late_prune_interval, FASTGS_LATE_PRUNE_INTERVAL)

    def test_gaussian_model_skips_gtnet_for_topology_mutations(self) -> None:
        source = (FASTGS_ROOT / "scene" / "gaussian_model.py").read_text(encoding="utf-8")

        self.assertIn('"name": "GTnet"', source)
        self.assertIn('if group.get("name") == "GTnet"', source)
        self.assertIn("create_deblur_net", source)

    def test_gaussian_model_uses_deblur_aware_vcd_and_vcp(self) -> None:
        source = (FASTGS_ROOT / "scene" / "gaussian_model.py").read_text(encoding="utf-8")

        self.assertIn("def _normalize_metric_signal", source)
        self.assertIn("importance_norm = self._normalize_metric_signal(importance_score", source)
        self.assertIn("grad_signal = torch.maximum", source)
        self.assertIn("score_blend = blend_alpha * importance_norm + (1.0 - blend_alpha) * grad_norm", source)
        self.assertIn("fastgs_vcd_blend_alpha", source)
        self.assertIn("fastgs_vcd_score_thresh", source)
        self.assertIn("blur_indicator", source)
        self.assertIn("blur_protect_weight", source)
        self.assertIn("score_values = score_values * (1.0 - protect_weight * blur_values)", source)


@unittest.skipIf(torch is None, f"torch import failed: {IMPORT_ERROR}")
class FastGSDeblurKernelTests(unittest.TestCase):
    def test_defocus_branch_shapes(self) -> None:
        model = GTnet(num_hidden=2, width=16, pos_delta=False, num_moments=2)
        pos = torch.zeros(5, 3)
        scales = torch.ones(5, 3)
        rotations = torch.ones(5, 4)
        viewdirs = torch.zeros(5, 3)

        scale_delta, rotation_delta, position_delta = model(pos, scales, rotations, viewdirs)

        self.assertEqual(tuple(scale_delta.shape), (5, 3))
        self.assertEqual(tuple(rotation_delta.shape), (5, 4))
        self.assertIsNone(position_delta)

    def test_motion_branch_clamps_position_moments(self) -> None:
        state = DeblurState(
            config=DeblurConfig(mode="motion", use_position=True, hidden=2, width=16, num_moments=3, max_position_delta=0.001),
            model=GTnet(num_hidden=2, width=16, pos_delta=True, num_moments=3),
        )
        means = torch.ones(4, 3) * 100.0
        scales = torch.ones(4, 3)
        rotations = torch.ones(4, 4)

        scale_delta, rotation_delta, position_delta = predict_deblur_transforms(
            state,
            means,
            scales,
            rotations,
            torch.zeros(3),
        )

        self.assertEqual(tuple(scale_delta.shape), (4, 12))
        self.assertEqual(tuple(rotation_delta.shape), (4, 16))
        self.assertEqual(tuple(position_delta.shape), (4, 9))
        self.assertTrue(bool(torch.all(torch.abs(position_delta) <= 0.001 + 1e-6)))

    def test_regularization_is_nonnegative(self) -> None:
        reg = deblur_transform_regularization(torch.ones(3, 3), torch.ones(3, 4) * 1.01, torch.zeros(3, 6))

        self.assertGreaterEqual(float(reg.item()), 0.0)

    def test_blur_indicator_shape_and_clamp(self) -> None:
        state = DeblurState(
            config=DeblurConfig(mode="defocus", use_position=False, hidden=2, width=16, num_moments=2),
            model=GTnet(num_hidden=2, width=16, pos_delta=False, num_moments=2),
        )
        means = torch.zeros(6, 3)
        scales = torch.ones(6, 3)
        rotations = torch.ones(6, 4)

        indicator = compute_blur_indicator(
            state,
            means,
            scales,
            rotations,
            [torch.zeros(3), torch.ones(3)],
        )

        self.assertEqual(tuple(indicator.shape), (6,))
        self.assertTrue(bool(torch.all(indicator >= 0.0)))
        self.assertTrue(bool(torch.all(indicator <= 1.0)))


if __name__ == "__main__":
    unittest.main()
