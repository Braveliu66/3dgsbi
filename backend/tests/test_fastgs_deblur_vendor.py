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

    def test_train_loop_allows_topology_updates_with_deblur_and_prunes_with_sharp_score(self) -> None:
        source = (FASTGS_ROOT / "train.py").read_text(encoding="utf-8")

        self.assertIn("can_update_topology = bool(iteration < opt.densify_until_iter)", source)
        self.assertNotIn("can_update_topology = bool(iteration < opt.densify_until_iter and not deblur_active)", source)
        self.assertIn("render_fastgs_deblur", source)
        self.assertIn("compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt", source)
        self.assertIn("fastgs_final_prune_min_opacity", source)
        self.assertIn("fastgs_final_prune_score_thresh", source)
        self.assertIn("score_thresh = opt.fastgs_final_prune_score_thresh", source)
        self.assertIn("fastgs_late_prune_enabled", source)
        self.assertIn("fastgs_late_prune_min_opacity", source)
        self.assertIn("score_thresh = opt.fastgs_late_prune_score_thresh", source)
        self.assertNotIn("and not deblur_state.enabled", source)
        self.assertIn('"deblur_final_prune_uses_sharp_score": True', source)

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


if __name__ == "__main__":
    unittest.main()
