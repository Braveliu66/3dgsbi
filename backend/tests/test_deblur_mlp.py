from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch

    from app.fine.deblur_mlp import (
        DeblurMLPConfig,
        DeblurMLPState,
        GTnet,
        attach_deblur_mlp_optimizer,
        build_deblur_mlp_state,
        predict_deblur_transforms,
    )
except Exception as exc:  # pragma: no cover - depends on local torch availability
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(torch is None, f"torch import failed: {IMPORT_ERROR}")
class DeblurMLPTests(unittest.TestCase):
    def test_gtnet_defocus_branch_shapes(self) -> None:
        model = GTnet(num_hidden=2, width=16, pos_delta=False, num_moments=2)
        pos = torch.zeros(5, 3)
        scales = torch.ones(5, 3)
        rotations = torch.ones(5, 4)
        viewdirs = torch.zeros(5, 3)

        scale_delta, rotation_delta, position_delta = model(pos, scales, rotations, viewdirs)

        self.assertEqual(tuple(scale_delta.shape), (5, 3))
        self.assertEqual(tuple(rotation_delta.shape), (5, 4))
        self.assertIsNone(position_delta)

    def test_motion_branch_outputs_position_moments(self) -> None:
        state = build_deblur_mlp_state(
            "motion",
            {"fine_deblur_width": 16, "fine_deblur_hidden": 2, "fine_deblur_num_moments": 3},
            device="cpu",
        )
        means = torch.zeros(4, 3)
        scales = torch.ones(4, 3)
        rotations = torch.ones(4, 4)
        camera_center = torch.zeros(1, 3)

        scale_delta, rotation_delta, position_delta = predict_deblur_transforms(state, means, scales, rotations, camera_center)

        self.assertEqual(tuple(scale_delta.shape), (4, 12))
        self.assertEqual(tuple(rotation_delta.shape), (4, 16))
        self.assertEqual(tuple(position_delta.shape), (4, 9))
        self.assertTrue(bool(torch.all(scale_delta >= 0.9)))
        self.assertTrue(bool(torch.all(rotation_delta >= 0.9)))

    def test_defocus_state_disables_position_moments(self) -> None:
        state = build_deblur_mlp_state("defocus", {"fine_deblur_width": 16, "fine_deblur_hidden": 2}, device="cpu")

        self.assertTrue(state.enabled)
        self.assertFalse(state.config.use_position)
        self.assertEqual(state.metrics()["deblur_algorithm"], "Deblurring-3DGS_GTnet")
        self.assertEqual(state.metrics()["deblur_mlp_min_clamp"], 0.9)

    def test_gtnet_optimizer_group_survives_topology_wrappers(self) -> None:
        class DummyGaussians:
            def __init__(self) -> None:
                self.xyz = torch.nn.Parameter(torch.ones(3, 1))
                self.optimizer = torch.optim.Adam([{"params": [self.xyz], "lr": 0.0, "name": "xyz"}], lr=0.0)
                self.prune_seen: list[str] = []
                self.cat_seen: list[str] = []

            def _prune_optimizer(self, mask):
                self.prune_seen = [group["name"] for group in self.optimizer.param_groups]
                return {"xyz": self.xyz}

            def cat_tensors_to_optimizer(self, tensors_dict):
                self.cat_seen = [group["name"] for group in self.optimizer.param_groups]
                return {"xyz": self.xyz}

        dummy = DummyGaussians()
        state = DeblurMLPState(
            config=DeblurMLPConfig(mode="motion", use_position=True, width=16, hidden=2, num_moments=2),
            model=GTnet(num_hidden=2, width=16, pos_delta=True, num_moments=2),
        )

        attach_deblur_mlp_optimizer(dummy, state)
        dummy._prune_optimizer(torch.tensor([True, False, True]))
        dummy.cat_tensors_to_optimizer({"xyz": torch.ones(1, 1)})

        self.assertIn("GTnet", [group["name"] for group in dummy.optimizer.param_groups])
        self.assertNotIn("GTnet", dummy.prune_seen)
        self.assertNotIn("GTnet", dummy.cat_seen)


if __name__ == "__main__":
    unittest.main()
