from __future__ import annotations

import sys
import types
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TRAINER_ROOT = WORKSPACE_ROOT / "worker" / "trainer" / "dash_deblur_group_gs"


class GsplatBackendTests(unittest.TestCase):
    def test_backend_imports_gsplat_lazily(self) -> None:
        source = (TRAINER_ROOT / "gaussian_renderer" / "backends" / "gsplat_backend.py").read_text(encoding="utf-8")

        self.assertNotIn("from gsplat", source.split("def gsplat_rasterize", 1)[0])
        self.assertIn("from gsplat import rasterization", source.split("def gsplat_rasterize", 1)[1])

    def test_gsplat_rasterize_returns_image_and_radii_shapes(self) -> None:
        try:
            import torch
        except Exception as exc:
            raise unittest.SkipTest(f"torch unavailable: {exc}") from exc

        sys.path.insert(0, str(TRAINER_ROOT))
        spec = importlib.util.spec_from_file_location(
            "gsplat_backend_under_test",
            TRAINER_ROOT / "gaussian_renderer" / "backends" / "gsplat_backend.py",
        )
        assert spec is not None
        assert spec.loader is not None
        gsplat_backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gsplat_backend)

        def rasterization(**kwargs):
            means = kwargs["means"]
            colors = kwargs["colors"]
            height = kwargs["height"]
            width = kwargs["width"]
            assert tuple(colors.shape) == (3, 1, 3)
            rendered = torch.ones((1, height, width, 3), dtype=means.dtype, device=means.device)
            alpha = torch.ones((1, height, width, 1), dtype=means.dtype, device=means.device)
            meta = {
                "radii": torch.arange(means.shape[0], dtype=means.dtype, device=means.device),
                "means2d": torch.zeros((1, means.shape[0], 2), dtype=means.dtype, device=means.device),
            }
            return rendered, alpha, meta

        gsplat_module = types.ModuleType("gsplat")
        gsplat_module.rasterization = rasterization
        camera = SimpleNamespace(
            world_view_transform=torch.eye(4),
            image_width=5,
            image_height=4,
            FoVx=0.8,
            FoVy=0.6,
        )
        pc = SimpleNamespace(
            get_xyz=torch.zeros((3, 3), dtype=torch.float32),
            get_rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3),
            get_scaling=torch.ones((3, 3), dtype=torch.float32),
            get_opacity=torch.ones((3, 1), dtype=torch.float32),
            get_features=torch.ones((3, 1, 3), dtype=torch.float32),
            active_sh_degree=0,
        )

        with patch.dict(sys.modules, {"gsplat": gsplat_module}):
            image, radii, means2d = gsplat_backend.gsplat_rasterize(camera, pc, torch.zeros(3))

        self.assertEqual(tuple(image.shape), (3, 4, 5))
        self.assertEqual(tuple(radii.shape), (3,))
        self.assertEqual(tuple(means2d.shape), (3, 3))

    def test_gsplat_rasterize_reduces_vector_radii_to_one_value_per_point(self) -> None:
        try:
            import torch
        except Exception as exc:
            raise unittest.SkipTest(f"torch unavailable: {exc}") from exc

        sys.path.insert(0, str(TRAINER_ROOT))
        spec = importlib.util.spec_from_file_location(
            "gsplat_backend_under_test_vector_radii",
            TRAINER_ROOT / "gaussian_renderer" / "backends" / "gsplat_backend.py",
        )
        assert spec is not None
        assert spec.loader is not None
        gsplat_backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gsplat_backend)

        def rasterization(**kwargs):
            means = kwargs["means"]
            height = kwargs["height"]
            width = kwargs["width"]
            rendered = torch.ones((1, height, width, 3), dtype=means.dtype, device=means.device)
            alpha = torch.ones((1, height, width, 1), dtype=means.dtype, device=means.device)
            meta = {
                "radii": torch.tensor([[[0.0, 2.0], [3.0, 1.0], [0.0, 0.0]]], dtype=means.dtype, device=means.device),
                "means2d": torch.zeros((1, means.shape[0], 2), dtype=means.dtype, device=means.device),
            }
            return rendered, alpha, meta

        gsplat_module = types.ModuleType("gsplat")
        gsplat_module.rasterization = rasterization
        camera = SimpleNamespace(
            world_view_transform=torch.eye(4),
            image_width=5,
            image_height=4,
            FoVx=0.8,
            FoVy=0.6,
        )
        pc = SimpleNamespace(
            get_xyz=torch.zeros((3, 3), dtype=torch.float32),
            get_rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3),
            get_scaling=torch.ones((3, 3), dtype=torch.float32),
            get_opacity=torch.ones((3, 1), dtype=torch.float32),
            get_features=torch.ones((3, 1, 3), dtype=torch.float32),
            active_sh_degree=0,
        )

        with patch.dict(sys.modules, {"gsplat": gsplat_module}):
            _image, radii, means2d = gsplat_backend.gsplat_rasterize(camera, pc, torch.zeros(3))

        self.assertEqual(radii.tolist(), [2.0, 3.0, 0.0])
        self.assertEqual(tuple(means2d.shape), (3, 3))


if __name__ == "__main__":
    unittest.main()
