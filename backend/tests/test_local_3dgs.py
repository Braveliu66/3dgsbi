from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    from plyfile import PlyData, PlyElement

    from app.fine.local_3dgs.sparse_compensation import compensate_sparse_point_cloud
except Exception as exc:  # pragma: no cover
    np = None
    PlyData = None
    PlyElement = None
    SPARSE_IMPORT_ERROR = exc
else:
    SPARSE_IMPORT_ERROR = None

try:
    import torch

    from app.fine.fastgs_policy import FastGSPolicy, visible_mean_loss_score
except Exception as exc:  # pragma: no cover
    torch = None
    FastGSPolicy = None
    visible_mean_loss_score = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class Local3DGSTests(unittest.TestCase):
    @unittest.skipIf(np is None, f"sparse compensation dependencies unavailable: {SPARSE_IMPORT_ERROR}")
    def test_sparse_compensation_adds_points_and_removes_outlier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sparse = Path(tmp) / "sparse" / "0"
            sparse.mkdir(parents=True)
            ply_path = sparse / "points3D.ply"
            xyz = np.array([[i, 0.0, 0.0] for i in range(20)] + [[1000.0, 1000.0, 1000.0]], dtype=np.float32)
            rgb = np.tile(np.array([[120, 80, 40]], dtype=np.uint8), (xyz.shape[0], 1))
            write_test_ply(ply_path, xyz, rgb)

            result = compensate_sparse_point_cloud(
                Path(tmp),
                {"fine_sparse_compensation_ratio": 0.5, "fine_sparse_compensation_max_points": 10},
            )

            vertices = PlyData.read(ply_path)["vertex"]
            self.assertGreater(result.metrics["sparse_compensation_points"], 0)
            self.assertGreaterEqual(result.metrics["sparse_compensation_removed_outliers"], 1)
            self.assertGreater(len(vertices), 20)

    @unittest.skipIf(torch is None, f"torch import failed: {IMPORT_ERROR}")
    def test_fastgs_visibility_score_assigns_visible_gaussians(self) -> None:
        visibility = torch.tensor([True, False, True, False])

        score = visible_mean_loss_score(visibility, 4, 0.25)

        self.assertEqual(tuple(score.shape), (4, 1))
        self.assertEqual(float(score[0].item()), 0.25)
        self.assertEqual(float(score[1].item()), 0.0)
        self.assertEqual(FastGSPolicy({"fine_fastgs_sample_cameras": 3}).metrics()["fastgs_sample_cameras"], 3)

    @unittest.skipIf(torch is None, f"torch import failed: {IMPORT_ERROR}")
    def test_fastgs_final_prune_removes_low_opacity_and_high_score(self) -> None:
        class DummyGaussians:
            def __init__(self) -> None:
                self.xyz = torch.zeros((4, 3))
                self.opacity = torch.tensor([[0.05], [0.2], [0.2], [0.2]])

            @property
            def get_xyz(self):
                return self.xyz

            def final_prune_fastgs(self, min_opacity, pruning_score=None, score_thresh=0.97):
                score = torch.zeros((self.xyz.shape[0],)) if pruning_score is None else pruning_score
                mask = torch.logical_or(self.opacity.squeeze(-1) < min_opacity, score > score_thresh)
                self.prune_points(mask)

            def prune_points(self, mask):
                keep = ~mask
                self.xyz = self.xyz[keep]
                self.opacity = self.opacity[keep]

        policy = FastGSPolicy({"fine_fastgs_final_prune_score_thresh": 0.9})
        policy.pruning_score = torch.tensor([[0.0], [0.95], [0.0], [0.0]])
        gaussians = DummyGaussians()

        policy.apply_final_prune(gaussians, min_opacity=0.1)

        self.assertEqual(tuple(gaussians.get_xyz.shape), (2, 3))
        self.assertEqual(policy.metrics()["fastgs_final_pruned_points"], 2)


def write_test_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    normals = np.zeros_like(xyz, dtype=np.float32)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    records = np.empty(xyz.shape[0], dtype=dtype)
    records["x"], records["y"], records["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    records["nx"], records["ny"], records["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    records["red"], records["green"], records["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(records, "vertex")]).write(path)


if __name__ == "__main__":
    unittest.main()
