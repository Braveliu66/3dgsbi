from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


TILE_SIZE = 256


@dataclass(slots=True)
class BatchState:
    radii_batch: torch.Tensor

    @classmethod
    def create(cls, batch_size: int, gaussian_count: int, *, device: torch.device | str = "cuda") -> "BatchState":
        return cls(torch.zeros((batch_size, gaussian_count), dtype=torch.float32, device=device))

    def insert(self, index: int, radii: torch.Tensor) -> None:
        if self.radii_batch.shape[1] != radii.shape[0]:
            self.radii_batch = torch.zeros((self.radii_batch.shape[0], radii.shape[0]), dtype=torch.float32, device=radii.device)
        self.radii_batch[index] = radii

    def resize(self, gaussian_count: int) -> None:
        self.radii_batch = torch.zeros((self.radii_batch.shape[0], gaussian_count), dtype=torch.float32, device=self.radii_batch.device)


class CGSolverState:
    def __init__(self, gaussian_count: int, options: Any, *, device: torch.device | str = "cuda") -> None:
        self.state: dict[str, torch.Tensor] = {}
        self.num_gaussians = int(gaussian_count)
        self.numberOfParams = self.num_gaussians * 14
        self.kernel = int(getattr(options, "kernel", 1))
        self.batch_size = int(getattr(options, "batch_size", 1))
        self.cg_iter = int(getattr(options, "cg_iter", 8))
        self.loss_fn = str(getattr(options, "loss_fn", "mse"))
        self.sampling_distribution = str(getattr(options, "sampling_distribution", "mobile_uniform"))
        self.sample_per_block = int(getattr(options, "N_sample_per_tile", 32))
        self.tile_block_dim = int(getattr(options, "tile_block_dimx", 16)), int(getattr(options, "tile_block_dimy", 16))
        self.lambda_dssim = float(getattr(options, "ssim_weight", 0.0))
        self.temperature = float(getattr(options, "temperature", 1.0))
        self.device = device
        self.state["d_colors"] = self._zeros((self.num_gaussians, 3))
        self.state["d_mean2D"] = self._zeros((self.num_gaussians, 3))
        self.state["d_cov2D"] = self._zeros((self.num_gaussians, 4))

    def _zeros(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def set_scene_size(self, scene: Any) -> None:
        camera = scene.train_cameras[1.0][0]
        self.width = int(camera.image_width)
        self.height = int(camera.image_height)
        self.width_blocks = (self.width + self.tile_block_dim[0] - 1) // self.tile_block_dim[0]
        self.height_blocks = (self.height + self.tile_block_dim[1] - 1) // self.tile_block_dim[1]
        self.total_blocks_sampled = self.width_blocks * self.height_blocks
        self.state["sampled_pixels"] = self._zeros((self.batch_size, self.total_blocks_sampled, self.sample_per_block), torch.int32)
        self.state["likelihoods"] = self._zeros((self.batch_size, self.total_blocks_sampled, TILE_SIZE))
        self.state["n_of_gaussians_per_pixel"] = self._zeros((self.batch_size, self.total_blocks_sampled, TILE_SIZE), torch.int32)
        self.full_image = torch.arange(TILE_SIZE, dtype=torch.int32, device=self.device).repeat(self.total_blocks_sampled, 1)
        self.total_pixels = self.width * self.height * 3 * self.batch_size

    def resize_cg_vectors(self, gaussian_count: int) -> None:
        self.num_gaussians = int(gaussian_count)
        self.numberOfParams = self.num_gaussians * 14
        self.state["d_colors"] = self._zeros((self.num_gaussians, 3))
        self.state["d_mean2D"] = self._zeros((self.num_gaussians, 3))
        self.state["d_cov2D"] = self._zeros((self.num_gaussians, 4))

