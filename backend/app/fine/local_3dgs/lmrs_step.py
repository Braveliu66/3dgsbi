from __future__ import annotations

from typing import Any

import torch

from app.fine.local_3dgs.cg_optimizer import LocalCGOptimizer


class UniformPixelSampler:
    def sample(self, current_batch: int, cg_state: Any) -> None:
        sample_size = int(cg_state.sample_per_block)
        total_tiles = int(cg_state.total_blocks_sampled)
        max_pixels = int(cg_state.tile_block_dim[0] * cg_state.tile_block_dim[1])
        sampled = torch.randint(low=0, high=max_pixels, size=(total_tiles, sample_size), dtype=torch.int32, device="cuda")
        cg_state.state["sampled_pixels"][current_batch] = sampled
        cg_state.state["likelihoods"][current_batch].fill_(1.0 / float(max_pixels))


class RandomCameraSampler:
    def __init__(self, cameras: list[Any]) -> None:
        self.cameras = cameras
        self.stack: list[Any] = []

    def get_camera(self, current_batch: int) -> Any:
        import random

        if not self.stack:
            self.stack = self.cameras.copy()
        return self.stack.pop(random.randrange(len(self.stack)))


def post_render_task(pixel_sampler: UniformPixelSampler, cg_state: Any, current_batch: int) -> None:
    if int(cg_state.sample_per_block) >= 256:
        cg_state.state["sampled_pixels"][current_batch] = cg_state.full_image
        cg_state.state["likelihoods"][current_batch].fill_(1.0 / 256.0)
        return
    pixel_sampler.sample(current_batch, cg_state)


def lmrs_step(
    *,
    pixel_sampler: UniformPixelSampler,
    camera_sampler: RandomCameraSampler,
    gaussians: Any,
    pipe: Any,
    background: torch.Tensor,
    optimizer: LocalCGOptimizer,
    render_fn,
) -> tuple[float, float]:
    import time

    started = time.monotonic()
    for batch_index in range(int(gaussians.cgState.batch_size)):
        viewpoint = camera_sampler.get_camera(batch_index)
        gt_image = viewpoint.original_image.to("cuda")
        pkg = render_fn(viewpoint, gaussians, pipe, background, gaussians.cgState, batch_index)
        gaussians.batchState.insert(batch_index, pkg["radii"])
        optimizer.append_residual(pkg["render"], gt_image, batch_index)
        post_render_task(pixel_sampler, gaussians.cgState, batch_index)
    loss = optimizer.linear_solve(gaussians, return_matvec_kernels=bool(getattr(pipe, "return_matvec_kernels", False)))
    optimizer.step(gaussians)
    torch.cuda.synchronize()
    return float(loss.item()), time.monotonic() - started

