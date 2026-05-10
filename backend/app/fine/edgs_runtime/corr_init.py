from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(slots=True)
class CorrInitResult:
    gaussian_count: int
    reference_count: int
    pair_count: int
    matched_points: int

    def metrics(self) -> dict[str, int]:
        return {
            "edgs_initial_gaussians": self.gaussian_count,
            "edgs_reference_count": self.reference_count,
            "edgs_pair_count": self.pair_count,
            "edgs_matched_points": self.matched_points,
        }


def instantiate_roma(device: str, model_name: str = "outdoor") -> Any:
    try:
        import torch
        from romatch import roma_indoor, roma_outdoor
    except Exception as exc:
        raise RuntimeError(f"RoMA runtime is unavailable: {exc}") from exc

    torch.set_float32_matmul_precision("highest")
    normalized = str(model_name or "outdoor").lower()
    weights, dinov2_weights = _load_cached_roma_weights(torch, normalized, device)
    if normalized == "indoor":
        return roma_indoor(device=device, weights=weights, dinov2_weights=dinov2_weights)
    return roma_outdoor(device=device, weights=weights, dinov2_weights=dinov2_weights)


def _load_cached_roma_weights(torch: Any, model_name: str, device: str) -> tuple[Any | None, Any | None]:
    cache_dir = Path(os.getenv("MODEL_CACHE_DIR", "/model-cache")) / "roma"
    model_path = cache_dir / f"roma_{'indoor' if model_name == 'indoor' else 'outdoor'}.pth"
    dinov2_path = cache_dir / "dinov2_vitl14_pretrain.pth"
    weights = torch.load(model_path, map_location="cpu") if model_path.exists() else None
    dinov2_weights = torch.load(dinov2_path, map_location="cpu") if dinov2_path.exists() else None
    return weights, dinov2_weights


def init_gaussians_with_corr(
    *,
    gaussians: Any,
    scene: Any,
    cfg: Any,
    device: str = "cuda",
    roma_model: Any | None = None,
) -> CorrInitResult:
    import torch
    from utils.graphics_utils import BasicPointCloud

    cameras = list(scene.getTrainCameras())
    if len(cameras) < 2:
        raise RuntimeError("EDGS dense initialization needs at least two train cameras")

    matches_per_ref = int(getattr(cfg, "matches_per_ref", 15_000))
    nns_per_ref = max(1, int(getattr(cfg, "nns_per_ref", 3)))
    num_refs = min(len(cameras), int(getattr(cfg, "num_refs", len(cameras))))
    max_points = int(getattr(cfg, "max_points", 500_000))
    reproj_error = float(getattr(cfg, "reprojection_error", 4.0))
    roma_name = str(getattr(cfg, "roma_model", "outdoor"))

    roma = roma_model if roma_model is not None else instantiate_roma(device, roma_name)
    centers = np.stack([_camera_center(camera) for camera in cameras], axis=0)
    reference_indices = _farthest_indices(centers, num_refs)
    neighbors = _neighbor_indices(centers, nns_per_ref)
    per_pair_matches = max(512, int(math.ceil(matches_per_ref / float(nns_per_ref))))

    xyz_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    pair_count = 0

    with tempfile.TemporaryDirectory(prefix="edgs_roma_") as tmp:
        image_paths = _write_camera_images(cameras, Path(tmp))
        for ref_idx in reference_indices:
            ref_camera = cameras[ref_idx]
            for nn_idx in neighbors[ref_idx]:
                if nn_idx == ref_idx:
                    continue
                pair_count += 1
                try:
                    kpts_ref, kpts_nn, certainty = _match_pair(
                        roma,
                        image_paths[ref_idx],
                        image_paths[nn_idx],
                        ref_camera,
                        cameras[nn_idx],
                        per_pair_matches,
                        device,
                    )
                except Exception:
                    continue
                if kpts_ref.shape[0] < 8:
                    continue

                points, valid = _triangulate_matches(ref_camera, cameras[nn_idx], kpts_ref, kpts_nn, reproj_error)
                if not bool(valid.any().item()):
                    continue
                points_np = points[valid].detach().cpu().numpy().astype(np.float32)
                colors_np = _sample_camera_colors(ref_camera, kpts_ref[valid]).astype(np.float32)
                scores_np = certainty[valid].detach().float().cpu().numpy().reshape(-1)
                xyz_chunks.append(points_np)
                rgb_chunks.append(colors_np)
                score_chunks.append(scores_np)

    if not xyz_chunks:
        raise RuntimeError("EDGS/RoMA did not produce valid triangulated correspondences")

    xyz = np.concatenate(xyz_chunks, axis=0)
    rgb = np.concatenate(rgb_chunks, axis=0)
    scores = np.concatenate(score_chunks, axis=0)
    if xyz.shape[0] > max_points:
        keep = np.argsort(scores)[-max_points:]
        xyz = xyz[keep]
        rgb = rgb[keep]

    normals = np.zeros_like(xyz, dtype=np.float32)
    pcd = BasicPointCloud(points=xyz, colors=np.clip(rgb, 0.0, 1.0), normals=normals)
    gaussians.create_from_pcd(pcd, cameras, scene.cameras_extent)
    return CorrInitResult(
        gaussian_count=int(xyz.shape[0]),
        reference_count=len(reference_indices),
        pair_count=pair_count,
        matched_points=int(sum(chunk.shape[0] for chunk in xyz_chunks)),
    )


def _match_pair(
    roma: Any,
    ref_path: Path,
    nn_path: Path,
    ref_camera: Any,
    nn_camera: Any,
    num_matches: int,
    device: str,
) -> tuple[Any, Any, Any]:
    import torch

    warp, certainty = roma.match(str(ref_path), str(nn_path), device=device)
    try:
        matches, certainty = roma.sample(warp, certainty, num=num_matches)
    except TypeError:
        matches, certainty = roma.sample(warp, certainty)
        if certainty.numel() > num_matches:
            keep = torch.argsort(certainty.reshape(-1), descending=True)[:num_matches]
            matches = matches[keep]
            certainty = certainty.reshape(-1)[keep]
    kpts_ref, kpts_nn = roma.to_pixel_coordinates(
        matches,
        int(ref_camera.image_height),
        int(ref_camera.image_width),
        int(nn_camera.image_height),
        int(nn_camera.image_width),
    )
    return kpts_ref.float(), kpts_nn.float(), certainty.reshape(-1).float()


def _triangulate_matches(camera_a: Any, camera_b: Any, kpts_a: Any, kpts_b: Any, max_reproj_error: float) -> tuple[Any, Any]:
    import torch

    device = kpts_a.device
    dtype = torch.float32
    p_a = _projection_matrix(camera_a, device=device, dtype=dtype)
    p_b = _projection_matrix(camera_b, device=device, dtype=dtype)
    u_a, v_a = kpts_a[:, 0], kpts_a[:, 1]
    u_b, v_b = kpts_b[:, 0], kpts_b[:, 1]
    rows = torch.stack(
        [
            u_a[:, None] * p_a[2] - p_a[0],
            v_a[:, None] * p_a[2] - p_a[1],
            u_b[:, None] * p_b[2] - p_b[0],
            v_b[:, None] * p_b[2] - p_b[1],
        ],
        dim=1,
    )
    _, _, vh = torch.linalg.svd(rows)
    homogeneous = vh[:, -1, :]
    xyz = homogeneous[:, :3] / _safe_denominator(homogeneous[:, 3:])

    depth_a = _camera_depth(camera_a, xyz)
    depth_b = _camera_depth(camera_b, xyz)
    err_a = _reprojection_error(p_a, xyz, kpts_a)
    err_b = _reprojection_error(p_b, xyz, kpts_b)
    valid = (
        torch.isfinite(xyz).all(dim=1)
        & torch.isfinite(err_a)
        & torch.isfinite(err_b)
        & (depth_a > 0.01)
        & (depth_b > 0.01)
        & (err_a <= max_reproj_error)
        & (err_b <= max_reproj_error)
    )
    return xyz, valid


def _projection_matrix(camera: Any, *, device: Any, dtype: Any) -> Any:
    import torch
    from utils.graphics_utils import fov2focal

    fx = float(fov2focal(camera.FoVx, int(camera.image_width)))
    fy = float(fov2focal(camera.FoVy, int(camera.image_height)))
    cx = float(camera.image_width) * 0.5
    cy = float(camera.image_height) * 0.5
    k = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
    r = torch.as_tensor(np.asarray(camera.R.T, dtype=np.float32), device=device, dtype=dtype)
    t = torch.as_tensor(np.asarray(camera.T, dtype=np.float32), device=device, dtype=dtype).reshape(3, 1)
    return k @ torch.cat([r, t], dim=1)


def _camera_depth(camera: Any, xyz: Any) -> Any:
    import torch

    r = torch.as_tensor(np.asarray(camera.R.T, dtype=np.float32), device=xyz.device, dtype=xyz.dtype)
    t = torch.as_tensor(np.asarray(camera.T, dtype=np.float32), device=xyz.device, dtype=xyz.dtype)
    cam = xyz @ r.T + t
    return cam[:, 2]


def _reprojection_error(proj: Any, xyz: Any, pixels: Any) -> Any:
    import torch

    ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    projected = torch.cat([xyz, ones], dim=1) @ proj.T
    uv = projected[:, :2] / _safe_denominator(projected[:, 2:])
    return torch.linalg.norm(uv - pixels.to(device=xyz.device, dtype=xyz.dtype), dim=1)


def _safe_denominator(values: Any) -> Any:
    import torch

    eps = torch.full_like(values, 1e-8)
    return torch.where(values.abs() < 1e-8, torch.where(values < 0, -eps, eps), values)


def _camera_center(camera: Any) -> np.ndarray:
    value = camera.camera_center
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(3)


def _neighbor_indices(centers: np.ndarray, count: int) -> list[list[int]]:
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    return [list(np.argsort(row)[1 : count + 1]) for row in distances]


def _farthest_indices(centers: np.ndarray, count: int) -> list[int]:
    if count >= centers.shape[0]:
        return list(range(centers.shape[0]))
    chosen = [0]
    distances = np.linalg.norm(centers - centers[0], axis=1)
    while len(chosen) < count:
        next_idx = int(np.argmax(distances))
        chosen.append(next_idx)
        distances = np.minimum(distances, np.linalg.norm(centers - centers[next_idx], axis=1))
    return sorted(set(chosen))


def _write_camera_images(cameras: list[Any], root: Path) -> list[Path]:
    paths = []
    for index, camera in enumerate(cameras):
        path = root / f"{index:06d}.jpg"
        _camera_to_pil(camera).save(path, format="JPEG", quality=94)
        paths.append(path)
    return paths


def _camera_to_pil(camera: Any) -> Image.Image:
    image = camera.original_image.detach().float().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _sample_camera_colors(camera: Any, pixels: Any) -> np.ndarray:
    image = camera.original_image.detach().float().cpu().permute(1, 2, 0).numpy()
    coords = pixels.detach().float().cpu().numpy()
    x = np.clip(np.rint(coords[:, 0]).astype(np.int64), 0, image.shape[1] - 1)
    y = np.clip(np.rint(coords[:, 1]).astype(np.int64), 0, image.shape[0] - 1)
    return image[y, x, :]
