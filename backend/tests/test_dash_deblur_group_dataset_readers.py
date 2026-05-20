from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATASET_READERS_SOURCE = WORKSPACE_ROOT / "worker" / "trainer" / "dash_deblur_group_gs" / "scene" / "dataset_readers.py"


class DashDeblurGroupDatasetReaderTests(unittest.TestCase):
    def test_read_colmap_cameras_accepts_sparse_image_ids(self) -> None:
        dataset_readers = load_dataset_readers_with_stubs()

        cam_extrinsics = {
            5: SimpleNamespace(
                camera_id=11,
                name="000010.jpg",
                qvec=[1.0, 0.0, 0.0, 0.0],
                tvec=[0.0, 0.0, 0.0],
            )
        }
        cam_intrinsics = {
            11: SimpleNamespace(
                id=11,
                model="PINHOLE",
                width=800,
                height=600,
                params=[100.0, 120.0],
            )
        }

        cameras = dataset_readers.readColmapCameras(cam_extrinsics, cam_intrinsics, "scene/images")

        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0].uid, 11)
        self.assertEqual(cameras[0].image_name, "000010")

    def test_read_colmap_scene_info_selects_named_pointcloud(self) -> None:
        dataset_readers = load_dataset_readers_with_stubs()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sparse_dir = root / "sparse" / "0"
            sparse_dir.mkdir(parents=True)
            (sparse_dir / "points3D_eap.ply").write_text("ply\n", encoding="ascii")

            dataset_readers.read_extrinsics_binary = lambda path: {}
            dataset_readers.read_intrinsics_binary = lambda path: {}
            dataset_readers.readColmapCameras = lambda cam_extrinsics, cam_intrinsics, images_folder: [SimpleNamespace(image_name="000010")]
            dataset_readers.getNerfppNorm = lambda cameras: {"radius": 1.0}
            dataset_readers.fetchPly = lambda path: SimpleNamespace(path=path)

            scene_info = dataset_readers.readColmapSceneInfo(str(root), None, False, pc_name="../points3D_eap")

        self.assertTrue(scene_info.ply_path.replace("\\", "/").endswith("sparse/0/points3D_eap.ply"))
        self.assertEqual(scene_info.point_cloud.path, scene_info.ply_path)


def load_dataset_readers_with_stubs():
    pil_module = types.ModuleType("PIL")
    image_module = types.ModuleType("PIL.Image")
    image_module.open = lambda path: SimpleNamespace(path=path)
    pil_module.Image = image_module

    numpy_module = types.ModuleType("numpy")
    numpy_module.array = lambda value, *args, **kwargs: value
    numpy_module.transpose = lambda value, *args, **kwargs: value

    colmap_loader_module = types.ModuleType("scene.colmap_loader")
    colmap_loader_module.qvec2rotmat = lambda qvec: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    colmap_loader_module.read_extrinsics_text = stub_not_used
    colmap_loader_module.read_intrinsics_text = stub_not_used
    colmap_loader_module.read_extrinsics_binary = stub_not_used
    colmap_loader_module.read_intrinsics_binary = stub_not_used
    colmap_loader_module.read_points3D_binary = stub_not_used
    colmap_loader_module.read_points3D_text = stub_not_used

    graphics_utils_module = types.ModuleType("utils.graphics_utils")
    graphics_utils_module.getWorld2View2 = stub_not_used
    graphics_utils_module.focal2fov = lambda focal, pixels: focal / pixels
    graphics_utils_module.fov2focal = stub_not_used

    plyfile_module = types.ModuleType("plyfile")
    plyfile_module.PlyData = SimpleNamespace(read=stub_not_used)
    plyfile_module.PlyElement = SimpleNamespace(describe=stub_not_used)

    sh_utils_module = types.ModuleType("utils.sh_utils")
    sh_utils_module.SH2RGB = stub_not_used

    gaussian_model_module = types.ModuleType("scene.gaussian_model")
    gaussian_model_module.BasicPointCloud = SimpleNamespace

    scene_module = types.ModuleType("scene")
    scene_module.__path__ = []
    utils_module = types.ModuleType("utils")
    utils_module.__path__ = []

    stubs = {
        "PIL": pil_module,
        "PIL.Image": image_module,
        "numpy": numpy_module,
        "plyfile": plyfile_module,
        "scene": scene_module,
        "scene.colmap_loader": colmap_loader_module,
        "scene.gaussian_model": gaussian_model_module,
        "utils": utils_module,
        "utils.graphics_utils": graphics_utils_module,
        "utils.sh_utils": sh_utils_module,
    }

    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("dataset_readers_under_test", DATASET_READERS_SOURCE)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def stub_not_used(*args, **kwargs):
    raise AssertionError("stub should not be used")


if __name__ == "__main__":
    unittest.main()
