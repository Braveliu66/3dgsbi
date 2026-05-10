from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from queue import Empty


def _install_optional_dependency_stubs() -> None:
    if "geocalib" not in sys.modules:
        geocalib = types.ModuleType("geocalib")

        class GeoCalib:
            def cuda(self):
                return self

            def calibrate(self, *_args, **_kwargs):
                raise RuntimeError("GeoCalib is disabled; pass a calibration file instead")

        geocalib.GeoCalib = GeoCalib
        sys.modules["geocalib"] = geocalib

    if "e3nn.o3" not in sys.modules:
        e3nn = types.ModuleType("e3nn")
        o3 = types.ModuleType("e3nn.o3")

        def _unused(*_args, **_kwargs):
            raise RuntimeError("e3nn is not part of the video ARTDECO runtime")

        o3.matrix_to_angles = _unused
        o3.wigner_D = _unused
        e3nn.o3 = o3
        sys.modules["e3nn"] = e3nn
        sys.modules["e3nn.o3"] = o3

    if "open3d" not in sys.modules:
        open3d = types.ModuleType("open3d")
        open3d.geometry = types.SimpleNamespace(TriangleMesh=lambda: types.SimpleNamespace(vertices=None, triangles=None, vertex_colors=None))
        open3d.utility = types.SimpleNamespace(Vector3dVector=lambda value: value, Vector3iVector=lambda value: value)

        def write_triangle_mesh(path, _mesh):
            Path(path).write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="ascii")
            return True

        open3d.io = types.SimpleNamespace(write_triangle_mesh=write_triangle_mesh)
        sys.modules["open3d"] = open3d


_install_optional_dependency_stubs()


def main() -> int:
    parser = argparse.ArgumentParser(description="ARTDECO video trainer entrypoint with Speed3R-Pi3")
    parser.add_argument("--artdeco-root", required=True)
    parser.add_argument("--speed3r-root", default="")
    parser.add_argument("--speed3r-model-dir", required=True)
    parser.add_argument("--artdeco-config", required=True)
    parser.add_argument("--metrics-json", required=True)
    parsed, passthrough = parser.parse_known_args()

    artdeco_root = Path(parsed.artdeco_root).resolve()
    if not artdeco_root.exists():
        raise RuntimeError(f"ARTDECO_ROOT does not exist: {artdeco_root}")
    for path in (
        artdeco_root,
        artdeco_root / "VSLAM",
        artdeco_root / "VSLAM" / "thirdparty" / "mast3r",
        artdeco_root / "VSLAM" / "thirdparty" / "mast3r" / "dust3r",
    ):
        if path.exists():
            sys.path.insert(0, str(path))
    if parsed.speed3r_root:
        sys.path.insert(0, str(Path(parsed.speed3r_root).resolve()))

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("XFORMERS_DISABLED", "1")

    _install_speed3r_pi3_patch(Path(parsed.speed3r_model_dir), Path(parsed.speed3r_root) if parsed.speed3r_root else None)
    _install_adam_compat()
    _run_artdeco_mapping(passthrough, Path(parsed.artdeco_config).resolve(), Path(parsed.metrics_json))
    return 0


def _install_speed3r_pi3_patch(model_dir: Path, speed3r_root: Path | None) -> None:
    if speed3r_root and str(speed3r_root.resolve()) not in sys.path:
        sys.path.insert(0, str(speed3r_root.resolve()))
    import torch
    import VSLAM.mast3r_slam.retrieval_database as retrieval_database
    from pi3.models.pi3_sparse import Pi3_Sparse

    class Speed3RCompatiblePi3(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = Pi3_Sparse.from_pretrained(str(model_dir)).eval()

        def to(self, *args, **kwargs):
            self.inner.to(*args, **kwargs)
            return self

        def eval(self):
            self.inner.eval()
            return self

        def load_state_dict(self, _state_dict, strict: bool = True):
            return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

        def forward(self, images):
            return self.inner(images)

    retrieval_database.Pi3 = Speed3RCompatiblePi3
    retrieval_database.load_file = lambda *_args, **_kwargs: {}


def _install_adam_compat() -> None:
    try:
        from app.fine.video.artdeco_optimizer_compat import install_artdeco_adam_compat

        install_artdeco_adam_compat()
    except Exception:
        return


def _run_artdeco_mapping(passthrough: list[str], config_path: Path, metrics_json: Path) -> None:
    import numpy as np
    import pypose as pp
    import torch
    import torch.multiprocessing as mp
    from Reconstruct.scene.keyframe import Keyframe
    from VSLAM.Backend import Backend
    from VSLAM.Frontend import Frontend
    from VSLAM.ImageFrame import Mode
    from VSLAM.SharedKeyframes import SharedKeyframes
    from VSLAM.SharedStates import SharedStates
    from VSLAM.utils_config import load_config
    from VSLAM.utils_mp import new_queue
    from dataloaders.args import get_args
    from dataloaders.utils_load import load_dataset

    old_argv = sys.argv
    backend_process = None
    frontend_process = None
    sys.argv = [old_argv[0], *passthrough, "--config", str(config_path)]
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.random.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        mp.set_start_method("spawn", force=True)

        args = get_args()
        config = load_config(args.config)
        dataset = load_dataset(args)
        manager = mp.Manager()
        keyframes = SharedKeyframes(config, manager, dataset.H_slam, dataset.W_slam, dataset.K_slam, device=args.device_shared)
        states = SharedStates(manager, dataset.H_slam, dataset.W_slam, device=args.device_shared)
        frontend = Frontend(args, config, dataset, keyframes, states, device=args.device_frontend)
        backend = Backend(
            args,
            config,
            dataset,
            dataset.H_slam,
            dataset.W_slam,
            dataset.K_slam,
            states,
            keyframes,
            model=frontend.model if args.device_frontend == args.device_backend else None,
            device=args.device_backend,
        )
        backend_process = mp.Process(target=backend.run)
        frontend_process = mp.Process(target=frontend.run)
        backend_process.start()
        frontend_process.start()

        modules = __import__("Reconstruct.scene.scene_models." + args.base_model, fromlist=[""])
        scene_model = getattr(modules, "SceneModel")(dataset.W_map, dataset.H_map, dataset.K_map.to(args.device_backend), args, device=args.device_mapper)
        reconstruction_start_time = time.time()
        mapper_index = 0
        related_frames: dict[int, list[int]] = {}
        while True:
            mode = states.get_mode()
            try:
                keyframe_map_dict = states.msgFromBackend()
            except Empty:
                time.sleep(0.001)
                if mode == Mode.TERMINATED:
                    break
                continue
            _map_one_frame(args, dataset, keyframes, scene_model, keyframe_map_dict, related_frames, mapper_index)
            mapper_index += 1

        if len(scene_model.keyframes) <= 0:
            raise RuntimeError("ARTDECO did not produce any keyframes")
        reconstruction_time = time.time() - reconstruction_start_time
        scene_model.enable_inference_mode()
        with torch.cuda.device(args.device_mapper):
            metrics = scene_model.save(args.model_path, reconstruction_time, len(dataset))
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        frontend_process.join()
        backend_process.join()
    finally:
        for process in (frontend_process, backend_process):
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=5)
        sys.argv = old_argv


def _map_one_frame(args, dataset, keyframes, scene_model, keyframe_map_dict, related_frames, mapper_index: int) -> None:
    import pypose as pp
    import torch
    from Reconstruct.scene.keyframe import Keyframe

    frame_id = keyframe_map_dict["frame_id"]
    last_keyframe_index = keyframe_map_dict["last_keyframe_index"]
    last_keyframe_frame_id = keyframe_map_dict.get("last_keyframe_frame_id", None)
    related_frames.setdefault(last_keyframe_index, []).append(mapper_index)

    original_img, info = dataset[frame_id]
    t_ckc = keyframe_map_dict["T_CkC"]
    t_ckc = t_ckc.to(args.device_mapper) if t_ckc is not None else None
    t_wc = pp.Sim3(keyframe_map_dict["T_WC"]).to(args.device_mapper)
    dense_point = keyframe_map_dict["densePoint"].to(args.device_mapper)
    point_map = dense_point[..., :3]
    point_conf = dense_point[..., 3]
    t_cw = pp.quat2unit(pp.SE3(t_wc.data[:, :7])).Inv().matrix()[0]
    image_for_mapper = dataset.transform.to_map(original_img, device=args.device_mapper)
    keyframe_map = Keyframe(
        image_for_mapper,
        info["name"],
        keyframe_map_dict["is_test"],
        t_cw,
        mapper_index,
        frame_id,
        last_keyframe_index,
        last_keyframe_frame_id,
        keyframe_map_dict["is_slam_keyframe"],
        torch.tensor([dataset.K_map[0, 0].item()]).to(args.device_mapper),
        args,
        T_CkCf=t_ckc,
        point_map=point_map,
        point_conf=point_conf,
        device_mapper=args.device_mapper,
    )
    if keyframe_map_dict["is_slam_keyframe"] and frame_id > 0 and len(scene_model.keyframes) > 0:
        old_c2ws = torch.zeros(len(scene_model.keyframes), 4, 4).to(args.device_mapper)
        new_c2ws = torch.zeros(len(scene_model.keyframes), 4, 4).to(args.device_mapper)
        cam_centres = torch.zeros(len(scene_model.keyframes), 3).to(args.device_mapper)
        for index, frame_ids in related_frames.items():
            for mapper_frame_id in frame_ids:
                if mapper_frame_id == len(scene_model.keyframes):
                    continue
                frame = scene_model.keyframes[mapper_frame_id]
                frame_slam = keyframes[frame.last_keyframe_index]
                if frame.is_slam_keyframe:
                    t_wcf = pp.quat2unit(pp.SE3(frame_slam.T_WC.data[:, :7])).to(args.device_mapper)
                else:
                    t_wck = frame_slam.T_WC.to(args.device_mapper)
                    t_wcf = pp.SE3(pp.quat2unit(t_wck.mul(frame.T_CkCf)).data[:, :7])
                new_rt = t_wcf.Inv().matrix()[0]
                old_rt = frame.get_Rt()
                frame.set_Rt(new_rt.to(old_rt.device))
                view_matrix = frame.get_Rt().transpose(0, 1).to(args.device_mapper)
                old_c2ws[mapper_frame_id] = torch.linalg.inv(old_rt).to(args.device_mapper)
                new_c2ws[mapper_frame_id] = torch.linalg.inv(new_rt).to(args.device_mapper)
                cam_centres[mapper_frame_id] = view_matrix.detach().inverse()[3, :3].to(args.device_mapper)
        scene_model.rigid_transform_gs(old_c2ws, new_c2ws, cam_centres)
    with torch.cuda.device(args.device_mapper):
        scene_model.add_keyframe(keyframe_map)
        if keyframe_map_dict["is_important"]:
            scene_model.add_new_gaussians()
        iterations = args.num_key_iterations if keyframe_map_dict["is_important"] else args.num_common_iterations
        scene_model.optimization_loop(iterations, keyframe_map_dict["is_important"])


if __name__ == "__main__":
    raise SystemExit(main())
