from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
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


def _log(message: str) -> None:
    print(f"[artdeco-entrypoint] {message}", flush=True)


def _cuda_summary(device: str) -> str:
    if not str(device).startswith("cuda"):
        return "cuda=disabled"
    try:
        import torch

        index = torch.device(device).index or 0
        allocated = torch.cuda.memory_allocated(index) / (1024**2)
        reserved = torch.cuda.memory_reserved(index) / (1024**2)
        return f"cuda_allocated_mb={allocated:.1f} cuda_reserved_mb={reserved:.1f}"
    except Exception as exc:
        return f"cuda_summary_unavailable={exc}"


def _scene_summary(scene_model) -> str:
    parts: list[str] = []
    try:
        parts.append(f"keyframes={len(scene_model.keyframes)}")
    except Exception:
        pass
    xyz = getattr(scene_model, "xyz", None)
    if xyz is not None and hasattr(xyz, "shape"):
        try:
            parts.append(f"xyz_shape={tuple(xyz.shape)}")
        except Exception:
            pass
    opacity = getattr(scene_model, "opacity", None)
    if opacity is not None and hasattr(opacity, "shape"):
        try:
            parts.append(f"opacity_shape={tuple(opacity.shape)}")
        except Exception:
            pass
    return " ".join(parts) if parts else "scene_summary=unavailable"


def _shape_summary(value) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return type(value).__name__
    try:
        return str(tuple(int(item) for item in shape))
    except Exception:
        return str(shape)


def _optimizer_summary(scene_model, limit: int = 8) -> str:
    optimizer = getattr(scene_model, "optimizer", None)
    params = getattr(optimizer, "params", None)
    if not isinstance(params, dict):
        return "optimizer_summary=unavailable"
    parts: list[str] = []
    for index, (name, param) in enumerate(params.items()):
        if index >= limit:
            parts.append(f"...+{len(params) - limit} more")
            break
        if not isinstance(param, dict):
            parts.append(f"{name}:param={type(param).__name__}")
            continue
        value = param.get("val")
        lr = param.get("lr")
        grad = getattr(value, "grad", None)
        parts.append(f"{name}:val={_shape_summary(value)} grad={_shape_summary(grad)} lr={_shape_summary(lr)}")
    return "optimizer_params=[" + "; ".join(parts) + "]"


def _render_selection_summary(scene_model, keyframe_id: int = -1) -> str:
    try:
        import torch

        with torch.no_grad():
            total = int(scene_model.n_active_gaussians)
            if total <= 0 or not scene_model.keyframes:
                return "render_selection total=0"
            keyframe = scene_model.keyframes[keyframe_id]
            view_matrix = keyframe.get_Rt().to(scene_model.device)
            cam_centre = view_matrix.detach().inverse()[:3, 3].to(scene_model.device)
            ob_dist = torch.linalg.vector_norm(scene_model.xyz - cam_centre, dim=1, keepdim=True)
            selection_mask = (ob_dist < 2 * scene_model.d_max).squeeze(-1)
            selected = int(selection_mask.sum().item())
            pct = selected * 100.0 / max(total, 1)
            return f"render_selection selected={selected} total={total} selected_pct={pct:.1f}"
    except Exception as exc:
        return f"render_selection_unavailable={exc}"


def _run_with_heartbeat(label: str, device: str, scene_model, target) -> None:
    done = threading.Event()
    start = time.monotonic()

    def heartbeat() -> None:
        while not done.wait(15):
            elapsed = time.monotonic() - start
            _log(f"{label} still running elapsed={elapsed:.1f}s {_cuda_summary(device)} {_scene_summary(scene_model)}")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        target()
    except BaseException as exc:
        elapsed = time.monotonic() - start
        _log(f"{label} failed elapsed={elapsed:.1f}s error={exc} {_cuda_summary(device)} {_scene_summary(scene_model)}")
        raise
    finally:
        done.set()
        thread.join(timeout=0.2)
    elapsed = time.monotonic() - start
    _log(f"{label} done elapsed={elapsed:.1f}s {_cuda_summary(device)} {_scene_summary(scene_model)}")


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
    _log(
        "runtime paths "
        f"artdeco_root={artdeco_root} speed3r_root={parsed.speed3r_root or '<none>'} "
        f"speed3r_model_dir={parsed.speed3r_model_dir}"
    )

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("XFORMERS_DISABLED", "1")

    _log_simple_knn_status()
    _install_speed3r_pi3_patch(Path(parsed.speed3r_model_dir), Path(parsed.speed3r_root) if parsed.speed3r_root else None)
    _install_adam_compat()
    _run_artdeco_mapping(
        passthrough,
        Path(parsed.artdeco_config).resolve(),
        Path(parsed.metrics_json),
        Path(parsed.speed3r_model_dir),
        Path(parsed.speed3r_root) if parsed.speed3r_root else None,
    )
    return 0


def _log_simple_knn_status() -> None:
    try:
        import simple_knn._C as simple_knn_c

        symbols = [name for name in ("distCUDA2", "distIndex2", "distIndexQ") if hasattr(simple_knn_c, name)]
        _log(f"simple_knn extension file={getattr(simple_knn_c, '__file__', '<unknown>')} symbols={symbols}")
    except Exception as exc:
        _log(f"simple_knn extension import failed: {exc}")


def _install_speed3r_pi3_patch(model_dir: Path, speed3r_root: Path | None) -> None:
    if speed3r_root and str(speed3r_root.resolve()) not in sys.path:
        sys.path.insert(0, str(speed3r_root.resolve()))
    import torch
    import VSLAM.mast3r_slam.retrieval_database as retrieval_database
    from pi3.models.pi3_sparse import Pi3_Sparse
    _log(f"installing Speed3R Pi3 patch model_dir={model_dir} speed3r_root={speed3r_root}")

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
    _log("installed Speed3R Pi3 patch")


def _install_adam_compat() -> None:
    try:
        from app.fine.video.artdeco_optimizer_compat import install_artdeco_adam_compat

        install_artdeco_adam_compat()
        _log("installed ARTDECO Adam compatibility symbols")
    except Exception as exc:
        _log(f"ARTDECO Adam compatibility install skipped: {exc}")
        return


def _run_artdeco_frontend_process(args, config, dataset, keyframes, states) -> None:
    from VSLAM.Frontend import Frontend

    frontend = Frontend(args, config, dataset, keyframes, states, device=args.device_frontend)
    frontend.run()


def _run_artdeco_backend_process(
    args,
    config,
    dataset,
    h_slam,
    w_slam,
    k_slam,
    states,
    keyframes,
    speed3r_model_dir: Path,
    speed3r_root: Path | None,
) -> None:
    _install_speed3r_pi3_patch(speed3r_model_dir, speed3r_root)
    from VSLAM.Backend import Backend

    backend = Backend(args, config, dataset, h_slam, w_slam, k_slam, states, keyframes, model=None, device=args.device_backend)
    backend.run()


def _run_artdeco_mapping(
    passthrough: list[str],
    config_path: Path,
    metrics_json: Path,
    speed3r_model_dir: Path,
    speed3r_root: Path | None,
) -> None:
    import numpy as np
    import pypose as pp
    import torch
    import torch.multiprocessing as mp
    from Reconstruct.scene.keyframe import Keyframe
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
    thread_errors: list[BaseException] = []
    sys.argv = [old_argv[0], *passthrough, "--config", str(config_path)]
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.random.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        mp.set_start_method("spawn", force=True)

        args = get_args()
        args.sh_degree = int(args.sh_degree)
        _log(
            "args loaded "
            f"frontend={args.device_frontend} backend={args.device_backend} "
            f"mapper={args.device_mapper} shared={args.device_shared} "
            f"cuda_available={torch.cuda.is_available()} "
            f"downsampling={args.downsampling} gs_add_ratio={args.gs_add_ratio} "
            f"visible_threshold={args.visible_threshold} sh_degree={args.sh_degree} "
            f"max_active_keyframes={args.max_active_keyframes} gaussian_total_cap=disabled"
        )
        config = load_config(args.config)
        _log("loading dataset")
        dataset = load_dataset(args)
        _log(f"dataset loaded frames={len(dataset)} slam={dataset.W_slam}x{dataset.H_slam} map={dataset.W_map}x{dataset.H_map}")
        manager = mp.Manager()
        _log("creating shared VSLAM state")
        keyframe_buffer = max(len(dataset) + 8, 16)
        keyframes = SharedKeyframes(config, manager, dataset.H_slam, dataset.W_slam, dataset.K_slam, buffer=keyframe_buffer, device=args.device_shared)
        states = SharedStates(manager, dataset.H_slam, dataset.W_slam, device=args.device_shared)
        if args.device_frontend == args.device_backend:
            from VSLAM.Backend import Backend
            from VSLAM.Frontend import Frontend

            _log("starting threaded frontend/backend with shared model")
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
                model=frontend.model,
                device=args.device_backend,
            )

            def run_threaded(name: str, target) -> None:
                try:
                    target()
                except BaseException as exc:
                    _log(f"{name} thread failed: {exc}")
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
                    thread_errors.append(RuntimeError(f"ARTDECO {name} thread failed: {exc}"))

            backend_process = threading.Thread(target=run_threaded, args=("backend", backend.run), daemon=True)
            frontend_process = threading.Thread(target=run_threaded, args=("frontend", frontend.run), daemon=True)
            backend_process.start()
            frontend_process.start()
            _log("started threaded backend/frontend")
        else:
            _log("starting frontend/backend processes")
            backend_process = mp.Process(
                target=_run_artdeco_backend_process,
                args=(
                    args,
                    config,
                    dataset,
                    dataset.H_slam,
                    dataset.W_slam,
                    dataset.K_slam,
                    states,
                    keyframes,
                    speed3r_model_dir,
                    speed3r_root,
                ),
            )
            frontend_process = mp.Process(target=_run_artdeco_frontend_process, args=(args, config, dataset, keyframes, states))
            backend_process.start()
            frontend_process.start()
            _log(f"started backend pid={backend_process.pid} frontend pid={frontend_process.pid}")

        _log(f"initializing mapper model={args.base_model}")
        modules = __import__("Reconstruct.scene.scene_models." + args.base_model, fromlist=[""])
        _log(f"imported mapper module={modules.__name__}")
        scene_model = getattr(modules, "SceneModel")(dataset.W_map, dataset.H_map, dataset.K_map.to(args.device_backend), args, device=args.device_mapper)
        _log(f"mapper initialized; waiting for backend keyframes {_cuda_summary(args.device_mapper)}")
        reconstruction_start_time = time.time()
        last_wait_log = reconstruction_start_time
        mapper_index = 0
        related_frames: dict[int, list[int]] = {}
        while True:
            mode = states.get_mode()
            try:
                keyframe_map_dict = states.msgFromBackend()
            except Empty:
                time.sleep(0.001)
                if thread_errors:
                    raise thread_errors[0]
                if mode == Mode.TERMINATED:
                    break
                for name, process in (("backend", backend_process), ("frontend", frontend_process)):
                    if process is not None and not isinstance(process, threading.Thread) and not process.is_alive() and process.exitcode is not None:
                        raise RuntimeError(f"ARTDECO {name} process exited before mapping completed with code {process.exitcode}")
                now = time.time()
                if now - last_wait_log >= 10:
                    _log(f"waiting for backend keyframes for {now - reconstruction_start_time:.1f}s")
                    last_wait_log = now
                continue
            _log(
                f"mapping frame_id={keyframe_map_dict['frame_id']} mapper_index={mapper_index} "
                f"important={keyframe_map_dict['is_important']} slam_keyframe={keyframe_map_dict['is_slam_keyframe']}"
            )
            map_start = time.time()
            _map_one_frame(args, dataset, keyframes, scene_model, keyframe_map_dict, related_frames, mapper_index)
            _log(
                f"mapped frame_id={keyframe_map_dict['frame_id']} mapper_index={mapper_index} "
                f"duration={time.time() - map_start:.2f}s keyframes={len(scene_model.keyframes)} "
                f"{_cuda_summary(args.device_mapper)}"
            )
            mapper_index += 1

        if len(scene_model.keyframes) <= 0:
            raise RuntimeError("ARTDECO did not produce any keyframes")
        reconstruction_time = time.time() - reconstruction_start_time
        scene_model.enable_inference_mode()
        with torch.cuda.device(args.device_mapper):
            metrics = scene_model.save(args.model_path, reconstruction_time, len(dataset))
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(f"saved reconstruction keyframes={len(scene_model.keyframes)}")
        frontend_process.join()
        backend_process.join()
    finally:
        _log("cleaning up frontend/backend workers")
        for process in (frontend_process, backend_process):
            if isinstance(process, threading.Thread):
                if process.is_alive():
                    try:
                        states.set_mode(Mode.TERMINATED)
                    except Exception:
                        pass
                    process.join(timeout=5)
            elif process is not None and process.is_alive():
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
    _log(
        f"map_one_frame start frame_id={frame_id} mapper_index={mapper_index} "
        f"last_keyframe_index={last_keyframe_index} last_keyframe_frame_id={last_keyframe_frame_id}"
    )

    original_img, info = dataset[frame_id]
    t_ckc = keyframe_map_dict["T_CkC"]
    t_ckc = t_ckc.to(args.device_mapper) if t_ckc is not None else None
    t_wc = pp.Sim3(keyframe_map_dict["T_WC"]).to(args.device_mapper)
    dense_point = keyframe_map_dict["densePoint"].to(args.device_mapper)
    point_map = dense_point[..., :3]
    point_conf = dense_point[..., 3]
    _log(
        f"map_one_frame tensors frame_id={frame_id} dense_shape={tuple(dense_point.shape)} "
        f"point_conf_dtype={point_conf.dtype} {_cuda_summary(args.device_mapper)}"
    )
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
        _log(f"map_one_frame applying rigid transform frame_id={frame_id} existing_keyframes={len(scene_model.keyframes)}")
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
        _run_with_heartbeat(
            f"map_one_frame rigid_transform_gs frame_id={frame_id}",
            args.device_mapper,
            scene_model,
            lambda: scene_model.rigid_transform_gs(old_c2ws, new_c2ws, cam_centres),
        )
    with torch.cuda.device(args.device_mapper):
        _log(f"map_one_frame add_keyframe start frame_id={frame_id}")
        _run_with_heartbeat(
            f"map_one_frame add_keyframe frame_id={frame_id}",
            args.device_mapper,
            scene_model,
            lambda: scene_model.add_keyframe(keyframe_map),
        )
        _log(f"map_one_frame add_keyframe done frame_id={frame_id} keyframes={len(scene_model.keyframes)}")
        if keyframe_map_dict["is_important"]:
            before_gaussians = scene_model.n_active_gaussians
            _log(f"map_one_frame add_new_gaussians start frame_id={frame_id} gaussians={before_gaussians} total_cap=disabled")
            _run_with_heartbeat(
                f"map_one_frame add_new_gaussians frame_id={frame_id}",
                args.device_mapper,
                scene_model,
                scene_model.add_new_gaussians,
            )
            _log(
                f"map_one_frame add_new_gaussians done frame_id={frame_id} "
                f"before={before_gaussians} after={scene_model.n_active_gaussians} "
                f"added={scene_model.n_active_gaussians - before_gaussians}"
            )
        iterations = args.num_key_iterations if keyframe_map_dict["is_important"] else args.num_common_iterations
        _log(f"map_one_frame optimization start frame_id={frame_id} iterations={iterations} important={keyframe_map_dict['is_important']}")
        _log(f"map_one_frame render selection frame_id={frame_id} {_render_selection_summary(scene_model)}")
        _log(f"map_one_frame optimizer state frame_id={frame_id} {_optimizer_summary(scene_model)}")
        _run_with_heartbeat(
            f"map_one_frame optimization frame_id={frame_id} iterations={iterations}",
            args.device_mapper,
            scene_model,
            lambda: scene_model.optimization_loop(iterations, keyframe_map_dict["is_important"]),
        )
        _log(f"map_one_frame optimization done frame_id={frame_id} iterations={iterations}")


if __name__ == "__main__":
    raise SystemExit(main())
