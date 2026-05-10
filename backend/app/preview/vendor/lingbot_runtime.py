from __future__ import annotations

import gc
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.preview.io.ply import write_point_cloud_ply
from app.preview.types import PreviewFailure
from app.preview.utils import VENDOR_ROOT, image_files, prepend_sys_path


Progress = Callable[[str, int, str, dict[str, Any] | None], None]
InferenceProgress = Callable[[dict[str, Any]], None]


class LingBotInferenceReporter:
    def __init__(
        self,
        progress: Progress,
        *,
        min_interval_seconds: float = 5.0,
        min_frame_step: int = 10,
    ) -> None:
        self.progress = progress
        self.min_interval_seconds = min_interval_seconds
        self.min_frame_step = min_frame_step
        self.started_at = time.monotonic()
        self.last_emit_at = 0.0
        self.last_frame = 0
        self.last_window = 0
        self.metrics: dict[str, Any] = {}

    def __call__(self, event: dict[str, Any]) -> None:
        kind = str(event.get("type") or "")
        if kind == "streaming_frame":
            self._report_frame(event)
        elif kind == "windowed_window":
            self._report_window(event)

    def _should_emit(self, *, frame: int = 0, window: int = 0, total: int = 0) -> bool:
        now = time.monotonic()
        if total and (frame >= total or window >= total):
            return True
        if frame and frame - self.last_frame >= self.min_frame_step:
            return True
        if window and window > self.last_window:
            return True
        return now - self.last_emit_at >= self.min_interval_seconds

    def _mark_emitted(self, *, frame: int = 0, window: int = 0) -> None:
        self.last_emit_at = time.monotonic()
        if frame:
            self.last_frame = frame
        if window:
            self.last_window = window

    def _report_frame(self, event: dict[str, Any]) -> None:
        current = int(event.get("current_frame") or 0)
        total = int(event.get("total_frames") or 0)
        if current <= 0 or total <= 0 or not self._should_emit(frame=current, total=total):
            return
        elapsed = float(event.get("elapsed_seconds") or max(0.0, time.monotonic() - self.started_at))
        seconds_per_frame = float(event.get("seconds_per_frame") or (elapsed / max(current, 1)))
        eta = max(0, int(math.ceil((total - current) * seconds_per_frame)))
        percent = inference_percent(current, total)
        metrics = {
            "lingbot_current_frame": current,
            "lingbot_total_frames": total,
            "lingbot_seconds_per_frame": round(seconds_per_frame, 3),
            "lingbot_current_inference_fps": round(current / max(elapsed, 1e-6), 3),
            "lingbot_inference_eta_seconds": eta,
        }
        self.metrics = {**self.metrics, **metrics}
        self._mark_emitted(frame=current)
        self.progress(
            "lingbot_inference",
            percent,
            f"LingBot inference: frame {current}/{total}, {seconds_per_frame:.1f}s/frame, ETA {format_eta_seconds(eta)}",
            metrics,
        )

    def _report_window(self, event: dict[str, Any]) -> None:
        current = int(event.get("current_window") or 0)
        total = int(event.get("total_windows") or 0)
        if current <= 0 or total <= 0 or not self._should_emit(window=current, total=total):
            return
        total_frames = int(event.get("total_frames") or 0)
        covered = int(event.get("covered_frames") or 0)
        window_start = int(event.get("window_start") or 0)
        window_end = int(event.get("window_end") or covered)
        elapsed = float(event.get("elapsed_seconds") or max(0.0, time.monotonic() - self.started_at))
        seconds_per_window = elapsed / max(current, 1)
        eta = max(0, int(math.ceil((total - current) * seconds_per_window)))
        percent = inference_percent(current, total)
        metrics = {
            "lingbot_current_window": current,
            "lingbot_total_windows": total,
            "lingbot_current_frame": covered,
            "lingbot_total_frames": total_frames,
            "lingbot_seconds_per_window": round(seconds_per_window, 3),
            "lingbot_inference_eta_seconds": eta,
        }
        self.metrics = {**self.metrics, **metrics}
        self._mark_emitted(window=current)
        self.progress(
            "lingbot_inference",
            percent,
            f"LingBot window {current}/{total}, frames {window_start}-{window_end}/{total_frames}, ETA {format_eta_seconds(eta)}",
            metrics,
        )


@dataclass(slots=True)
class LingBotPlan:
    input_mode: str
    mode: str
    backend: str
    use_sdpa: bool
    compile: bool
    offload_to_cpu: bool
    dtype_name: str
    num_scale_frames: int
    keyframe_interval: int
    kv_cache_sliding_window: int
    window_size: int
    overlap_size: int
    overlap_keyframes: int | None
    window_profile: str
    camera_num_iterations: int
    keyframes_only_points: bool
    max_frame_num: int
    frame_budget: int
    original_frame_count: int
    selected_frame_count: int
    gpu_memory_total_mb: int
    gpu_memory_free_mb: int
    profile: str
    retry_level: int = 0


@dataclass(slots=True)
class FrameSelectionResult:
    frame_dir: Path
    selected_count: int
    source_frame_count: int
    source_fps: float
    mode: str
    reasons: dict[str, int] = field(default_factory=dict)

    def metrics(self) -> dict[str, Any]:
        return {
            "lingbot_frame_selection": self.mode,
            "lingbot_source_video_frames": self.source_frame_count,
            "lingbot_selected_frame_count": self.selected_count,
            "lingbot_source_fps": round(self.source_fps, 3) if self.source_fps else 0,
            "lingbot_frame_selection_reasons": self.reasons,
        }


@dataclass(slots=True)
class PointExportData:
    world_points: np.ndarray
    confidence: np.ndarray
    colors: np.ndarray
    is_keyframe: np.ndarray | None
    metrics: dict[str, Any]


def run_lingbot_pointcloud(
    *,
    input_dir: Path,
    input_video: Path | None,
    checkpoint_path: Path,
    output_ply: Path,
    fps: int,
    max_frames: int,
    confidence_quantile: float,
    max_points: int,
    progress: Progress,
    runtime_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_options = runtime_options or {}
    configure_torch_compile_cache(runtime_options)

    import torch

    if not torch.cuda.is_available():
        raise PreviewFailure("GPU_RESOURCE_UNAVAILABLE", "LingBot-Map preview requires CUDA")
    torch.cuda.reset_peak_memory_stats()

    input_mode = str(runtime_options.get("lingbot_input_mode") or ("offline_video" if input_video else "image_sequence"))

    with prepend_sys_path(VENDOR_ROOT / "lingbot"):
        from lingbot_map.models.gct_stream import GCTStream
        from lingbot_map.models.gct_stream_window import GCTStream as GCTStreamWindow
        from lingbot_map.utils.load_fn import load_and_preprocess_images

        frame_dir = input_dir
        selection_metrics: dict[str, Any] = {}
        if input_video:
            frame_selection = normalized_option(runtime_options.get("lingbot_frame_selection")) or (
                "fixed_fps" if input_mode in {"realtime_camera", "offline_video"} else "scene_keyframes"
            )
            stage = "extracting realtime camera chunk frames" if input_mode == "realtime_camera" else "selecting video scene keyframes"
            progress("lingbot_extract_frames", 18, f"{stage} ({frame_selection})")
            frame_dir = Path(tempfile.mkdtemp(prefix="lingbot_frames_", dir=str(output_ply.parent)))
            if frame_selection == "scene_keyframes":
                selection = extract_scene_keyframes(input_video, frame_dir, runtime_options=runtime_options)
            else:
                selection = extract_frames(input_video, frame_dir, fps=fps)
            selection_metrics = selection.metrics()
            progress(
                "lingbot_extract_frames",
                20,
                f"selected {selection.selected_count}/{selection.source_frame_count} video frames using {selection.mode}",
                selection_metrics,
            )
            if selection.selected_count == 0:
                raise PreviewFailure("VIDEO_FRAME_EXTRACTION_FAILED", f"no frames extracted from {input_video}")

        all_files = image_files(frame_dir)
        files = select_frame_files(all_files, max_frames)
        if len(files) < 2:
            raise PreviewFailure("LINGBOT_NOT_ENOUGH_FRAMES", "LingBot-Map preview requires at least 2 frames")

        progress("lingbot_preprocess", 28, f"loading {len(files)} frames")
        images = load_and_preprocess_images([str(path) for path in files], mode="crop", image_size=518, patch_size=14)

        base_plan = resolve_lingbot_plan(
            torch=torch,
            frame_count=len(files),
            original_frame_count=len(all_files),
            frame_budget=max_frames,
            input_mode=input_mode,
            runtime_options=runtime_options,
        )
        plans = [base_plan, downgrade_plan(base_plan)]
        predictions: dict[str, torch.Tensor] | None = None
        inference_metrics: dict[str, Any] = {}
        used_plan = base_plan
        last_oom: RuntimeError | None = None

        for plan in plans:
            try:
                predictions, inference_metrics = run_inference_once(
                    torch=torch,
                    model_cls=GCTStreamWindow if plan.mode == "windowed" else GCTStream,
                    checkpoint_path=checkpoint_path,
                    images=images,
                    plan=plan,
                    progress=progress,
                )
                used_plan = plan
                break
            except RuntimeError as exc:
                if not is_cuda_oom(exc) or plan.retry_level > 0:
                    raise
                last_oom = exc
                cleanup_cuda(torch)
                progress(
                    "lingbot_oom_retry",
                    50,
                    "CUDA OOM; retrying LingBot with a lower-memory adaptive profile",
                )

        if predictions is None:
            message = str(last_oom) if last_oom else "LingBot inference did not produce predictions"
            raise PreviewFailure("CUDA_OUT_OF_MEMORY", message)

        progress("lingbot_export_ply", 72, "filtering world points and writing PLY")
        export = prepare_lingbot_point_export(predictions, images)
        export = filter_keyframe_point_export(export, enabled=used_plan.keyframes_only_points)
        progress(
            "lingbot_export_ply",
            73,
            (
                "prepared LingBot point export "
                f"world={export.metrics['lingbot_export_world_points_shape']} "
                f"conf={export.metrics['lingbot_export_conf_shape']} "
                f"colors={export.metrics['lingbot_export_colors_shape']}"
            ),
            export.metrics,
        )

        threshold = np.quantile(export.confidence.reshape(-1), confidence_quantile)
        mask = export.confidence >= threshold
        selected_points = int(mask.sum())
        progress(
            "lingbot_export_ply",
            74,
            f"writing LingBot PLY from {selected_points} confidence-filtered points",
            {"lingbot_export_filtered_points": selected_points, "preview_max_points": max_points},
        )
        point_count = write_point_cloud_ply(
            export.world_points[mask],
            export.colors[mask],
            output_ply,
            confidence=export.confidence[mask],
            max_points=max_points,
        )
        peak_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024)
        plan_metrics = asdict(used_plan)
        return {
            "input_frame_count": len(files),
            "source_frame_count": len(all_files),
            "lingbot_fps": fps,
            "point_count": point_count,
            "confidence_quantile": confidence_quantile,
            "cuda_memory_peak_mb": peak_mb,
            "lingbot_mode": used_plan.mode,
            "lingbot_backend": used_plan.backend,
            "lingbot_dtype": used_plan.dtype_name,
            "lingbot_compile": inference_metrics.get("lingbot_compile_effective", used_plan.compile),
            "lingbot_compile_requested": used_plan.compile,
            "lingbot_offload_to_cpu": used_plan.offload_to_cpu,
            "lingbot_keyframe_interval": used_plan.keyframe_interval,
            "lingbot_window_size": used_plan.window_size,
            "lingbot_overlap_size": used_plan.overlap_size,
            "lingbot_overlap_keyframes": used_plan.overlap_keyframes,
            "lingbot_keyframes_only_points": used_plan.keyframes_only_points,
            "lingbot_window_profile": used_plan.window_profile,
            "lingbot_profile": used_plan.profile,
            "lingbot_retry_level": used_plan.retry_level,
            "lingbot_plan": plan_metrics,
            "lingbot_point_cloud_ply": str(output_ply),
            "lingbot_point_cloud_ply_size": output_ply.stat().st_size if output_ply.exists() else None,
            **selection_metrics,
            **inference_metrics,
            **export.metrics,
        }


def run_inference_once(
    *,
    torch,
    model_cls,
    checkpoint_path: Path,
    images,
    plan: LingBotPlan,
    progress: Progress,
) -> tuple[dict[str, Any], dict[str, Any]]:
    device = torch.device("cuda")
    dtype = dtype_from_plan(torch, plan)
    progress("lingbot_loading_model", 36, f"loading LingBot checkpoint: {checkpoint_path.name}")
    model = model_cls(
        img_size=518,
        patch_size=14,
        enable_point=True,
        enable_3d_rope=True,
        max_frame_num=plan.max_frame_num,
        kv_cache_sliding_window=plan.kv_cache_sliding_window,
        kv_cache_scale_frames=plan.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=plan.use_sdpa,
        camera_num_iterations=plan.camera_num_iterations,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)

    compile_metrics: dict[str, Any] = {
        "lingbot_compile_requested": plan.compile,
        "lingbot_compile_effective": False,
        "compile_warm_seconds": 0.0,
        "compile_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
    }
    should_compile = should_compile_lingbot_plan(plan)
    if should_compile:
        compile_started = time.monotonic()
        try:
            warmup_streaming_compile(torch, model, images, plan, dtype, device, progress)
            compile_warm_seconds = round(time.monotonic() - compile_started, 3)
            compile_metrics = {
                **compile_metrics,
                "lingbot_compile_effective": True,
                "lingbot_compile_fallback": False,
                "compile_warm_seconds": compile_warm_seconds,
                "lingbot_compile_warm_seconds": compile_warm_seconds,
            }
        except Exception as exc:
            cleanup_cuda(torch)
            compile_warm_seconds = round(time.monotonic() - compile_started, 3)
            compile_metrics = {
                **compile_metrics,
                "lingbot_compile_fallback": True,
                "lingbot_compile_error": str(exc),
                "compile_warm_seconds": compile_warm_seconds,
                "lingbot_compile_warm_seconds": compile_warm_seconds,
            }
            progress(
                "lingbot_compile_fallback",
                44,
                f"torch.compile failed; continuing with eager LingBot inference: {exc}",
                compile_metrics,
            )
    elif plan.compile:
        compile_metrics = {
            **compile_metrics,
            "lingbot_compile_warmup_skipped": True,
            "lingbot_compile_warmup_threshold_frames": 500,
            "lingbot_compile_warmup_frame_count": plan.selected_frame_count,
        }
        progress(
            "lingbot_compile_skipped",
            44,
            f"skipping torch.compile warmup for short LingBot video ({plan.selected_frame_count}/500 frames)",
            compile_metrics,
        )

    cleanup_cuda(torch)
    progress(
        "lingbot_inference",
        48,
        (
            f"running {plan.mode} LingBot inference "
            f"(backend={plan.backend}, dtype={plan.dtype_name}, compile={compile_metrics['lingbot_compile_effective']}, "
            f"keyframe_interval={plan.keyframe_interval})"
        ),
    )
    output_device = torch.device("cpu") if plan.offload_to_cpu else None
    reporter = LingBotInferenceReporter(progress)
    synchronize_cuda(torch)
    inference_started = time.monotonic()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if plan.mode == "windowed":
            predictions = model.inference_windowed(
                images,
                window_size=plan.window_size,
                overlap_size=plan.overlap_size,
                overlap_keyframes=plan.overlap_keyframes,
                num_scale_frames=plan.num_scale_frames,
                keyframe_interval=plan.keyframe_interval,
                output_device=output_device,
                progress_callback=reporter,
            )
        else:
            predictions = model.inference_streaming(
                images,
                num_scale_frames=plan.num_scale_frames,
                keyframe_interval=plan.keyframe_interval,
                output_device=output_device,
                progress_callback=reporter,
            )
    synchronize_cuda(torch)
    throughput_metrics = inference_throughput_metrics(int(images.shape[0]), time.monotonic() - inference_started)
    progress(
        "lingbot_inference",
        72,
        f"LingBot inference completed at {throughput_metrics['inference_fps']:.2f} fps",
        throughput_metrics,
    )
    del model
    cleanup_cuda(torch)
    return predictions, {**compile_metrics, **reporter.metrics, **throughput_metrics}


def resolve_lingbot_plan(
    *,
    torch,
    frame_count: int,
    original_frame_count: int,
    frame_budget: int,
    input_mode: str,
    runtime_options: dict[str, Any],
) -> LingBotPlan:
    total_mb, free_mb = cuda_memory_mb(torch)
    low_vram = total_mb and total_mb < 14_000
    realtime = input_mode == "realtime_camera"
    offline_video = input_mode == "offline_video"
    forced_mode = normalized_option(runtime_options.get("lingbot_mode"))
    mode = forced_mode if forced_mode in {"streaming", "windowed"} else ("streaming" if realtime or frame_count <= 500 else "windowed")

    backend = normalized_option(option_value(runtime_options, "lingbot_backend", "backend")) or "flashinfer"
    if backend not in {"flashinfer", "sdpa"}:
        raise PreviewFailure("LINGBOT_BACKEND_UNSUPPORTED", f"unsupported LingBot backend: {backend}")
    allow_sdpa_fallback = parse_bool(option_value(runtime_options, "lingbot_allow_sdpa_fallback", "allow_sdpa_fallback"), False)
    require_flashinfer = parse_bool(option_value(runtime_options, "lingbot_require_flashinfer", "require_flashinfer"), offline_video)
    if backend == "flashinfer" and not flashinfer_available():
        if allow_sdpa_fallback and not require_flashinfer:
            backend = "sdpa"
        else:
            raise PreviewFailure(
                "FLASHINFER_UNAVAILABLE",
                "FlashInfer is required for the default LingBot preview backend; install flashinfer-python or set lingbot_backend=sdpa",
            )

    dtype_name = resolve_dtype_name(torch, normalized_option(option_value(runtime_options, "lingbot_dtype", "dtype")) or "auto")
    compile_default = offline_video
    compile_enabled = parse_bool(option_value(runtime_options, "lingbot_compile", "compile", "torch_compile"), compile_default)
    compile_enabled = bool(compile_enabled and hasattr(torch, "compile"))
    offload_to_cpu = parse_bool(option_value(runtime_options, "lingbot_offload_to_cpu", "offload_to_cpu"), True)

    keyframe_target = read_int(option_value(runtime_options, "lingbot_keyframe_target", "keyframe_target"), 160 if realtime else 240 if low_vram else 320)
    keyframe_interval_default = 8 if realtime or offline_video else 0
    keyframe_interval = read_int(option_value(runtime_options, "lingbot_keyframe_interval", "keyframe_interval"), keyframe_interval_default)
    if keyframe_interval <= 0:
        keyframe_interval = max(1, math.ceil(frame_count / max(keyframe_target, 1))) if mode == "streaming" and frame_count > keyframe_target else 1

    num_scale_frames = read_int(option_value(runtime_options, "lingbot_num_scale_frames", "num_scale_frames"), 2 if realtime or offline_video else 8)
    num_scale_frames = min(max(1, num_scale_frames), max(1, frame_count - 1))
    window_profile, default_window_size, default_overlap_size = resolve_window_profile(
        total_mb,
        normalized_option(option_value(runtime_options, "lingbot_window_profile", "window_profile")) or "auto",
    )
    window_size = read_int(option_value(runtime_options, "lingbot_window_size", "window_size"), default_window_size if offline_video else 48 if low_vram else 64)
    overlap_size = read_int(option_value(runtime_options, "lingbot_overlap_size", "overlap_size"), default_overlap_size if offline_video else 12 if low_vram else 16)
    overlap_keyframes = read_optional_positive_int(option_value(runtime_options, "lingbot_overlap_keyframes", "overlap_keyframes"))
    if overlap_keyframes is None and offline_video:
        overlap_keyframes = 8
    kv_cache_sliding_window = read_int(option_value(runtime_options, "lingbot_kv_cache_sliding_window", "kv_cache_sliding_window"), 48 if low_vram else 64)
    camera_num_iterations = read_int(option_value(runtime_options, "lingbot_camera_num_iterations", "camera_num_iterations"), 1 if realtime or offline_video else 4)
    keyframes_only_points = parse_bool(option_value(runtime_options, "lingbot_keyframes_only_points", "keyframes_only_points"), realtime or offline_video)

    profile = "realtime" if realtime else "fast_video" if offline_video else "low_vram" if low_vram else "official"
    return LingBotPlan(
        input_mode=input_mode,
        mode=mode,
        backend=backend,
        use_sdpa=backend == "sdpa",
        compile=compile_enabled,
        offload_to_cpu=offload_to_cpu,
        dtype_name=dtype_name,
        num_scale_frames=num_scale_frames,
        keyframe_interval=keyframe_interval,
        kv_cache_sliding_window=kv_cache_sliding_window,
        window_size=window_size,
        overlap_size=overlap_size,
        overlap_keyframes=overlap_keyframes if mode == "windowed" else None,
        window_profile=window_profile if offline_video else profile,
        camera_num_iterations=camera_num_iterations,
        keyframes_only_points=keyframes_only_points,
        max_frame_num=read_int(runtime_options.get("lingbot_max_frame_num"), 1024),
        frame_budget=max(frame_budget, 0),
        original_frame_count=original_frame_count,
        selected_frame_count=frame_count,
        gpu_memory_total_mb=total_mb,
        gpu_memory_free_mb=free_mb,
        profile=profile,
    )


def downgrade_plan(plan: LingBotPlan) -> LingBotPlan:
    return replace(
        plan,
        compile=False,
        keyframe_interval=max(plan.keyframe_interval, math.ceil(plan.selected_frame_count / 160)),
        kv_cache_sliding_window=max(24, min(plan.kv_cache_sliding_window, 32)),
        window_size=max(24, min(plan.window_size, 32)),
        overlap_size=max(plan.num_scale_frames, min(plan.overlap_size, 8)),
        camera_num_iterations=1,
        profile=f"{plan.profile}_retry_lowmem",
        retry_level=1,
    )


def warmup_streaming_compile(torch, model, images, plan: LingBotPlan, dtype, device, progress: Progress) -> None:
    scale_frames = min(plan.num_scale_frames, int(images.shape[0]))
    if scale_frames >= int(images.shape[0]):
        scale_frames = max(1, int(images.shape[0]) - 1)
    warm_stream_n = min(10, max(1, int(images.shape[0]) - scale_frames))
    warm_count = scale_frames + warm_stream_n
    warm_images = images[:warm_count].to(device, non_blocking=True)
    progress("lingbot_compile_warmup", 40, "warming LingBot streaming graph before compile")
    warm_streaming(torch, model, warm_images, scale_frames, warm_stream_n, dtype, passes=1, keyframe_interval=plan.keyframe_interval)
    progress("lingbot_compile", 42, "compiling LingBot hot modules")
    compile_model(torch, model)
    progress("lingbot_compile_warmup", 44, "warming compiled LingBot graph")
    warm_streaming(torch, model, warm_images, scale_frames, warm_stream_n, dtype, passes=3, keyframe_interval=plan.keyframe_interval)
    del warm_images
    cleanup_cuda(torch)


def should_compile_lingbot_plan(plan: LingBotPlan) -> bool:
    return bool(plan.compile and plan.selected_frame_count > 500)


def compile_model(torch, model) -> None:
    agg = model.aggregator
    for index, block in enumerate(agg.frame_blocks):
        agg.frame_blocks[index] = torch.compile(block, mode="reduce-overhead")
    for index, block in enumerate(agg.patch_embed.blocks):
        agg.patch_embed.blocks[index] = torch.compile(block, mode="reduce-overhead")
    for block in agg.global_blocks:
        if hasattr(block, "attn_pre"):
            block.attn_pre = torch.compile(block.attn_pre, mode="reduce-overhead")
        if hasattr(block, "ffn_residual"):
            block.ffn_residual = torch.compile(block.ffn_residual, mode="reduce-overhead")
        block.attn.proj = torch.compile(block.attn.proj, mode="reduce-overhead")


def warm_streaming(
    torch,
    model,
    images,
    scale_frames: int,
    warm_stream_n: int,
    dtype,
    *,
    passes: int,
    keyframe_interval: int,
) -> None:
    num_avail = int(images.shape[0])
    scale_frames = max(1, min(int(scale_frames), num_avail))
    if scale_frames >= num_avail:
        scale_frames = max(1, num_avail - 1)
    warm_stream_n = max(1, min(int(warm_stream_n), num_avail - scale_frames))
    warm_scale = images[:scale_frames].unsqueeze(0).to(dtype)
    warm_stream = images[scale_frames : scale_frames + warm_stream_n].unsqueeze(0).to(dtype)
    for _ in range(passes):
        model.clean_kv_cache()
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            model.forward(
                warm_scale,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=scale_frames,
                causal_inference=True,
            )
        for index in range(warm_stream_n):
            is_keyframe = keyframe_interval <= 1 or index % keyframe_interval == 0
            if not is_keyframe:
                model._set_skip_append(True)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                model.forward(
                    warm_stream[:, index : index + 1],
                    num_frame_for_scale=scale_frames,
                    num_frame_per_block=1,
                    causal_inference=True,
                )
            if not is_keyframe:
                model._set_skip_append(False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model.clean_kv_cache()


def extract_frames(video_path: Path, frame_dir: Path, *, fps: int) -> FrameSelectionResult:
    frame_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PreviewFailure("VIDEO_OPEN_FAILED", f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval = max(1, round(src_fps / max(1, fps)))
    saved = 0
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % interval == 0:
            cv2.imwrite(str(frame_dir / f"{saved:06d}.jpg"), frame)
            saved += 1
        index += 1
    cap.release()
    return FrameSelectionResult(
        frame_dir=frame_dir,
        selected_count=saved,
        source_frame_count=source_frame_count or index,
        source_fps=float(src_fps),
        mode="fixed_fps",
        reasons={"fixed_fps": saved},
    )


def extract_scene_keyframes(video_path: Path, frame_dir: Path, *, runtime_options: dict[str, Any]) -> FrameSelectionResult:
    frame_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PreviewFailure("VIDEO_OPEN_FAILED", f"cannot open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    min_gap = max(1, int(round(src_fps * read_float(runtime_options.get("lingbot_scene_min_interval_seconds"), 0.25))))
    max_gap = max(min_gap, int(round(src_fps * read_float(runtime_options.get("lingbot_scene_max_interval_seconds"), 1.25))))
    min_blur = read_float(runtime_options.get("lingbot_scene_min_blur"), 8.0)
    threshold_option = runtime_options.get("lingbot_scene_change_threshold")
    fixed_threshold = read_float(threshold_option, 0.0) if threshold_option not in (None, "") else None

    saved = 0
    index = 0
    last_selected = -max_gap
    last_frame: np.ndarray | None = None
    last_index = -1
    previous_feature: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    scores: list[float] = []
    reasons: dict[str, int] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last_frame = frame
        last_index = index
        feature = scene_feature(frame)
        blur = laplacian_variance(feature[0])
        score = scene_change_score(previous_feature, feature)
        if previous_feature is not None:
            scores.append(score)

        reason: str | None = None
        if index == 0:
            reason = "first_frame"
        elif index - last_selected >= min_gap:
            threshold = fixed_threshold if fixed_threshold is not None else adaptive_scene_threshold(scores)
            if score >= threshold and blur >= min_blur:
                reason = "visual_change"
            elif index - last_selected >= max_gap and blur >= min_blur * 0.5:
                reason = "time_guard"

        if reason:
            saved = save_selected_frame(frame_dir, frame, saved)
            reasons[reason] = reasons.get(reason, 0) + 1
            last_selected = index

        previous_feature = feature
        index += 1

    cap.release()
    if last_frame is not None and last_index != last_selected:
        saved = save_selected_frame(frame_dir, last_frame, saved)
        reasons["last_frame"] = reasons.get("last_frame", 0) + 1

    return FrameSelectionResult(
        frame_dir=frame_dir,
        selected_count=saved,
        source_frame_count=source_frame_count or index,
        source_fps=src_fps,
        mode="scene_keyframes",
        reasons=reasons,
    )


def save_selected_frame(frame_dir: Path, frame: np.ndarray, saved: int) -> int:
    cv2.imwrite(str(frame_dir / f"{saved:06d}.jpg"), frame)
    return saved + 1


def scene_feature(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thumb = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    edges = cv2.Canny(gray, 80, 160)
    return gray, hist.reshape(-1), edges


def scene_change_score(
    previous: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    current: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    if previous is None:
        return 0.0
    gray_prev, hist_prev, edges_prev = previous
    gray, hist, edges = current
    gray_diff = float(np.mean(cv2.absdiff(gray_prev, gray)) / 255.0)
    hist_diff = float(cv2.compareHist(hist_prev.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))
    edge_diff = float(np.mean(cv2.absdiff(edges_prev, edges)) / 255.0)
    return gray_diff * 0.5 + hist_diff * 0.35 + edge_diff * 0.15


def adaptive_scene_threshold(scores: list[float]) -> float:
    if len(scores) < 12:
        return 0.10
    recent = np.asarray(scores[-120:], dtype=np.float32)
    return float(max(0.055, min(0.22, recent.mean() + recent.std() * 1.25)))


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_frame_files(files: list[Path], max_frames: int) -> list[Path]:
    if max_frames <= 0 or len(files) <= max_frames:
        return files
    indexes = np.linspace(0, len(files) - 1, num=max_frames, dtype=int)
    return [files[int(index)] for index in indexes]


def cuda_memory_mb(torch) -> tuple[int, int]:
    try:
        free, total = torch.cuda.mem_get_info()
        return int(total / 1024 / 1024), int(free / 1024 / 1024)
    except Exception:
        props = torch.cuda.get_device_properties(0)
        return int(props.total_memory / 1024 / 1024), 0


def resolve_window_profile(total_mb: int, profile: str) -> tuple[str, int, int]:
    if profile == "low_vram":
        return profile, 32, 12
    if profile == "balanced":
        return profile, 48, 16
    if profile == "large_vram":
        return profile, 96, 24
    if total_mb and total_mb < 10_000:
        return "auto_low_vram", 32, 12
    if total_mb and total_mb < 14_000:
        return "auto_balanced", 48, 16
    if total_mb and total_mb < 20_000:
        return "auto_mid_vram", 64, 20
    return "auto_large_vram", 96, 24


def dtype_from_plan(torch, plan: LingBotPlan):
    return torch.bfloat16 if plan.dtype_name == "bf16" else torch.float16 if plan.dtype_name == "fp16" else torch.float32


def resolve_dtype_name(torch, value: str) -> str:
    if value in {"bf16", "bfloat16"}:
        return "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    if value in {"fp16", "float16"}:
        return "fp16"
    if value in {"fp32", "float32"}:
        return "fp32"
    return "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"


def flashinfer_available() -> bool:
    try:
        __import__("flashinfer")
        return True
    except Exception:
        return False


def configure_torch_compile_cache(runtime_options: dict[str, Any]) -> None:
    cache_dir = option_value(runtime_options, "lingbot_compile_cache_dir", "compile_cache_dir")
    if cache_dir:
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
        os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")


def synchronize_cuda(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def inference_throughput_metrics(processed_frames: int, seconds: float) -> dict[str, Any]:
    elapsed = max(0.0, float(seconds))
    fps = processed_frames / max(elapsed, 1e-6) if processed_frames > 0 else 0.0
    return {
        "processed_frames": int(processed_frames),
        "inference_seconds": round(elapsed, 3),
        "inference_fps": round(fps, 3),
        "lingbot_processed_frames": int(processed_frames),
        "lingbot_inference_seconds": round(elapsed, 3),
        "lingbot_inference_fps": round(fps, 3),
    }


def cleanup_cuda(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cuda_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "cuda out of memory" in message or "cublas_status_alloc_failed" in message


def squeeze_batch(value):
    if value is None:
        raise PreviewFailure("LINGBOT_OUTPUT_MISSING", "LingBot output is missing world point confidence")
    if getattr(value, "ndim", 0) >= 5 and value.shape[0] == 1:
        return value[0]
    return value


def prepare_lingbot_point_export(predictions: dict[str, Any], fallback_images: Any) -> PointExportData:
    world_points = numpy_tensor(predictions.get("world_points"))
    confidence = numpy_tensor(predictions.get("world_points_conf"))
    images = numpy_tensor(predictions.get("images", fallback_images))
    is_keyframe = keyframe_mask_from_predictions(predictions)

    world_points = remove_leading_batch(world_points, "world_points")
    confidence = remove_leading_batch(confidence, "world_points_conf")
    images = remove_leading_batch(images, "images")
    if is_keyframe is not None:
        is_keyframe = remove_leading_batch(is_keyframe, "is_keyframe").astype(bool, copy=False).reshape(-1)

    if world_points.ndim != 4 or world_points.shape[-1] != 3:
        raise export_shape_error(world_points, confidence, images, "world_points must have shape [S,H,W,3]")
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    if confidence.ndim != 3:
        raise export_shape_error(world_points, confidence, images, "world_points_conf must have shape [S,H,W]")

    colors = image_colors(images, world_points.shape)
    if world_points.shape[:3] != confidence.shape:
        raise export_shape_error(world_points, confidence, colors, "world_points and confidence frame/size dimensions differ")
    if world_points.shape[:3] != colors.shape[:3]:
        raise export_shape_error(world_points, confidence, colors, "world_points and colors frame/size dimensions differ")

    return PointExportData(
        world_points=world_points.astype(np.float32, copy=False),
        confidence=confidence.astype(np.float32, copy=False),
        colors=colors,
        is_keyframe=is_keyframe if is_keyframe is not None and is_keyframe.shape[0] == world_points.shape[0] else None,
        metrics={
            "lingbot_export_world_points_shape": list(world_points.shape),
            "lingbot_export_conf_shape": list(confidence.shape),
            "lingbot_export_colors_shape": list(colors.shape),
            "lingbot_point_export_frame_count": int(world_points.shape[0]),
            "lingbot_point_export_keyframe_count": int(is_keyframe.sum()) if is_keyframe is not None and is_keyframe.shape[0] == world_points.shape[0] else None,
            "lingbot_point_export_has_keyframe_mask": bool(is_keyframe is not None and is_keyframe.shape[0] == world_points.shape[0]),
        },
    )


def filter_keyframe_point_export(export: PointExportData, *, enabled: bool) -> PointExportData:
    if not enabled:
        return PointExportData(
            world_points=export.world_points,
            confidence=export.confidence,
            colors=export.colors,
            is_keyframe=export.is_keyframe,
            metrics={**export.metrics, "lingbot_keyframes_only_points_applied": False},
        )
    if export.is_keyframe is None or export.is_keyframe.shape[0] != export.world_points.shape[0]:
        return PointExportData(
            world_points=export.world_points,
            confidence=export.confidence,
            colors=export.colors,
            is_keyframe=export.is_keyframe,
            metrics={**export.metrics, "lingbot_keyframes_only_points_applied": False, "lingbot_keyframes_only_points_fallback": "missing_keyframe_mask"},
        )
    mask = export.is_keyframe.astype(bool, copy=False)
    if not bool(mask.any()):
        return PointExportData(
            world_points=export.world_points,
            confidence=export.confidence,
            colors=export.colors,
            is_keyframe=export.is_keyframe,
            metrics={**export.metrics, "lingbot_keyframes_only_points_applied": False, "lingbot_keyframes_only_points_fallback": "empty_keyframe_mask"},
        )
    return PointExportData(
        world_points=export.world_points[mask],
        confidence=export.confidence[mask],
        colors=export.colors[mask],
        is_keyframe=export.is_keyframe[mask],
        metrics={
            **export.metrics,
            "lingbot_keyframes_only_points_applied": True,
            "lingbot_point_export_frame_count": int(export.world_points.shape[0]),
            "lingbot_point_export_keyframe_count": int(mask.sum()),
            "lingbot_point_export_used_frame_count": int(mask.sum()),
        },
    )


def keyframe_mask_from_predictions(predictions: dict[str, Any]) -> np.ndarray | None:
    if predictions.get("is_keyframe") is not None:
        return numpy_tensor(predictions["is_keyframe"]).astype(bool, copy=False)
    if predictions.get("frame_type") is not None:
        return numpy_tensor(predictions["frame_type"]) != 2
    return None


def numpy_tensor(value: Any) -> np.ndarray:
    if value is None:
        raise PreviewFailure("LINGBOT_OUTPUT_MISSING", "LingBot output is missing world points or confidence")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def remove_leading_batch(value: np.ndarray, name: str) -> np.ndarray:
    if name in {"world_points", "images"} and value.ndim == 5 and value.shape[0] == 1:
        return value[0]
    if name == "world_points_conf" and value.ndim == 4 and value.shape[0] == 1 and value.shape[-1] != 1:
        return value[0]
    if name == "is_keyframe" and value.ndim >= 2 and value.shape[0] == 1:
        return value[0]
    return value


def image_colors(images: np.ndarray, world_shape: tuple[int, ...]) -> np.ndarray:
    if images.ndim == 4 and images.shape[1] == 3:
        colors = np.transpose(images, (0, 2, 3, 1))
    elif images.ndim == 4 and images.shape[-1] == 3:
        colors = images
    else:
        raise export_shape_error(np.empty(world_shape), np.empty(world_shape[:3]), images, "images must have shape [S,3,H,W] or [S,H,W,3]")

    colors = np.clip(colors * 255.0 if colors.dtype.kind == "f" and float(np.nanmax(colors)) <= 1.5 else colors, 0, 255).astype(np.uint8)
    target_h, target_w = int(world_shape[1]), int(world_shape[2])
    if colors.shape[1:3] == (target_h, target_w):
        return colors
    resized = np.empty((colors.shape[0], target_h, target_w, 3), dtype=np.uint8)
    for index, frame in enumerate(colors):
        resized[index] = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized


def export_shape_error(world_points: np.ndarray, confidence: np.ndarray, colors: np.ndarray, reason: str) -> PreviewFailure:
    return PreviewFailure(
        "LINGBOT_EXPORT_SHAPE_MISMATCH",
        (
            f"{reason}; world_points={list(getattr(world_points, 'shape', []))}, "
            f"confidence={list(getattr(confidence, 'shape', []))}, colors={list(getattr(colors, 'shape', []))}"
        ),
    )


def normalized_option(value: Any) -> str:
    return str(value or "").strip().lower()


def parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def option_value(options: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = options.get(name)
        if value is not None and value != "":
            return value
    return None


def read_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def read_optional_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def read_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def inference_percent(current: int, total: int) -> int:
    if total <= 0:
        return 48
    return max(48, min(72, 48 + int(round(24 * current / total))))


def format_eta_seconds(seconds: int) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
