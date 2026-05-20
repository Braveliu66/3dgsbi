from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fine.dash_deblur_group import (  # noqa: E402
    DashDeblurGroupPaths,
    build_blur_labels,
    build_training_command,
    build_training_config,
    config_uses_gsplat,
    detect_trainer_flavor,
    gsplat_kernels_are_precompiled,
    locate_final_ply,
    parse_iteration,
    prewarm_gsplat_kernels,
    resolve_runtime_paths,
    resolve_effective_deblur_mode,
    run_dash_deblur_group_training,
    write_training_config,
)
from app.fine.runner import remap_blur_registry_to_scene_images  # noqa: E402
from app.fine.preprocess import normalize_detector_blur_type  # noqa: E402


class DashDeblurGroupRuntimeTests(unittest.TestCase):
    def test_build_training_config_uses_scene_and_deblur_preset(self) -> None:
        config = build_training_config(
            {
                "scene_type": "outdoor",
                "fine_deblur_mode": "defocus",
                "fine_iterations": 1234,
            }
        )

        self.assertEqual(config["iterations"], 1234)
        self.assertEqual(config["resolution"], -1)
        self.assertEqual(config["deblur"], 1)
        self.assertEqual(config["use_pos"], 0)
        self.assertEqual(config["densify_with_depth"], 1)
        self.assertEqual(config["lambda_p"], 0.01)
        self.assertEqual(config["densify_until_iter"], 740)
        self.assertEqual(config["densification_interval"], 100)
        self.assertEqual(config["densify_grad_threshold"], 0.0002)
        self.assertEqual(config["pts_iter"], 999999)
        self.assertEqual(config["pts_rate"], 0.0)
        self.assertEqual(config["pts_N_pts"], 0)
        self.assertEqual(config["pc_name"], "points3D_eap")
        self.assertEqual(config["renderer_backend"], "original")
        self.assertEqual(config["renderer_backend_deblur"], "original")

    def test_eap_switch_selects_enhanced_initial_pointcloud(self) -> None:
        self.assertEqual(build_training_config({})["pc_name"], "points3D_eap")
        self.assertEqual(build_training_config({"fine_eap_enabled": False})["pc_name"], "points3D")
        self.assertEqual(build_training_config({"fine_eap_enabled": True})["pc_name"], "points3D_eap")

    def test_gsplat_switch_only_changes_sharp_renderer_backend(self) -> None:
        config = build_training_config({"fine_gsplat_enabled": True})

        self.assertEqual(config["renderer_backend"], "gsplat")
        self.assertEqual(config["renderer_backend_deblur"], "original")
        self.assertTrue(config_uses_gsplat(config))

    def test_gsplat_prewarm_runs_tiny_cuda_rasterization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = DashDeblurGroupPaths(repo_dir=root, train_py=root / "train.py", python="python")
            events: list[tuple[str, int, str]] = []

            with patch("app.fine.dash_deblur_group.subprocess.run") as run:
                run.return_value = SimpleNamespace(returncode=0, stdout="gsplat CUDA kernels ready\n")
                prewarm_gsplat_kernels(paths, progress=lambda stage, value, message: events.append((stage, value, message)))

        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[:3], ["python", "-u", "-c"])
        self.assertIn("from gsplat import rasterization", command[3])
        self.assertIn("torch.cuda.synchronize()", command[3])
        self.assertIn(("fine_training_preflight", 42, "precompiling gsplat CUDA kernels"), events)
        self.assertIn(("fine_training_preflight", 42, "gsplat kernels ready"), events)

    def test_gsplat_prewarm_skips_when_baked_extensions_are_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            extension_dir = Path(tmp) / "torch_extensions"
            extension_dir.mkdir()
            marker = extension_dir / ".gsplat_precompiled"
            marker.write_text("ready\n", encoding="utf-8")
            paths = DashDeblurGroupPaths(repo_dir=Path(tmp), train_py=Path(tmp) / "train.py", python="python")
            events: list[tuple[str, int, str]] = []

            with (
                patch.dict("os.environ", {"TORCH_EXTENSIONS_DIR": str(extension_dir), "GSPLAT_PRECOMPILED_MARKER": str(marker)}),
                patch("app.fine.dash_deblur_group.subprocess.run") as run,
            ):
                self.assertTrue(gsplat_kernels_are_precompiled())
                prewarm_gsplat_kernels(paths, progress=lambda stage, value, message: events.append((stage, value, message)))

        run.assert_not_called()
        self.assertEqual(events, [("fine_training_preflight", 42, "gsplat kernels ready")])

    def test_legacy_mix_maps_to_motion_even_with_defocus_analysis(self) -> None:
        blur = SimpleNamespace(
            per_frame_blur={
                "000000.jpg": {"rejected": False, "blurred": True, "kind": "defocus"},
                "000001.jpg": {"rejected": False, "blurred": True, "kind": "defocus"},
            }
        )

        config = build_training_config(
            {
                "scene_type": "indoor",
                "fine_deblur_mode": "mix",
                "use_pos": True,
                "num_moments": 4,
                "hidden": 3,
                "lambda_p": 0.01,
                "densify_with_depth": True,
            },
            blur_analysis=blur,
        )

        self.assertEqual(config["deblur"], 1)
        self.assertEqual(config["use_pos"], 0)
        self.assertEqual(config["num_moments"], 4)
        self.assertEqual(config["hidden"], 3)
        self.assertEqual(config["lambda_p"], 0.01)
        self.assertEqual(config["densify_with_depth"], 1)

    def test_legacy_mix_maps_to_motion_after_stale_saved_options(self) -> None:
        blur = SimpleNamespace(
            per_frame_blur={
                "000000.jpg": {"rejected": False, "blurred": True, "kind": "motion"},
                "000001.jpg": {"rejected": False, "blurred": True, "kind": "motion"},
            }
        )

        config = build_training_config(
            {
                "scene_type": "indoor",
                "fine_deblur_mode": "mix",
                "use_pos": False,
                "num_moments": 3,
                "lambda_p": 0.0,
            },
            blur_analysis=blur,
        )

        self.assertEqual(config["deblur"], 1)
        self.assertEqual(config["use_pos"], 1)
        self.assertEqual(config["num_moments"], 4)
        self.assertEqual(config["lambda_p"], 0.01)

    def test_mix_deblur_mode_uses_motion_without_auto_vote(self) -> None:
        blur = SimpleNamespace(
            per_frame_blur={
                "000000.jpg": {"rejected": False, "blurred": True, "kind": "motion"},
                "000001.jpg": {"rejected": False, "blurred": True, "kind": "motion"},
                "000002.jpg": {"rejected": False, "blurred": False, "kind": "sharp"},
            }
        )

        mode = resolve_effective_deblur_mode({"fine_deblur_mode": "mix"}, blur)
        config = build_training_config({"scene_type": "indoor", "fine_deblur_mode": "mix"}, blur_analysis=blur)

        self.assertEqual(mode.effective, "motion")
        self.assertEqual(mode.confidence, "explicit")
        self.assertEqual(mode.reason, "per_image_blur_labels")
        self.assertEqual(mode.motion_frames, 2)
        self.assertEqual(config["deblur"], 1)

    def test_mix_deblur_mode_ignores_defocus_vote(self) -> None:
        blur = SimpleNamespace(
            per_frame_blur={
                "000000.jpg": {"rejected": False, "blurred": True, "kind": "defocus"},
                "000001.jpg": {"rejected": False, "blurred": True, "kind": "defocus"},
                "000002.jpg": {"rejected": True, "blurred": True, "kind": "motion"},
            }
        )

        mode = resolve_effective_deblur_mode({"scene_type": "outdoor", "fine_deblur_mode": "mix"}, blur)
        config = build_training_config({"scene_type": "outdoor", "fine_deblur_mode": "mix"}, blur_analysis=blur)

        self.assertEqual(mode.effective, "defocus")
        self.assertEqual(mode.defocus_frames, 2)
        self.assertEqual(config["deblur"], 1)
        self.assertEqual(config["use_pos"], 0)
        self.assertEqual(config["num_moments"], 4)

    def test_mix_deblur_mode_ignores_auto_override_when_uncertain(self) -> None:
        blur = SimpleNamespace(
            per_frame_blur={
                "000000.jpg": {"rejected": False, "blurred": True, "kind": "motion"},
                "000001.jpg": {"rejected": False, "blurred": True, "kind": "defocus"},
                "000002.jpg": {"rejected": False, "blurred": True, "kind": "mixed"},
            }
        )

        mode = resolve_effective_deblur_mode({"scene_type": "outdoor", "fine_deblur_mode": "mix"}, blur)

        self.assertEqual(mode.effective, "motion")
        self.assertEqual(mode.confidence, "explicit")
        self.assertEqual(mode.reason, "per_image_blur_labels")

    def test_explicit_deblur_mode_overrides_blur_analysis(self) -> None:
        blur = SimpleNamespace(
            per_frame_blur={
                "000000.jpg": {"rejected": False, "blurred": True, "kind": "defocus"},
                "000001.jpg": {"rejected": False, "blurred": True, "kind": "defocus"},
            }
        )

        mode = resolve_effective_deblur_mode({"fine_deblur_mode": "motion"}, blur)
        config = build_training_config({"fine_deblur_mode": "motion"}, blur_analysis=blur)

        self.assertEqual(mode.effective, "defocus")
        self.assertEqual(mode.confidence, "explicit")
        self.assertEqual(config["deblur"], 1)

    def test_requested_legacy_mix_key_maps_to_motion(self) -> None:
        blur = SimpleNamespace(
            per_frame_blur={
                "000000.jpg": {"rejected": False, "blurred": True, "kind": "defocus"},
                "000001.jpg": {"rejected": False, "blurred": True, "kind": "defocus"},
            }
        )

        mode = resolve_effective_deblur_mode({"fine_deblur_mode_requested": "mix"}, blur)
        config = build_training_config({"scene_type": "indoor", "fine_deblur_mode_requested": "mix"}, blur_analysis=blur)

        self.assertEqual(mode.effective, "defocus")
        self.assertEqual(config["deblur"], 1)
        self.assertEqual(config["use_pos"], 0)

    def test_build_blur_labels_normalizes_to_three_training_labels(self) -> None:
        blur = SimpleNamespace(
            per_frame_blur={
                "000000.jpg": {"rejected": False, "training_image": "000000.jpg", "blurred": False, "kind": "sharp"},
                "000001.jpg": {"rejected": False, "training_image": "000001.jpg", "blurred": True, "kind": "motion"},
                "000002.jpg": {"rejected": False, "training_image": "000002.jpg", "blurred": True, "kind": "defocus"},
                "000003.jpg": {"rejected": False, "training_image": "000003.jpg", "blurred": True, "kind": "blur_unknown"},
                "000004.jpg": {"rejected": False, "training_image": "000004.jpg", "blurred": True, "kind": "motion", "detector_label": "sharp", "detector_blur_type": "none"},
                "000005.jpg": {"rejected": False, "training_image": "000005.jpg", "blurred": True, "kind": "motion", "detector_label": "blurry", "detector_blur_type": "motion_blur", "detector_normalized_blur_type": "sharp"},
                "rejected:old.jpg": {"rejected": True, "training_image": None, "blurred": True, "kind": "defocus"},
            }
        )

        self.assertEqual(
            build_blur_labels(blur),
            {
                "000000.jpg": "sharp",
                "000001.jpg": "motion",
                "000002.jpg": "defocus",
                "000003.jpg": "motion",
                "000004.jpg": "sharp",
                "000005.jpg": "motion",
            },
        )

    def test_detector_raw_label_is_trusted_before_normalized_label(self) -> None:
        self.assertEqual(
            normalize_detector_blur_type({"label": "sharp", "blur_type": "none", "normalized_blur_type": "motion"}),
            "sharp",
        )

    def test_all_sharp_blur_labels_disable_deblur_config(self) -> None:
        blur = SimpleNamespace(
            per_frame_blur={
                "000000.jpg": {"rejected": False, "training_image": "000000.jpg", "blurred": False, "kind": "sharp"},
                "000001.jpg": {"rejected": False, "training_image": "000001.jpg", "blurred": False, "kind": "sharp"},
            }
        )

        config = build_training_config({"fine_deblur_mode": "motion"}, blur_analysis=blur)

        self.assertEqual(config["deblur"], 0)
        self.assertEqual(config["use_pos"], 0)

    def test_blur_registry_remaps_to_colmap_scene_image_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fine_input = root / "fine_input"
            scene_dir = root / "fine_scene" / "colmap"
            images_dir = scene_dir / "images"
            fine_input.mkdir(parents=True)
            images_dir.mkdir(parents=True)
            for name in ("000000.jpg", "000001.jpg"):
                (fine_input / name).write_bytes(b"image")
            (images_dir / "001_000000.jpg").write_bytes(b"image")
            (images_dir / "002_000001.jpg").write_bytes(b"image")
            blur = SimpleNamespace(
                per_frame_blur={
                    "000000.jpg": {"rejected": False, "training_image": "000000.jpg", "blurred": False, "kind": "sharp"},
                    "000001.jpg": {"rejected": False, "training_image": "000001.jpg", "blurred": True, "kind": "motion"},
                }
            )

            remap_blur_registry_to_scene_images(blur, fine_input, scene_dir)

        self.assertEqual(sorted(blur.per_frame_blur), ["001_000000.jpg", "002_000001.jpg"])
        self.assertEqual(blur.per_frame_blur["002_000001.jpg"]["training_image"], "002_000001.jpg")
        self.assertEqual(build_blur_labels(blur)["002_000001.jpg"], "motion")

    def test_write_training_config_preserves_deblur_keys(self) -> None:
        config = build_training_config({"scene_type": "indoor", "fine_deblur_mode": "motion"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.txt"
            write_training_config(path, config)

            text = path.read_text(encoding="utf-8")

        self.assertIn("deblur = 1", text)
        self.assertIn("resolution = -1", text)
        self.assertIn("position_lr_final = 1.6e-05", text)
        self.assertIn("percent_dense = 0.01", text)
        self.assertIn("lambda_dssim = 0.2", text)
        self.assertIn("densify_with_depth = 1", text)
        self.assertIn("densify_grad_threshold = 0.0005", text)
        self.assertIn("densify_prune_threshold = 0.01", text)
        self.assertIn("pts_N_pts = 0", text)
        self.assertIn("pts_iter = 999999", text)
        self.assertIn("pts_rate = 0.0", text)
        self.assertIn("blur_code_dim = 4", text)
        self.assertIn("pre_deblur_warmup_enable = True", text)
        self.assertIn("pre_deblur_warmup_iters = 500", text)
        self.assertIn("luminance_enable = True", text)
        self.assertIn("gdags_stats_enable = True", text)
        self.assertIn("pc_name = points3D_eap", text)
        self.assertIn("renderer_backend = original", text)
        self.assertIn("renderer_backend_deblur = original", text)
        self.assertNotIn("dash_enable", text)
        self.assertNotIn("Grouping", text)
        self.assertNotIn("protect_new_points_iters", text)

    def test_trainer_uses_official_densify_order_without_default_caps(self) -> None:
        source = (Path(__file__).resolve().parents[2] / "worker" / "trainer" / "dash_deblur_group_gs" / "scene" / "gaussian_model.py").read_text(encoding="utf-8")
        densify_source = source.split("def densify_and_prune", 1)[1].split("def add_densification_stats", 1)[0]

        self.assertLess(densify_source.index("self.densify_and_clone"), densify_source.index("self.densify_and_split"))
        self.assertLess(densify_source.index("self.densify_and_split"), densify_source.index("prune_mask ="))
        self.assertNotIn("opacity > 0.015", source)
        self.assertNotIn("seen >=", source)

    def test_trainer_uses_braveliu_auto_point_schedule(self) -> None:
        trainer_root = Path(__file__).resolve().parents[2] / "worker" / "trainer" / "dash_deblur_group_gs"
        train_source = (trainer_root / "train.py").read_text(encoding="utf-8")
        gaussian_source = (trainer_root / "scene" / "gaussian_model.py").read_text(encoding="utf-8")

        self.assertIn("def auto_point_addition_iter", train_source)
        self.assertIn("def auto_densify_until_iter", train_source)
        self.assertIn("def random_point_addition_enabled", train_source)
        self.assertIn("warmup_active", train_source)
        self.assertIn("densify_warmup_clone_split", train_source)
        self.assertIn("opt.pts_iter = auto_pts_iter", train_source)
        self.assertIn("pts_N_pts = int(min(volume / (opt.pts_rate ** 3), 200000))", train_source)
        self.assertNotIn("def resolve_add_points_count", train_source)
        self.assertNotIn("kept={add_stats['kept']} rejected={add_stats['rejected']}", train_source)
        self.assertNotIn("torch.cdist", gaussian_source)

    def test_trainer_uses_sharp_images_for_per_image_blur_eval_and_configurable_codes(self) -> None:
        trainer_root = Path(__file__).resolve().parents[2] / "worker" / "trainer" / "dash_deblur_group_gs"
        train_source = (trainer_root / "train.py").read_text(encoding="utf-8")
        blur_kernel_source = (trainer_root / "scene" / "blur_kernel.py").read_text(encoding="utf-8")

        self.assertIn("sharp_camera_subset(scene.getTestCameras())", train_source)
        self.assertIn("sharp_camera_subset(scene.getTrainCameras())[:5]", train_source)
        self.assertIn("code_dim=opt.blur_code_dim", train_source)
        self.assertIn("nn.Embedding(num_images, blur_code_dim)", blur_kernel_source)

    def test_blur_code_dim_can_be_set_to_four_eight_or_sixteen(self) -> None:
        self.assertEqual(build_training_config({"blur_code_dim": 4})["blur_code_dim"], 4)
        self.assertEqual(build_training_config({"blur_code_dim": 8})["blur_code_dim"], 8)
        self.assertEqual(build_training_config({"blur_code_dim": 16})["blur_code_dim"], 16)

    def test_trainer_accepts_eap_pointcloud_and_lazy_gsplat_backend(self) -> None:
        trainer_root = Path(__file__).resolve().parents[2] / "worker" / "trainer" / "dash_deblur_group_gs"
        args_source = (trainer_root / "arguments" / "__init__.py").read_text(encoding="utf-8")
        scene_source = (trainer_root / "scene" / "__init__.py").read_text(encoding="utf-8")
        train_source = (trainer_root / "train.py").read_text(encoding="utf-8")
        renderer_source = (trainer_root / "gaussian_renderer" / "__init__.py").read_text(encoding="utf-8")
        backend_source = (trainer_root / "gaussian_renderer" / "backends" / "gsplat_backend.py").read_text(encoding="utf-8")

        self.assertIn('self.pc_name = "points3D"', args_source)
        self.assertIn('self.renderer_backend = "original"', args_source)
        self.assertIn('self.renderer_backend_deblur = "original"', args_source)
        self.assertIn('getattr(args, "pc_name", "points3D")', scene_source)
        self.assertIn("force_original_backend=(iteration < opt.densify_until_iter)", train_source)
        self.assertIn('not force_original_backend and getattr(pipe, "renderer_backend", "original") == "gsplat"', renderer_source)
        self.assertIn("from gaussian_renderer.backends.gsplat_backend import gsplat_rasterize", renderer_source)
        self.assertNotIn("from gsplat", backend_source.split("def gsplat_rasterize", 1)[0])
        self.assertIn("from gsplat import rasterization", backend_source.split("def gsplat_rasterize", 1)[1])

    def test_build_training_command_uses_colmap_scene_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = DashDeblurGroupPaths(
                repo_dir=root / "repo",
                train_py=root / "repo" / "train.py",
                python="python",
            )
            scene_dir = root / "scene"
            output_dir = root / "work" / "model"
            config_path = root / "work" / "config.txt"

            command = build_training_command(
                paths=paths,
                scene_dir=scene_dir,
                output_dir=output_dir,
                config_path=config_path,
                expname="exp",
                config=build_training_config({}),
            )

        self.assertEqual(command[:3], ["python", "-u", str(paths.train_py)])
        self.assertIn("-s", command)
        self.assertIn(str(scene_dir), command)
        self.assertIn("--model_path", command)
        self.assertIn(str(output_dir), command)
        self.assertIn("--config", command)
        self.assertIn("--test_iterations", command)
        self.assertIn("3001", command)

    def test_training_exports_filtered_final_ply(self) -> None:
        try:
            import numpy  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest(f"viewer filter dependency unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "train.py").write_text("parser.add_argument('--config')\ndeblur = 1\n", encoding="utf-8")
            produced = root / "produced.ply"
            self._write_xyz_binary_ply(produced, [(float(index % 12), float(index // 12), 0.0) for index in range(120)] + [(1000.0, 1000.0, 1000.0)])
            final_ply = root / "final.ply"

            with (
                patch("app.fine.dash_deblur_group.run_training_process") as run_training,
                patch("app.fine.dash_deblur_group.locate_final_ply", return_value=produced),
            ):
                result = run_dash_deblur_group_training(
                    scene_dir=root / "scene",
                    work_dir=root / "work",
                    final_ply=final_ply,
                    final_spz=None,
                    options={"fine_trainer_repo": str(repo), "fine_scene_profile": "indoor_full"},
                    repo_cache_dir=root / "cache",
                    blur_analysis=SimpleNamespace(kept_images=14),
                )

        run_training.assert_called_once()
        self.assertEqual(result.splat_count, 120)
        self.assertEqual(result.metrics["far_noise_removed_points"], 1)
        self.assertEqual(result.final_ply, final_ply)
        self.assertEqual(result.metrics["deblur_strategy"], "all_training_images")
        self.assertEqual(result.metrics["deblur_applied_images"], 14)

    def test_unknown_trainer_flavor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "foreign_trainer"
            repo.mkdir()
            (repo / "train.py").write_text("from argparse import ArgumentParser\n", encoding="utf-8")

            with self.assertRaises(Exception) as raised:
                resolve_runtime_paths({"fine_trainer_repo": str(repo), "fine_training_flavor": "foreign"}, root / "repo-cache")

        self.assertIn("unsupported fine training flavor", str(raised.exception))

    def test_merged_repo_detection_uses_deblur_train_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "train.py").write_text("parser.add_argument('--config')\ndeblur = 1\n", encoding="utf-8")

            self.assertEqual(detect_trainer_flavor(repo), "dash_deblur_group")

    def test_locate_final_ply_prefers_highest_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            low = root / "point_cloud" / "iteration_100" / "point_cloud.ply"
            high = root / "point_cloud" / "iteration_200" / "point_cloud.ply"
            low.parent.mkdir(parents=True)
            high.parent.mkdir(parents=True)
            low.write_bytes(b"ply\nlow")
            high.write_bytes(b"ply\nhigh")

            self.assertEqual(locate_final_ply(root), high)

    def test_parse_iteration_accepts_common_log_formats(self) -> None:
        self.assertEqual(parse_iteration("iteration 12 loss=0.1"), 12)
        self.assertEqual(parse_iteration("iter=34"), 34)
        self.assertEqual(parse_iteration("56/1000 [00:01]"), 56)

    def _write_xyz_binary_ply(self, path: Path, points: list[tuple[float, float, float]]) -> None:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        ).encode("ascii")
        path.write_bytes(header + b"".join(struct.pack("<fff", *point) for point in points))


if __name__ == "__main__":
    unittest.main()
