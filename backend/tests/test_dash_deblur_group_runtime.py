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
    build_training_command,
    build_training_config,
    detect_trainer_flavor,
    locate_final_ply,
    parse_iteration,
    resolve_runtime_paths,
    resolve_effective_deblur_mode,
    run_dash_deblur_group_training,
    write_training_config,
)


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
        self.assertEqual(config["resolution"], 1)
        self.assertEqual(config["deblur"], 6)
        self.assertEqual(config["use_pos"], 0)
        self.assertEqual(config["densify_with_depth"], 0)
        self.assertEqual(config["dash_enable"], True)
        self.assertEqual(config["dash_start_iter"], 1)
        self.assertEqual(config["resolution_mode"], "freq")
        self.assertEqual(config["densify_mode"], "freq")
        self.assertEqual(config["grouping_interval"], 600)
        self.assertEqual(config["Grouping"], False)
        self.assertEqual(config["lambda_p"], 0.0)
        self.assertEqual(config["densify_until_iter"], 18000)
        self.assertEqual(config["densification_interval"], 100)
        self.assertEqual(config["densify_grad_threshold"], 0.0002)
        self.assertEqual(config["max_n_gaussian"], -1)
        self.assertEqual(config["dash_max_densify_rate_per_step"], 0.09)
        self.assertEqual(config["pts_iter"], 999999)
        self.assertEqual(config["pts_N_pts"], 0)

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
        self.assertEqual(config["use_pos"], 1)
        self.assertEqual(config["num_moments"], 4)
        self.assertEqual(config["hidden"], 3)
        self.assertEqual(config["lambda_p"], 0.01)
        self.assertEqual(config["densify_with_depth"], 0)

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
        self.assertEqual(mode.reason, "user_selected")
        self.assertEqual(mode.motion_frames, 0)
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

        self.assertEqual(mode.effective, "motion")
        self.assertEqual(mode.defocus_frames, 0)
        self.assertEqual(config["deblur"], 1)
        self.assertEqual(config["use_pos"], 1)
        self.assertEqual(config["num_moments"], 6)
        self.assertEqual(config["dash_start_iter"], 1)

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
        self.assertEqual(mode.reason, "user_selected")

    def test_explicit_deblur_mode_overrides_blur_analysis(self) -> None:
        blur = SimpleNamespace(
            per_frame_blur={
                "000000.jpg": {"rejected": False, "blurred": True, "kind": "defocus"},
                "000001.jpg": {"rejected": False, "blurred": True, "kind": "defocus"},
            }
        )

        mode = resolve_effective_deblur_mode({"fine_deblur_mode": "motion"}, blur)
        config = build_training_config({"fine_deblur_mode": "motion"}, blur_analysis=blur)

        self.assertEqual(mode.effective, "motion")
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

        self.assertEqual(mode.effective, "motion")
        self.assertEqual(config["deblur"], 1)
        self.assertEqual(config["use_pos"], 1)

    def test_write_training_config_preserves_deblur_dash_group_keys(self) -> None:
        config = build_training_config({"scene_type": "indoor", "fine_deblur_mode": "motion"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.txt"
            write_training_config(path, config)

            text = path.read_text(encoding="utf-8")

        self.assertIn("deblur = 1", text)
        self.assertIn("resolution = 1", text)
        self.assertIn("position_lr_final = 1.6e-05", text)
        self.assertIn("percent_dense = 0.01", text)
        self.assertIn("lambda_dssim = 0.2", text)
        self.assertIn("dash_enable = True", text)
        self.assertIn("densify_with_depth = 0", text)
        self.assertIn("resolution_mode = freq", text)
        self.assertIn("densify_mode = freq", text)
        self.assertIn("max_n_gaussian = -1", text)
        self.assertIn("dash_max_densify_rate_per_step = 0.12", text)
        self.assertIn("pts_N_pts = 0", text)
        self.assertIn("pts_iter = 999999", text)
        self.assertIn("Grouping = False", text)
        self.assertIn("grouping_method = Opacity-weighted", text)
        self.assertNotIn("protect_new_points_iters", text)

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

    def test_training_exports_filtered_final_ply(self) -> None:
        try:
            import numpy  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest(f"viewer filter dependency unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "train.py").write_text("parser.add_argument('--config')\ndeblur = 1\nGrouping = False\n", encoding="utf-8")
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
                    blur_analysis=None,
                )

        run_training.assert_called_once()
        self.assertEqual(result.splat_count, 120)
        self.assertEqual(result.metrics["far_noise_removed_points"], 1)
        self.assertEqual(result.final_ply, final_ply)

    def test_dashgaussian_flavor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "DashGaussian"
            (repo / "utils").mkdir(parents=True)
            (repo / "train.py").write_text("from argparse import ArgumentParser\n", encoding="utf-8")
            (repo / "train_dash.py").write_text("from utils.schedule_utils import TrainingScheduler\n", encoding="utf-8")
            (repo / "utils" / "schedule_utils.py").write_text("class TrainingScheduler: pass\n", encoding="utf-8")

            with self.assertRaises(Exception) as raised:
                resolve_runtime_paths({"fine_trainer_repo": str(repo), "fine_training_flavor": "dashgaussian"}, root / "repo-cache")

        self.assertIn("unsupported fine training flavor", str(raised.exception))

    def test_merged_repo_detection_wins_over_dash_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "utils").mkdir()
            (repo / "train.py").write_text("parser.add_argument('--config')\ndeblur = 1\n", encoding="utf-8")
            (repo / "train_dash.py").write_text("", encoding="utf-8")
            (repo / "utils" / "schedule_utils.py").write_text("", encoding="utf-8")

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
