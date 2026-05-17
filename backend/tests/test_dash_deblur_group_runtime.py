from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
        self.assertEqual(config["deblur"], 2)
        self.assertEqual(config["dash_start_iter"], 5000)
        self.assertEqual(config["grouping_interval"], 1000)
        self.assertEqual(config["lambda_p"], 0.0)

    def test_mix_deblur_mode_uses_motion_vote(self) -> None:
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
        self.assertEqual(mode.confidence, "high")
        self.assertEqual(config["deblur"], 1)

    def test_mix_deblur_mode_uses_defocus_vote(self) -> None:
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
        self.assertEqual(config["deblur"], 2)
        self.assertEqual(config["dash_start_iter"], 5000)

    def test_mix_deblur_mode_falls_back_to_motion_when_uncertain(self) -> None:
        blur = SimpleNamespace(
            per_frame_blur={
                "000000.jpg": {"rejected": False, "blurred": True, "kind": "motion"},
                "000001.jpg": {"rejected": False, "blurred": True, "kind": "defocus"},
                "000002.jpg": {"rejected": False, "blurred": True, "kind": "mixed"},
            }
        )

        mode = resolve_effective_deblur_mode({"scene_type": "outdoor", "fine_deblur_mode": "mix"}, blur)

        self.assertEqual(mode.effective, "motion")
        self.assertEqual(mode.confidence, "low")
        self.assertEqual(mode.reason, "outdoor_conservative_default")

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

    def test_write_training_config_preserves_deblur_dash_group_keys(self) -> None:
        config = build_training_config({"scene_type": "indoor", "fine_deblur_mode": "motion"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.txt"
            write_training_config(path, config)

            text = path.read_text(encoding="utf-8")

        self.assertIn("deblur = 1", text)
        self.assertIn("dash_enable = True", text)
        self.assertIn("Grouping = True", text)
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


if __name__ == "__main__":
    unittest.main()
