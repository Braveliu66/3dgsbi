from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fine.fastgs_defaults import (  # noqa: E402
    FASTGS_DENSIFICATION_INTERVAL,
    FASTGS_DEBLUR_EXTRA_POINTS_MANDATORY,
    FASTGS_DEBLUR_EXTRA_POINTS_TARGET,
    FASTGS_DEBLUR_EXTRA_POINTS_WEAK_TARGET,
    FASTGS_FINAL_PRUNE_MIN_OPACITY,
    FASTGS_FINAL_PRUNE_ENABLED,
    FASTGS_FINAL_PRUNE_SCORE_THRESH,
    FASTGS_GRAD_ABS_THRESH,
    FASTGS_GRAD_THRESH,
    FASTGS_LATE_PRUNE_ENABLED,
    FASTGS_LATE_PRUNE_INTERVAL,
    FASTGS_LATE_PRUNE_MAX_FRACTION,
    FASTGS_LATE_PRUNE_MIN_OPACITY,
    FASTGS_DEBLUR_AUTO_SCHEDULE,
    FASTGS_DEBLUR_BLURRED_VIEWS_ONLY,
    FASTGS_DEBLUR_ENABLED,
    FASTGS_DEBLUR_LATE_DENSIFY_ENABLED,
    FASTGS_DEBLUR_MODE,
    FASTGS_DEBLUR_NUM_MOMENTS,
    FASTGS_DEBLUR_SCHEDULE_PROFILE,
    FASTGS_DEBLUR_TRANSFORM_REG_WEIGHT,
    FASTGS_DEBLUR_XYZ_LR_SCALE,
    FASTGS_MULT,
    FASTGS_SAMPLE_CAMERAS,
    FASTGS_VCD_BLEND_ALPHA,
    FASTGS_VCD_SCORE_THRESH,
    FASTGS_VCP_BLUR_PROTECT_WEIGHT,
)
from app.fine.types import FineFailure  # noqa: E402


def import_trainer():
    try:
        from app.fine.official_fastgs_big_trainer import train_official_fastgs_big
    except Exception as exc:
        raise unittest.SkipTest(f"official FastGS trainer import unavailable: {exc}") from exc
    return train_official_fastgs_big


class OfficialFastGSBigTrainerTests(unittest.TestCase):
    def test_train_official_fastgs_big_builds_local_vendor_command(self) -> None:
        train_official_fastgs_big = import_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = root / "vendor" / "fastgs"
            self._write_vendor_stub(vendor)
            scene = root / "scene"
            output = root / "output"
            scene.mkdir()
            ply = output / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            ply.parent.mkdir(parents=True)
            ply.write_bytes(b"ply\n")

            process = SimpleNamespace(
                stdout=iter(["Training progress\n", "[ITER 30000] Saving Gaussians\n"]),
                wait=lambda: 0,
                returncode=0,
            )
            with patch.dict(os.environ, {"FASTGS_VENDOR_ROOT": str(vendor)}), patch(
                "app.fine.official_fastgs_big_trainer.subprocess.Popen",
                return_value=process,
            ) as popen:
                result = train_official_fastgs_big(
                    scene_dir=scene,
                    output_dir=output,
                    iterations=30000,
                    options={},
                    progress=lambda *_args: None,
                )

            command = popen.call_args.args[0]
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(Path(command[1]), (vendor / "train.py").resolve())
            self.assertIn("--data_device", command)
            self.assertIn("cuda", command)
            self.assertIn("--densification_interval", command)
            self.assertEqual(command[command.index("--densification_interval") + 1], str(FASTGS_DENSIFICATION_INTERVAL))
            self.assertIn("-r", command)
            self.assertEqual(command[command.index("-r") + 1], "1500")
            self.assertEqual(command[command.index("--position_lr_max_steps") + 1], "30000")
            self.assertEqual(command[command.index("--densify_from_iter") + 1], "500")
            self.assertEqual(command[command.index("--densify_until_iter") + 1], "30000")
            self.assertEqual(command[command.index("--opacity_reset_interval") + 1], "100000")
            self.assertEqual(command[command.index("--grad_thresh") + 1], str(FASTGS_GRAD_THRESH))
            self.assertEqual(command[command.index("--grad_abs_thresh") + 1], str(FASTGS_GRAD_ABS_THRESH))
            self.assertEqual(command[command.index("--fastgs_sample_cameras") + 1], str(FASTGS_SAMPLE_CAMERAS))
            self.assertEqual(command[command.index("--fastgs_vcd_blend_alpha") + 1], str(FASTGS_VCD_BLEND_ALPHA))
            self.assertEqual(command[command.index("--fastgs_vcd_score_thresh") + 1], str(FASTGS_VCD_SCORE_THRESH))
            self.assertEqual(command[command.index("--fastgs_vcp_blur_protect_weight") + 1], str(FASTGS_VCP_BLUR_PROTECT_WEIGHT))
            self.assertEqual(command[command.index("--mult") + 1], str(FASTGS_MULT))
            self.assertIn("--fastgs_final_prune_min_opacity", command)
            self.assertEqual(command[command.index("--fastgs_final_prune_min_opacity") + 1], str(FASTGS_FINAL_PRUNE_MIN_OPACITY))
            self.assertEqual(command[command.index("--fastgs_final_prune_enabled") + 1], "true" if FASTGS_FINAL_PRUNE_ENABLED else "false")
            self.assertIn("--fastgs_final_prune_score_thresh", command)
            self.assertEqual(command[command.index("--fastgs_final_prune_score_thresh") + 1], str(FASTGS_FINAL_PRUNE_SCORE_THRESH))
            self.assertEqual(command[command.index("--fastgs_late_prune_enabled") + 1], "true" if FASTGS_LATE_PRUNE_ENABLED else "false")
            self.assertEqual(command[command.index("--fastgs_late_prune_interval") + 1], str(FASTGS_LATE_PRUNE_INTERVAL))
            self.assertEqual(command[command.index("--fastgs_late_prune_from_iter") + 1], "30000")
            self.assertEqual(command[command.index("--fastgs_late_prune_until_iter") + 1], "30000")
            self.assertEqual(command[command.index("--fastgs_late_prune_min_opacity") + 1], str(FASTGS_LATE_PRUNE_MIN_OPACITY))
            self.assertEqual(command[command.index("--fastgs_late_prune_max_fraction") + 1], str(FASTGS_LATE_PRUNE_MAX_FRACTION))
            self.assertEqual(command[command.index("--scene_type") + 1], "auto")
            self.assertEqual(command[command.index("--deblur_enabled") + 1], FASTGS_DEBLUR_ENABLED)
            self.assertEqual(command[command.index("--deblur_mode") + 1], FASTGS_DEBLUR_MODE)
            self.assertEqual(command[command.index("--deblur_auto_schedule") + 1], FASTGS_DEBLUR_AUTO_SCHEDULE)
            self.assertEqual(command[command.index("--deblur_schedule_profile") + 1], FASTGS_DEBLUR_SCHEDULE_PROFILE)
            self.assertEqual(command[command.index("--deblur_late_densify_enabled") + 1], FASTGS_DEBLUR_LATE_DENSIFY_ENABLED)
            self.assertEqual(command[command.index("--deblur_warmup_iters") + 1], "3000")
            self.assertEqual(command[command.index("--deblur_extra_points_mandatory") + 1], FASTGS_DEBLUR_EXTRA_POINTS_MANDATORY)
            self.assertEqual(command[command.index("--deblur_extra_points_target") + 1], str(FASTGS_DEBLUR_EXTRA_POINTS_TARGET))
            self.assertEqual(command[command.index("--deblur_extra_points_weak_target") + 1], str(FASTGS_DEBLUR_EXTRA_POINTS_WEAK_TARGET))
            self.assertEqual(command[command.index("--deblur_sharp_refine_from_iter") + 1], "29999")
            self.assertEqual(command[command.index("--deblur_num_moments") + 1], str(FASTGS_DEBLUR_NUM_MOMENTS))
            self.assertEqual(command[command.index("--deblur_transform_reg_weight") + 1], str(FASTGS_DEBLUR_TRANSFORM_REG_WEIGHT))
            self.assertEqual(command[command.index("--deblur_xyz_lr_scale") + 1], str(FASTGS_DEBLUR_XYZ_LR_SCALE))
            self.assertEqual(command[command.index("--deblur_blurred_views_only") + 1], FASTGS_DEBLUR_BLURRED_VIEWS_ONLY)
            self.assertNotIn("--eval", command)
            self.assertNotIn("git", command)
            self.assertNotIn("github.com/fastgs/FastGS", " ".join(command))
            self.assertEqual(result.ply_path, ply)
            self.assertEqual(result.metrics["training_backend"], "official_fastgs_big")

    def test_train_official_fastgs_big_scales_schedule_with_iterations(self) -> None:
        train_official_fastgs_big = import_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = root / "vendor" / "fastgs"
            self._write_vendor_stub(vendor)
            scene = root / "scene"
            output = root / "output"
            scene.mkdir()
            ply = output / "point_cloud" / "iteration_35000" / "point_cloud.ply"
            ply.parent.mkdir(parents=True)
            ply.write_bytes(b"ply\n")

            process = SimpleNamespace(stdout=iter([]), wait=lambda: 0, returncode=0)
            with patch.dict(os.environ, {"FASTGS_VENDOR_ROOT": str(vendor)}), patch(
                "app.fine.official_fastgs_big_trainer.subprocess.Popen",
                return_value=process,
            ) as popen:
                result = train_official_fastgs_big(
                    scene_dir=scene,
                    output_dir=output,
                    iterations=35000,
                    options={},
                    progress=lambda *_args: None,
                )

            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--iterations") + 1], "35000")
            self.assertEqual(command[command.index("--position_lr_max_steps") + 1], "35000")
            self.assertEqual(command[command.index("--densify_from_iter") + 1], "500")
            self.assertEqual(command[command.index("--densify_until_iter") + 1], "34000")
            self.assertEqual(command[command.index("--opacity_reset_interval") + 1], "100000")
            self.assertEqual(command[command.index("--fastgs_late_prune_interval") + 1], "4000")
            self.assertEqual(command[command.index("--fastgs_late_prune_from_iter") + 1], "35000")
            self.assertEqual(command[command.index("--fastgs_late_prune_until_iter") + 1], "35000")
            self.assertEqual(command[command.index("--deblur_warmup_iters") + 1], "3000")
            self.assertEqual(result.metrics["densify_until_iter"], 34000)

    def test_train_official_fastgs_big_applies_outdoor_profile_defaults(self) -> None:
        train_official_fastgs_big = import_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = root / "vendor" / "fastgs"
            self._write_vendor_stub(vendor)
            scene = root / "scene"
            output = root / "output"
            scene.mkdir()
            ply = output / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            ply.parent.mkdir(parents=True)
            ply.write_bytes(b"ply\n")

            process = SimpleNamespace(stdout=iter([]), wait=lambda: 0, returncode=0)
            with patch.dict(os.environ, {"FASTGS_VENDOR_ROOT": str(vendor)}), patch(
                "app.fine.official_fastgs_big_trainer.subprocess.Popen",
                return_value=process,
            ) as popen:
                result = train_official_fastgs_big(
                    scene_dir=scene,
                    output_dir=output,
                    iterations=30000,
                    options={"fine_scene_profile": "outdoor_fast_clean"},
                    progress=lambda *_args: None,
                )

            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--scene_type") + 1], "outdoor_full")
            self.assertEqual(command[command.index("--densify_until_iter") + 1], "30000")
            self.assertEqual(command[command.index("--fastgs_late_prune_interval") + 1], "2500")
            self.assertEqual(command[command.index("--fastgs_late_prune_from_iter") + 1], "30000")
            self.assertEqual(command[command.index("--fastgs_late_prune_max_world_scale_ratio") + 1], "0.18")
            self.assertEqual(command[command.index("--fastgs_late_prune_max_fraction") + 1], "0.02")
            self.assertEqual(command[command.index("--fastgs_final_prune_max_world_scale_ratio") + 1], "0.15")
            self.assertEqual(command[command.index("--deblur_sharp_refine_from_iter") + 1], "29999")
            self.assertEqual(result.metrics["scene_parameter_profile"], "outdoor")

    def test_train_official_fastgs_big_passes_deblur_options_and_registry(self) -> None:
        train_official_fastgs_big = import_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = root / "vendor" / "fastgs"
            self._write_vendor_stub(vendor)
            scene = root / "scene"
            output = root / "output"
            scene.mkdir()
            registry = root / "blur_frame_registry.json"
            registry.write_text("{}", encoding="utf-8")
            ply = output / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            ply.parent.mkdir(parents=True)
            ply.write_bytes(b"ply\n")

            process = SimpleNamespace(stdout=iter([]), wait=lambda: 0, returncode=0)
            with patch.dict(os.environ, {"FASTGS_VENDOR_ROOT": str(vendor)}), patch(
                "app.fine.official_fastgs_big_trainer.subprocess.Popen",
                return_value=process,
            ) as popen:
                result = train_official_fastgs_big(
                    scene_dir=scene,
                    output_dir=output,
                    iterations=30000,
                    options={
                        "fine_deblur_enabled": "auto",
                        "fine_deblur_mode": "mixed",
                        "fine_deblur_auto_schedule": "false",
                        "fine_deblur_schedule_profile": "balanced",
                        "fine_deblur_late_densify_enabled": "true",
                        "fine_deblur_blur_registry": str(registry),
                    },
                    progress=lambda *_args: None,
                )

            command = popen.call_args.args[0]
            self.assertIn("--deblur_enabled", command)
            self.assertIn("auto", command)
            self.assertIn("--deblur_mode", command)
            self.assertIn("mixed", command)
            self.assertEqual(command[command.index("--deblur_auto_schedule") + 1], "false")
            self.assertEqual(command[command.index("--deblur_schedule_profile") + 1], "balanced")
            self.assertEqual(command[command.index("--deblur_late_densify_enabled") + 1], "true")
            self.assertIn("--deblur_blur_registry", command)
            self.assertIn(str(registry), command)
            self.assertEqual(result.metrics["deblur_enabled"], "auto")
            self.assertEqual(result.metrics["deblur_auto_schedule"], "false")
            self.assertEqual(result.metrics["deblur_schedule_profile"], "balanced")
            self.assertEqual(result.metrics["deblur_late_densify_enabled"], "true")

    def test_train_official_fastgs_big_allows_late_prune_overrides(self) -> None:
        train_official_fastgs_big = import_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = root / "vendor" / "fastgs"
            self._write_vendor_stub(vendor)
            scene = root / "scene"
            output = root / "output"
            scene.mkdir()
            ply = output / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            ply.parent.mkdir(parents=True)
            ply.write_bytes(b"ply\n")

            process = SimpleNamespace(stdout=iter([]), wait=lambda: 0, returncode=0)
            with patch.dict(os.environ, {"FASTGS_VENDOR_ROOT": str(vendor)}), patch(
                "app.fine.official_fastgs_big_trainer.subprocess.Popen",
                return_value=process,
            ) as popen:
                result = train_official_fastgs_big(
                    scene_dir=scene,
                    output_dir=output,
                    iterations=30000,
                    options={
                        "fine_fastgs_late_prune_enabled": False,
                        "fine_fastgs_late_prune_interval": 1200,
                        "fine_fastgs_final_prune_min_opacity": 0.04,
                        "fine_fastgs_vcd_blend_alpha": 0.6,
                        "fine_fastgs_vcd_score_thresh": 0.45,
                        "fine_fastgs_vcp_blur_protect_weight": 0.35,
                        "fine_densify_until_iter": 9000,
                        "fine_deblur_warmup_iters": 4500,
                    },
                    progress=lambda *_args: None,
                )

            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--fastgs_late_prune_enabled") + 1], "false")
            self.assertEqual(command[command.index("--fastgs_late_prune_interval") + 1], "1200")
            self.assertEqual(command[command.index("--fastgs_final_prune_min_opacity") + 1], "0.04")
            self.assertEqual(command[command.index("--fastgs_vcd_blend_alpha") + 1], "0.6")
            self.assertEqual(command[command.index("--fastgs_vcd_score_thresh") + 1], "0.45")
            self.assertEqual(command[command.index("--fastgs_vcp_blur_protect_weight") + 1], "0.35")
            self.assertEqual(command[command.index("--densify_until_iter") + 1], "9000")
            self.assertEqual(command[command.index("--deblur_warmup_iters") + 1], "4500")
            self.assertEqual(result.metrics["fastgs_late_prune_enabled"], "false")
            self.assertEqual(result.metrics["fastgs_vcd_blend_alpha"], 0.6)
            self.assertEqual(result.metrics["fastgs_vcd_score_thresh"], 0.45)
            self.assertEqual(result.metrics["fastgs_vcp_blur_protect_weight"], 0.35)

    def test_train_official_fastgs_big_finds_final_ply(self) -> None:
        train_official_fastgs_big = import_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vendor = root / "vendor" / "fastgs"
            self._write_vendor_stub(vendor)
            scene = root / "scene"
            output = root / "output"
            scene.mkdir()
            expected = output / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"ply\npayload")

            process = SimpleNamespace(stdout=iter([]), wait=lambda: 0, returncode=0)
            with patch.dict(os.environ, {"FASTGS_VENDOR_ROOT": str(vendor)}), patch(
                "app.fine.official_fastgs_big_trainer.subprocess.Popen",
                return_value=process,
            ):
                result = train_official_fastgs_big(
                    scene_dir=scene,
                    output_dir=output,
                    iterations=30000,
                    options={},
                    progress=lambda *_args: None,
                )

            self.assertEqual(result.ply_path, expected)
            self.assertEqual(result.metrics["final_ply_bytes"], expected.stat().st_size)

    def test_train_official_fastgs_big_fails_when_vendor_missing(self) -> None:
        train_official_fastgs_big = import_trainer()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"FASTGS_VENDOR_ROOT": str(Path(tmp) / "missing")}):
            with self.assertRaises(FineFailure) as raised:
                train_official_fastgs_big(
                    scene_dir=Path(tmp) / "scene",
                    output_dir=Path(tmp) / "output",
                    iterations=30000,
                    options={},
                    progress=lambda *_args: None,
                )

        self.assertEqual(raised.exception.code, "FASTGS_VENDOR_MISSING")

    def _write_vendor_stub(self, vendor: Path) -> None:
        for relative in ("gaussian_renderer", "scene", "utils"):
            (vendor / relative).mkdir(parents=True, exist_ok=True)
        (vendor / "train.py").write_text("# stub\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
