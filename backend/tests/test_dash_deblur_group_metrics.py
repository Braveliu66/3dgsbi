from __future__ import annotations

import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
METRICS_SOURCE = WORKSPACE_ROOT / "worker" / "trainer" / "dash_deblur_group_gs" / "metrics.py"
TRAIN_SOURCE = WORKSPACE_ROOT / "worker" / "trainer" / "dash_deblur_group_gs" / "train.py"


class DashDeblurGroupMetricsTests(unittest.TestCase):
    def test_ssim_uses_current_skimage_channel_axis(self) -> None:
        source = METRICS_SOURCE.read_text(encoding="utf-8")

        self.assertIn("channel_axis=-1", source)
        self.assertIn("data_range=2.0", source)

    def test_trainer_writes_experiment_curve_csv(self) -> None:
        source = TRAIN_SOURCE.read_text(encoding="utf-8")

        self.assertIn('EXPERIMENT_METRICS_FILE = "experiment_metrics.csv"', source)
        self.assertIn('"loss_photo_raw"', source)
        self.assertIn('"num_gaussians"', source)
        self.assertIn('"vram_reserved_mb"', source)
        self.assertIn('"fps"', source)

    def test_per_image_blur_eval_falls_back_to_lowest_blur_weight(self) -> None:
        source = TRAIN_SOURCE.read_text(encoding="utf-8")

        self.assertIn("def sharp_or_low_blur_weight_subset(cameras, fallback_limit=None):", source)
        self.assertIn('label_record.get("deblur_weight", label_record.get("deblurweight", label_record.get("blur_weight")))', source)
        self.assertIn('sorted(cameras, key=lambda camera: getattr(camera, "blur_weight", 1.0))', source)
        self.assertIn("sharp_or_low_blur_weight_subset(scene.getTestCameras(), fallback_limit=5)", source)
        self.assertIn("sharp_or_low_blur_weight_subset(scene.getTrainCameras(), fallback_limit=5)[:5]", source)

    def test_per_image_blur_trains_on_all_sharp_images(self) -> None:
        source = TRAIN_SOURCE.read_text(encoding="utf-8")

        self.assertIn("def include_sharp_test_cameras_in_train(scene):", source)
        self.assertIn("train_cameras.append(camera)", source)
        self.assertIn("include_sharp_test_cameras_in_train(scene)", source)


if __name__ == "__main__":
    unittest.main()
