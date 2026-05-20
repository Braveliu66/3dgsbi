from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    _cv2_placeholder = True
    sys.modules["cv2"] = types.SimpleNamespace()
else:
    _cv2_placeholder = False

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    _numpy_placeholder = True
    sys.modules["numpy"] = types.SimpleNamespace(isnan=math.isnan, nan=float("nan"))
else:
    _numpy_placeholder = False

try:
    import tqdm  # noqa: F401
except ModuleNotFoundError:
    _tqdm_placeholder = True
    sys.modules["tqdm"] = types.SimpleNamespace(tqdm=lambda value, *args, **kwargs: value)
else:
    _tqdm_placeholder = False

from blur_detection_no_percent_adaptive_canny import (  # noqa: E402
    BLUR_EFFECT_BLURRY_THRESHOLD,
    apply_blur_effect_motion_policy_rows,
    classify_blur_effect,
    resolve_worker_count,
)

if _numpy_placeholder:
    del sys.modules["numpy"]
if _cv2_placeholder:
    del sys.modules["cv2"]
if _tqdm_placeholder:
    del sys.modules["tqdm"]


class BlurDetectionMotionPolicyTests(unittest.TestCase):
    def test_blur_effect_thresholds_match_requested_labels(self) -> None:
        self.assertEqual(classify_blur_effect(BLUR_EFFECT_BLURRY_THRESHOLD), "blurry")
        self.assertEqual(classify_blur_effect(0.22), "sharp")
        self.assertEqual(classify_blur_effect(0.30), "uncertain")

    def test_worker_count_defaults_to_bounded_parallelism(self) -> None:
        self.assertEqual(resolve_worker_count(workers=1, image_count=20), 1)
        self.assertEqual(resolve_worker_count(workers=4, image_count=20), 4)
        self.assertEqual(resolve_worker_count(workers=0, image_count=1), 1)
        self.assertGreaterEqual(resolve_worker_count(workers=0, image_count=20), 1)

    def test_blur_effect_only_creates_motion_for_non_defocus_rows(self) -> None:
        rows = apply_blur_effect_motion_policy_rows(
            [
                {"filename": "defocus.jpg", "label": "blurry", "blur_type": "defocus_blur", "blur_effect": 0.10},
                {"filename": "motion.jpg", "label": "sharp", "blur_type": "none", "blur_effect": 0.36},
                {"filename": "uncertain.jpg", "label": "uncertain", "blur_type": "uncertain", "blur_effect": 0.30},
                {"filename": "sharp.jpg", "label": "sharp", "blur_type": "none", "blur_effect": 0.10},
            ]
        )

        self.assertEqual([row["blur_type"] for row in rows], ["defocus_blur", "motion_blur", "none", "none"])
        self.assertEqual([row["label"] for row in rows], ["blurry", "blurry", "sharp", "sharp"])
        self.assertEqual(rows[0]["blur_policy"], "preserve_defocus")
        self.assertEqual(rows[1]["blur_policy"], "blur_effect_motion")
        self.assertEqual(rows[2]["blur_policy"], "blur_effect_clear")


if __name__ == "__main__":
    unittest.main()
