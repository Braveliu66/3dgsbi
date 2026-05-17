from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fine.preprocess import BlurScore  # noqa: E402
from app.fine.video_preprocess import choose_video_sampling, filter_video_frames  # noqa: E402


class FineVideoPreprocessTests(unittest.TestCase):
    def test_choose_video_sampling_uses_denser_indoor_defaults(self) -> None:
        indoor_fps, indoor_side = choose_video_sampling("indoor", 45.0, "auto")
        outdoor_fps, outdoor_side = choose_video_sampling("outdoor", 45.0, "auto")

        self.assertGreater(indoor_fps, outdoor_fps)
        self.assertGreater(indoor_side, outdoor_side)

    def test_filter_video_frames_removes_duplicate_and_low_quality_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            out = root / "out"
            raw.mkdir()
            out.mkdir()
            paths = []
            for index in range(10):
                path = raw / f"image_{index:06d}.jpg"
                path.write_bytes(b"fake")
                paths.append(path)
            scores = [
                BlurScore(
                    path=path,
                    laplacian=180.0,
                    gradient=55.0,
                    fft_high_ratio=0.12,
                    texture_density=0.08,
                    exposure_bad_ratio=0.0,
                )
                for path in paths
            ]
            scores[-1].exposure_bad_ratio = 0.50
            hashes = {path: (index + 1) * 0x1111111111111111 for index, path in enumerate(paths)}
            hashes[paths[2]] = hashes[paths[1]]

            with patch("app.fine.video_preprocess.score_blur_images", return_value=scores), patch(
                "app.fine.video_preprocess.dhash",
                side_effect=lambda path: hashes[path],
            ):
                kept, metrics = filter_video_frames(raw, out, min_frames=8)

            self.assertEqual(len(kept), 8)
            self.assertEqual(metrics["duplicate_frames_removed"], 1)
            self.assertEqual(metrics["quality_frames_removed"], 1)
            self.assertEqual([path.name for path in kept], [f"{index:06d}.jpg" for index in range(8)])


if __name__ == "__main__":
    unittest.main()
