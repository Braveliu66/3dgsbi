from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from PIL import Image

    from app.preview.image_preprocess import normalize_image_directory
    from app.preview.types import PreviewFailure
except Exception as exc:  # pragma: no cover - depends on local Pillow availability
    Image = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(Image is None, f"image preprocess dependencies unavailable: {IMPORT_ERROR}")
class ImagePreprocessTests(unittest.TestCase):
    def test_normalizes_exif_orientation_and_downscales(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            output = root / "output"
            source.mkdir()

            image = Image.new("RGB", (300, 600), (220, 20, 20))
            exif = Image.Exif()
            exif[274] = 6
            image.save(source / "rotated.jpg", exif=exif)

            result = normalize_image_directory(source, output, max_side=224, jpeg_quality=85)

            with Image.open(output / "000000.jpg") as normalized:
                self.assertEqual(normalized.mode, "RGB")
                self.assertLessEqual(max(normalized.size), 224)
                self.assertGreater(normalized.width, normalized.height)
            self.assertEqual(result.input_count, 1)
            self.assertEqual(result.output_count, 1)
            self.assertEqual(result.resized_count, 1)
            self.assertEqual(result.exif_transposed_count, 1)

    def test_flattens_alpha_to_rgb_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            output = root / "output"
            source.mkdir()
            Image.new("RGBA", (20, 20), (20, 120, 200, 128)).save(source / "alpha.png")

            normalize_image_directory(source, output, max_side=1600, jpeg_quality=90)

            with Image.open(output / "000000.jpg") as normalized:
                self.assertEqual(normalized.mode, "RGB")
                self.assertEqual(normalized.size, (20, 20))

    def test_downscale_preserves_original_aspect_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            output = root / "output"
            source.mkdir()
            Image.new("RGB", (1200, 600), (30, 90, 140)).save(source / "wide.jpg")

            normalize_image_directory(source, output, max_side=300, jpeg_quality=90)

            with Image.open(output / "000000.jpg") as normalized:
                self.assertEqual(normalized.size, (300, 150))

    def test_zero_max_side_keeps_original_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            output = root / "output"
            source.mkdir()
            Image.new("RGB", (1200, 600), (30, 90, 140)).save(source / "wide.jpg")

            result = normalize_image_directory(source, output, max_side=0, jpeg_quality=90)

            with Image.open(output / "000000.jpg") as normalized:
                self.assertEqual(normalized.size, (1200, 600))
            self.assertEqual(result.max_side, 0)
            self.assertEqual(result.resized_count, 0)

    def test_downscale_preserves_portrait_aspect_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            output = root / "output"
            source.mkdir()
            Image.new("RGB", (333, 1001), (30, 90, 140)).save(source / "portrait.jpg")

            normalize_image_directory(source, output, max_side=300, jpeg_quality=90)

            with Image.open(output / "000000.jpg") as normalized:
                self.assertEqual(max(normalized.size), 300)
                self.assertAlmostEqual(normalized.width / normalized.height, 333 / 1001, places=2)

    def test_no_exif_jpg_and_png_normalize_without_camera_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            output = root / "output"
            source.mkdir()
            Image.new("RGB", (32, 24), (10, 20, 30)).save(source / "plain.jpg")
            Image.new("RGB", (24, 32), (40, 50, 60)).save(source / "plain.png")

            result = normalize_image_directory(source, output, max_side=1600, jpeg_quality=90)

            self.assertEqual(result.input_count, 2)
            self.assertEqual(result.output_count, 2)
            self.assertEqual(result.exif_transposed_count, 0)
            self.assertTrue((output / "000000.jpg").exists())
            self.assertTrue((output / "000001.jpg").exists())

    def test_fails_clearly_without_supported_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            source.mkdir()
            (source / "notes.txt").write_text("not an image", encoding="utf-8")

            with self.assertRaises(PreviewFailure) as raised:
                normalize_image_directory(source, root / "output", max_side=1600)

            self.assertEqual(raised.exception.code, "IMAGE_INPUT_NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
