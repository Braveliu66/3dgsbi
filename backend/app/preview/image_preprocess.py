from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from app.preview.types import PreviewFailure
from app.preview.utils import image_files


@dataclass(slots=True)
class ImagePreprocessResult:
    output_dir: Path
    input_count: int
    output_count: int
    max_side: int
    resized_count: int
    exif_transposed_count: int

    def metrics(self) -> dict[str, int]:
        return {
            "input_image_count": self.input_count,
            "normalized_image_count": self.output_count,
            "image_preprocess_max_side": self.max_side,
            "image_preprocess_resized_count": self.resized_count,
            "image_preprocess_exif_transposed_count": self.exif_transposed_count,
        }


def normalize_image_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    max_side: int,
    jpeg_quality: int = 90,
) -> ImagePreprocessResult:
    """Normalize uploaded photos into RGB JPEGs for preview algorithms."""

    register_optional_heif_support()
    files = image_files(input_dir)
    if not files:
        raise PreviewFailure("IMAGE_INPUT_NOT_FOUND", f"no supported image files found in {input_dir}")

    max_side = max(224, int(max_side))
    jpeg_quality = max(60, min(95, int(jpeg_quality)))
    output_dir.mkdir(parents=True, exist_ok=True)

    resized_count = 0
    exif_transposed_count = 0
    for index, path in enumerate(files):
        try:
            with Image.open(path) as original:
                original.load()
                orientation = original.getexif().get(274)
                image = ImageOps.exif_transpose(original)
                if orientation and orientation != 1:
                    exif_transposed_count += 1
                image = convert_to_rgb(image)
                width, height = image.size
                if max(width, height) > max_side:
                    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                    resized_count += 1
                image.save(output_dir / f"{index:06d}.jpg", format="JPEG", quality=jpeg_quality, optimize=True)
        except UnidentifiedImageError as exc:
            raise PreviewFailure("IMAGE_PREPROCESS_FAILED", f"unsupported image format: {path.name}") from exc
        except OSError as exc:
            raise PreviewFailure("IMAGE_PREPROCESS_FAILED", f"failed to normalize image {path.name}: {exc}") from exc

    return ImagePreprocessResult(
        output_dir=output_dir,
        input_count=len(files),
        output_count=len(files),
        max_side=max_side,
        resized_count=resized_count,
        exif_transposed_count=exif_transposed_count,
    )


def convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def register_optional_heif_support() -> None:
    try:
        from pillow_heif import register_heif_opener
    except Exception:
        return
    register_heif_opener()
