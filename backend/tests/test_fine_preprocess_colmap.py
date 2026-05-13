from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fine.types import FineFailure  # noqa: E402


def import_preprocess():
    try:
        from app.fine.preprocess import build_pycolmap_scene
    except Exception as exc:
        raise unittest.SkipTest(f"fine preprocess import unavailable: {exc}") from exc
    return build_pycolmap_scene


class FinePreprocessColmapTests(unittest.TestCase):
    def test_pycolmap_matcher_auto_uses_exhaustive_under_250(self) -> None:
        result, pycolmap = self._run_scene(image_count=100)

        self.assertEqual(result.metrics["colmap_matcher"], "exhaustive")
        pycolmap.match_exhaustive.assert_called_once()
        pycolmap.match_sequential.assert_not_called()

    def test_pycolmap_matcher_auto_uses_sequential_over_250(self) -> None:
        result, pycolmap = self._run_scene(image_count=1000)

        self.assertEqual(result.metrics["colmap_matcher"], "sequential")
        pycolmap.match_sequential.assert_called_once()
        pycolmap.match_exhaustive.assert_not_called()

    def test_pycolmap_rejects_low_registered_ratio(self) -> None:
        build_pycolmap_scene = import_preprocess()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            scene_dir = root / "scene"
            input_dir.mkdir()
            for index in range(10):
                (input_dir / f"{index:06d}.jpg").write_bytes(b"image")
            pycolmap = self._pycolmap_stub(scene_dir, registered=5)

            with patch.dict(sys.modules, {"pycolmap": pycolmap}):
                with self.assertRaises(FineFailure) as raised:
                    build_pycolmap_scene(
                        input_dir,
                        scene_dir,
                        max_num_features=8192,
                        max_image_size=1600,
                        min_model_size=3,
                        num_threads=2,
                        matcher="auto",
                        progress=lambda *_args: None,
                    )

        self.assertEqual(raised.exception.code, "COLMAP_RECONSTRUCTION_INCOMPLETE")
        self.assertIn("below threshold", raised.exception.message)

    def _run_scene(self, *, image_count: int):
        build_pycolmap_scene = import_preprocess()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            scene_dir = root / "scene"
            input_dir.mkdir()
            for index in range(image_count):
                (input_dir / f"{index:06d}.jpg").write_bytes(b"image")
            pycolmap = self._pycolmap_stub(scene_dir, registered=image_count)

            with patch.dict(sys.modules, {"pycolmap": pycolmap}):
                result = build_pycolmap_scene(
                    input_dir,
                    scene_dir,
                    max_num_features=8192,
                    max_image_size=1600,
                    min_model_size=3,
                    num_threads=2,
                    matcher="auto",
                    progress=lambda *_args: None,
                )
        return result, pycolmap

    def _pycolmap_stub(self, scene_dir: Path, *, registered: int):
        reconstruction = SimpleNamespace(
            images={index: object() for index in range(registered)},
            points3D={index: object() for index in range(42)},
            cameras={},
            write=Mock(),
        )

        def incremental_mapping(*, output_path: Path, **_kwargs):
            model = output_path / "0"
            model.mkdir(parents=True, exist_ok=True)
            (model / "images.bin").write_bytes(b"images")

        return SimpleNamespace(
            extract_features=Mock(),
            match_exhaustive=Mock(),
            match_sequential=Mock(),
            incremental_mapping=Mock(side_effect=incremental_mapping),
            IncrementalPipelineOptions=lambda: SimpleNamespace(mapper=SimpleNamespace()),
            Reconstruction=Mock(return_value=reconstruction),
        )


if __name__ == "__main__":
    unittest.main()
