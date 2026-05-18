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
        from app.fine import preprocess
    except Exception as exc:
        raise unittest.SkipTest(f"fine preprocess import unavailable: {exc}") from exc
    return preprocess


class FinePreprocessColmapTests(unittest.TestCase):
    def test_blur_analysis_default_records_without_rejecting_frames(self) -> None:
        preprocess = import_preprocess()
        scores = [
            preprocess.BlurScore(Path("blurred.jpg"), laplacian=1.0, gradient=1.0, fft_high_ratio=0.001),
            preprocess.BlurScore(Path("sharp1.jpg"), laplacian=600.0, gradient=80.0, fft_high_ratio=0.25),
            preprocess.BlurScore(Path("sharp2.jpg"), laplacian=650.0, gradient=85.0, fft_high_ratio=0.27),
            preprocess.BlurScore(Path("sharp3.jpg"), laplacian=700.0, gradient=90.0, fft_high_ratio=0.30),
        ]

        with patch.object(preprocess, "score_blur_images", return_value=scores):
            analysis = preprocess.analyze_blur(Path("input"))

        self.assertGreaterEqual(analysis.blurred_images, 1)
        self.assertEqual(analysis.rejected_images, 0)
        self.assertEqual(analysis.kept_images, len(scores))
        self.assertEqual(analysis.rejected_blur_frames, 0)

    def test_pycolmap_matcher_auto_uses_exhaustive_under_250(self) -> None:
        result, pycolmap = self._run_scene(image_count=100)

        self.assertEqual(result.metrics["colmap_matcher"], "exhaustive")
        self.assertTrue(result.metrics["sfm_undistorted"])
        pycolmap.match_exhaustive.assert_called_once()
        pycolmap.match_sequential.assert_not_called()
        pycolmap.undistort_images.assert_called_once()

    def test_pycolmap_matcher_auto_uses_sequential_over_250(self) -> None:
        result, pycolmap = self._run_scene(image_count=1000)

        self.assertEqual(result.metrics["colmap_matcher"], "sequential")
        pycolmap.match_sequential.assert_called_once()
        pycolmap.match_exhaustive.assert_not_called()

    def test_pycolmap_rejects_low_registered_ratio(self) -> None:
        preprocess = import_preprocess()
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
                    preprocess.build_pycolmap_scene(
                        input_dir,
                        scene_dir,
                        max_num_features=8192,
                        max_image_size=1600,
                        min_model_size=3,
                        num_threads=2,
                        matcher="auto",
                        min_registered_ratio=0.70,
                        progress=lambda *_args: None,
                    )

        self.assertEqual(raised.exception.code, "COLMAP_RECONSTRUCTION_INCOMPLETE")
        self.assertIn("below threshold", raised.exception.message)
        pycolmap.undistort_images.assert_not_called()

    def test_pycolmap_allows_low_registered_ratio_by_default(self) -> None:
        result, pycolmap = self._run_scene(image_count=10, registered=5)

        self.assertEqual(result.registered_images, 5)
        self.assertIsNone(result.metrics["sfm_min_registered_ratio"])
        pycolmap.undistort_images.assert_called_once()

    def test_pycolmap_writes_filtered_sparse_ply_and_reports_counts(self) -> None:
        preprocess = import_preprocess()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            scene_dir = root / "scene"
            input_dir.mkdir()
            for index in range(10):
                (input_dir / f"{index:06d}.jpg").write_bytes(b"image")
            pycolmap = self._pycolmap_stub(scene_dir, registered=10)

            with patch.dict(sys.modules, {"pycolmap": pycolmap}), patch.object(preprocess, "write_filtered_sparse_points_ply", return_value=17) as filter_ply:
                result = preprocess.build_pycolmap_scene(
                    input_dir,
                    scene_dir,
                    max_num_features=8192,
                    max_image_size=1600,
                    min_model_size=3,
                    num_threads=2,
                    matcher="auto",
                    progress=lambda *_args: None,
                )

        filter_ply.assert_called_once()
        self.assertEqual(result.point_count, 17)
        self.assertEqual(result.metrics["sfm_sparse_points"], 17)
        self.assertEqual(result.metrics["sfm_sparse_points_raw"], 42)
        self.assertEqual(result.metrics["sfm_sparse_points_filtered"], 17)
        self.assertEqual(result.metrics["sfm_sparse_filter_removed"], 25)

    def test_colmap_camera_validation_rejects_misordered_pinhole_params(self) -> None:
        preprocess = import_preprocess()
        reconstruction = SimpleNamespace(
            cameras={
                1: SimpleNamespace(
                    model="PINHOLE",
                    width=2400,
                    height=1599,
                    params=[2467.5, 1200.0, 799.5, 0.0023],
                )
            }
        )

        with self.assertRaises(FineFailure) as raised:
            preprocess.validate_colmap_pinhole_scene(reconstruction)

        self.assertEqual(raised.exception.code, "COLMAP_CAMERA_INVALID")
        self.assertIn("principal point", raised.exception.message)

    def _run_scene(self, *, image_count: int, registered: int | None = None):
        preprocess = import_preprocess()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            scene_dir = root / "scene"
            input_dir.mkdir()
            for index in range(image_count):
                (input_dir / f"{index:06d}.jpg").write_bytes(b"image")
            pycolmap = self._pycolmap_stub(scene_dir, registered=registered if registered is not None else image_count)

            with patch.dict(sys.modules, {"pycolmap": pycolmap}):
                result = preprocess.build_pycolmap_scene(
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
        camera = SimpleNamespace(
            model="PINHOLE",
            width=2400,
            height=1599,
            params=[2467.5, 2467.5, 1200.0, 799.5],
        )
        reconstruction = SimpleNamespace(
            images={index: object() for index in range(registered)},
            points3D={index: object() for index in range(42)},
            cameras={1: camera},
            write=Mock(),
        )

        def incremental_mapping(*, output_path: Path, **_kwargs):
            model = output_path / "0"
            model.mkdir(parents=True, exist_ok=True)
            (model / "images.bin").write_bytes(b"images")

        def undistort_images(*, output_path: str, **_kwargs):
            sparse = Path(output_path) / "sparse"
            sparse.mkdir(parents=True, exist_ok=True)
            (sparse / "images.bin").write_bytes(b"images")

        return SimpleNamespace(
            extract_features=Mock(),
            match_exhaustive=Mock(),
            match_sequential=Mock(),
            incremental_mapping=Mock(side_effect=incremental_mapping),
            undistort_images=Mock(side_effect=undistort_images),
            IncrementalPipelineOptions=lambda: SimpleNamespace(mapper=SimpleNamespace()),
            Reconstruction=Mock(return_value=reconstruction),
        )


if __name__ == "__main__":
    unittest.main()
