from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fine.types import FineFailure  # noqa: E402


def import_fine_runtime():
    try:
        from app.fine.litevggt_sfm import align_to_litevggt_batch
        from app.fine.preprocess import BlurScore, SceneBuildResult, summarize_blur_scores
        from app.fine.runner import build_scene, deblur_mlp_enabled_by_default, normalize_fine_pipeline
    except Exception as exc:
        raise unittest.SkipTest(f"fine runtime dependencies unavailable: {exc}") from exc
    return (
        align_to_litevggt_batch,
        BlurScore,
        SceneBuildResult,
        summarize_blur_scores,
        build_scene,
        deblur_mlp_enabled_by_default,
        normalize_fine_pipeline,
    )


class FineRuntimeTests(unittest.TestCase):
    def test_fine_runtime_does_not_require_romatch(self) -> None:
        algorithms_source = (BACKEND_ROOT / "app" / "algorithms.py").read_text(encoding="utf-8")
        fine_status_block = algorithms_source.split("def fine_runtime_status", 1)[1].split("def ", 1)[0]

        self.assertNotIn("romatch", fine_status_block)
        self.assertIn("litevggt", fine_status_block)

    def test_trainer_uses_lmrs_phase_two(self) -> None:
        trainer_source = (BACKEND_ROOT / "app" / "fine" / "mobilegs_trainer.py").read_text(encoding="utf-8")
        lmrs_source = (BACKEND_ROOT / "app" / "fine" / "lmrs_runtime.py").read_text(encoding="utf-8")

        self.assertIn("gauss_newton_step", trainer_source)
        self.assertIn("fine_lmrs_phase_start", trainer_source)
        self.assertIn("CGOptimizer", trainer_source)
        self.assertIn("get_JTv", lmrs_source)
        self.assertIn("cgState.set_scene_size", lmrs_source)
        self.assertNotIn("Phase 2 wrapper is not enabled", trainer_source)

    def test_compact_box_patch_is_build_input(self) -> None:
        patch_source = (BACKEND_ROOT.parent / "worker" / "patches" / "lmrs-fastgs-compact-box.patch").read_text(encoding="utf-8")

        self.assertIn("duplicateToTilesTouched", patch_source)
        self.assertIn("MOBILEGS_COMPACT_BOX", patch_source)
        dockerfile = (BACKEND_ROOT.parent / "worker" / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("get_JTv", dockerfile)
        self.assertIn("submodule update --init --recursive", dockerfile)
        self.assertIn("LMRS_GLM", dockerfile)
        self.assertIn("glm/glm.hpp", dockerfile)
        self.assertIn("VENDORED_GLM", dockerfile)

    def test_fine_code_is_split_by_integration_boundary(self) -> None:
        fine_root = BACKEND_ROOT / "app" / "fine"

        self.assertTrue((fine_root / "fastgs_policy.py").exists())
        self.assertTrue((fine_root / "lmrs_runtime.py").exists())
        self.assertTrue((fine_root / "option_utils.py").exists())

    def test_compose_uses_one_worker_image(self) -> None:
        compose_source = (BACKEND_ROOT.parent / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("image: 3dgsbi-worker:local", compose_source)
        self.assertIn("target: worker", compose_source)
        self.assertNotIn("target: worker-preview", compose_source)
        self.assertNotIn("target: worker-fine", compose_source)

    def test_legacy_fine_pipeline_aliases_to_mobilegs_lmrs(self) -> None:
        *_, normalize_fine_pipeline = import_fine_runtime()

        self.assertEqual(normalize_fine_pipeline("fused_quality_3dgs"), "mobilegs_lmrs")
        self.assertEqual(normalize_fine_pipeline(None), "mobilegs_lmrs")

    def test_blur_summary_reports_kept_images(self) -> None:
        _, BlurScore, _, summarize_blur_scores, *_ = import_fine_runtime()
        scores = [
            BlurScore(path=Path(f"{index}.jpg"), laplacian=140.0, gradient=50.0, fft_high_ratio=0.1)
            for index in range(10)
        ]

        summary = summarize_blur_scores(scores, reject_ratio=0.2)

        self.assertEqual(summary.rejected_images, 2)
        self.assertEqual(summary.kept_images, 8)

    def test_sfm_prefers_litevggt(self) -> None:
        _, _, SceneBuildResult, _, build_scene, *_ = import_fine_runtime()
        expected = SceneBuildResult(Path("scene"), "litevggt_colmap_no_exif", 8, 8, 100, {"sfm_backend": "litevggt_colmap_no_exif"})
        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={}, progress=None)
        with tempfile.TemporaryDirectory() as tmp, patch("app.fine.runner.build_litevggt_colmap_scene", return_value=expected) as litevggt, patch(
            "app.fine.runner.build_pycolmap_scene"
        ) as pycolmap:
            result = build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8)

        self.assertEqual(result.backend, "litevggt_colmap_no_exif")
        litevggt.assert_called_once()
        pycolmap.assert_not_called()

    def test_sfm_does_not_auto_fallback_to_pycolmap(self) -> None:
        _, _, _, _, build_scene, *_ = import_fine_runtime()
        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={}, progress=None)
        with tempfile.TemporaryDirectory() as tmp, patch(
            "app.fine.runner.build_litevggt_colmap_scene",
            side_effect=FineFailure("LITEVGGT_WEIGHT_MISSING", "missing"),
        ), patch("app.fine.runner.build_pycolmap_scene") as pycolmap:
            with self.assertRaises(FineFailure) as raised:
                build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8)

        self.assertEqual(raised.exception.code, "LITEVGGT_WEIGHT_MISSING")
        pycolmap.assert_not_called()

    def test_explicit_pycolmap_backend_is_diagnostic_only(self) -> None:
        _, _, SceneBuildResult, _, build_scene, *_ = import_fine_runtime()
        expected = SceneBuildResult(Path("scene"), "pycolmap", 3, 3, 42, {"sfm_backend": "pycolmap"})
        ctx = SimpleNamespace(model_cache_dir=Path("cache"), options={"fine_sfm_backend": "pycolmap"}, progress=None)
        with tempfile.TemporaryDirectory() as tmp, patch("app.fine.runner.build_litevggt_colmap_scene") as litevggt, patch(
            "app.fine.runner.build_pycolmap_scene", return_value=expected
        ) as pycolmap:
            result = build_scene(ctx, Path(tmp), Path(tmp) / "scene", 8192, 1600, 8)

        self.assertEqual(result.backend, "pycolmap")
        self.assertEqual(result.metrics["sfm_backend_requested"], "pycolmap_diagnostic")
        litevggt.assert_not_called()
        pycolmap.assert_called_once()

    def test_litevggt_batch_padding_keeps_real_image_count_visible(self) -> None:
        align_to_litevggt_batch, *_ = import_fine_runtime()
        files = [Path(f"{index}.jpg") for index in range(3)]
        padded = align_to_litevggt_batch(files)
        source = (BACKEND_ROOT / "app" / "fine" / "litevggt_sfm.py").read_text(encoding="utf-8")

        self.assertEqual(len(padded), 8)
        self.assertEqual(padded[:3], files)
        self.assertEqual(padded[3:], [files[-1]] * 5)
        self.assertIn('backend="litevggt_colmap_no_exif"', source)
        self.assertIn("processed[:real_count]", source)
        self.assertIn('"litevggt_padding_images"', source)

    def test_deblur_mlp_replaces_scaling_heuristic(self) -> None:
        trainer_source = (BACKEND_ROOT / "app" / "fine" / "mobilegs_trainer.py").read_text(encoding="utf-8")
        deblur_source = (BACKEND_ROOT / "app" / "fine" / "deblur_mlp.py").read_text(encoding="utf-8")

        self.assertNotIn("deblur_scaling_modifier", trainer_source)
        self.assertNotIn("fine_deblur_scaling_modifier", trainer_source)
        self.assertIn("GTnet", deblur_source)
        self.assertIn("FourierEmbedding", deblur_source)
        self.assertIn("position_delta", deblur_source)
        self.assertIn("render_with_deblur_mlp", trainer_source)

    def test_deblur_auto_controls_lmrs_default(self) -> None:
        *_, deblur_mlp_enabled_by_default, _ = import_fine_runtime()

        self.assertTrue(deblur_mlp_enabled_by_default("motion", {}))
        self.assertTrue(deblur_mlp_enabled_by_default("defocus", {}))
        self.assertTrue(deblur_mlp_enabled_by_default("mixed", {}))
        self.assertFalse(deblur_mlp_enabled_by_default("sharp", {}))
        self.assertFalse(deblur_mlp_enabled_by_default("motion", {"fine_deblur_enabled": "false"}))

    def test_worker_runtime_avoids_duplicate_cuda_stacks(self) -> None:
        worker_root = BACKEND_ROOT.parent / "worker"
        requirements = (worker_root / "requirements.txt").read_text(encoding="utf-8").lower()
        dockerfile = (worker_root / "Dockerfile").read_text(encoding="utf-8").lower()
        combined = requirements + "\n" + dockerfile

        self.assertNotIn("transformer-engine", requirements)
        self.assertNotIn("pycolmap", requirements)
        self.assertIn("transformer-engine[pytorch]==2.4.0", dockerfile)
        self.assertIn("pycolmap==3.12.6", dockerfile)
        self.assertIn("import pycolmap", dockerfile)
        self.assertIn("'einops==0.8.0' 'transformer-engine[pytorch]==2.4.0'", dockerfile)
        self.assertLess(
            dockerfile.index("'einops==0.8.0' 'transformer-engine[pytorch]==2.4.0'"),
            dockerfile.index("import transformer_engine.pytorch as te"),
        )
        self.assertIn("import einops", dockerfile)
        self.assertIn("libcudnn9-dev-cuda-12", dockerfile)
        self.assertIn("cudnn.h", dockerfile)
        self.assertNotIn("nvidia-cudnn-cu12", combined)
        self.assertNotIn("flashinfer", combined)
        self.assertNotIn("kaolin", combined)
        self.assertNotIn("open3d", combined)
        self.assertNotIn("pip install torch", dockerfile)
        self.assertNotIn("pip install torchvision", dockerfile)


if __name__ == "__main__":
    unittest.main()
