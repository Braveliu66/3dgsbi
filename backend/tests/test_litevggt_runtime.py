from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

try:
    import numpy as np

    RUNTIME_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - local dependency guard
    np = None
    RUNTIME_IMPORT_ERROR = exc


@unittest.skipIf(RUNTIME_IMPORT_ERROR is not None, f"LiteVGGT runtime dependencies unavailable: {RUNTIME_IMPORT_ERROR}")
class LiteVGGTSceneTests(unittest.TestCase):
    def test_fine_litevggt_scene_uses_official_single_runtime(self) -> None:
        from app.fine import litevggt_scene

        reconstruction = SimpleNamespace(
            images=np.zeros((8, 2, 2, 3), dtype=np.float32),
            w2c=np.tile(np.eye(4, dtype=np.float32), (8, 1, 1)),
            intrinsics=np.tile(np.eye(3, dtype=np.float32), (8, 1, 1)),
            points=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            colors=np.array([[255, 255, 255]], dtype=np.uint8),
            metrics={"official_single_path": True},
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            model_cache_dir = root / "model-cache"
            scene_dir = root / "scene"
            weight = model_cache_dir / "litevggt" / "te_dict.pt"
            input_dir.mkdir()
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"weight")

            with patch("app.fine.litevggt_scene.run_litevggt_reconstruction", return_value=reconstruction) as run:
                result = litevggt_scene.build_litevggt_scene(
                    input_dir,
                    scene_dir,
                    model_cache_dir=model_cache_dir,
                    options={},
                    progress=lambda stage, value, message: None,
                )

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["keep_ratio"], 0.90)
        self.assertEqual(kwargs["max_points"], 1_500_000)
        self.assertEqual(kwargs["frame_selection"], "all")
        self.assertEqual(result.metrics["litevggt_official_single_path"], True)


if __name__ == "__main__":
    unittest.main()
