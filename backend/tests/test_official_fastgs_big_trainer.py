from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
            self.assertIn("cpu", command)
            self.assertNotIn("--eval", command)
            self.assertNotIn("git", command)
            self.assertNotIn("github.com/fastgs/FastGS", " ".join(command))
            self.assertEqual(result.ply_path, ply)
            self.assertEqual(result.metrics["training_backend"], "official_fastgs_big")

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
