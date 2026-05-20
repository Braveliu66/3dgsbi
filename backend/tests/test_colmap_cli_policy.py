from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fine.colmap_cli import (  # noqa: E402
    ColmapCapabilities,
    _build_mapper_command,
    _colmap_option_family_compat_command,
    _resolve_matcher_command,
    _run_colmap_with_gpu_fallback,
    detect_colmap_capabilities,
    write_filtered_sparse_points_ply,
)
from app.fine.types import FineFailure  # noqa: E402


STANDARD_COLMAP_HELP = """
Usage:
  colmap [command]

Available commands:
    feature_extractor
    exhaustive_matcher
    sequential_matcher
    mapper
    global_mapper
    image_undistorter
    point_triangulator
"""


class ColmapCliPolicyTests(unittest.TestCase):
    def test_standard_colmap_commands_are_required(self) -> None:
        with patch("app.fine.colmap_cli.shutil.which", return_value="/usr/bin/colmap"), patch(
            "app.fine.colmap_cli.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout=STANDARD_COLMAP_HELP),
        ):
            capabilities = detect_colmap_capabilities()

        self.assertIn("feature_extractor", capabilities.commands)
        self.assertIn("exhaustive_matcher", capabilities.commands)
        self.assertIn("mapper", capabilities.commands)
        self.assertIn("global_mapper", capabilities.commands)
        self.assertIn("image_undistorter", capabilities.commands)
        self.assertIn("point_triangulator", capabilities.commands)

    def test_colmap_help_detection_handles_unindented_commands(self) -> None:
        help_text = "Available commands: feature_extractor exhaustive_matcher mapper image_undistorter point_triangulator"
        with patch("app.fine.colmap_cli.shutil.which", return_value="/usr/bin/colmap"), patch(
            "app.fine.colmap_cli.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout=help_text),
        ):
            capabilities = detect_colmap_capabilities()

        self.assertIn("exhaustive_matcher", capabilities.commands)

    def test_auto_matcher_uses_sequential_for_large_sets(self) -> None:
        self.assertEqual(_resolve_matcher_command("auto", 80), "exhaustive_matcher")
        self.assertEqual(_resolve_matcher_command("auto", 81), "sequential_matcher")

    def test_global_mapper_command_uses_global_mapper(self) -> None:
        capabilities = ColmapCapabilities(
            executable="colmap",
            help_text="",
            commands={"global_mapper"},
        )

        command = _build_mapper_command(
            capabilities,
            "global_mapper",
            Path("database.db"),
            Path("images"),
            Path("sparse"),
            use_gpu=True,
        )

        self.assertEqual(command[1], "global_mapper")

    def test_incremental_mapper_command_still_uses_mapper(self) -> None:
        capabilities = ColmapCapabilities(
            executable="colmap",
            help_text="",
            commands={"mapper"},
        )

        command = _build_mapper_command(
            capabilities,
            "mapper",
            Path("database.db"),
            Path("images"),
            Path("sparse"),
            use_gpu=True,
        )

        self.assertEqual(command[1], "mapper")

    def test_sift_extraction_gpu_option_retries_with_generic_colmap_option(self) -> None:
        command = [
            "colmap",
            "feature_extractor",
            "--SiftExtraction.use_gpu",
            "1",
        ]

        with patch(
            "app.fine.colmap_cli._run_colmap_command",
            side_effect=[
                FineFailure("COLMAP_COMMAND_FAILED", "unrecognised option '--SiftExtraction.use_gpu'"),
                None,
            ],
        ) as run_command:
            _run_colmap_with_gpu_fallback(command, "--SiftExtraction.use_gpu", lambda *_: None, "stage", 1)

        retry_command = run_command.mock_calls[1].args[0]
        self.assertIn("--FeatureExtraction.use_gpu", retry_command)

    def test_sift_matching_gpu_option_retries_with_generic_colmap_option(self) -> None:
        command = ["colmap", "exhaustive_matcher", "--SiftMatching.use_gpu", "1"]

        compat = _colmap_option_family_compat_command(command, "unrecognised option '--SiftMatching.use_gpu'")

        self.assertEqual(compat, ["colmap", "exhaustive_matcher", "--FeatureMatching.use_gpu", "1"])

    def test_sparse_ply_filter_removes_track_and_bbox_outliers(self) -> None:
        from app.fine import sparse_filter

        if sparse_filter.np is None:
            raise unittest.SkipTest("NumPy is required for sparse point filtering")

        class Track:
            def __init__(self, length: int) -> None:
                self._length = length

            def length(self) -> int:
                return self._length

        points = {}
        for index in range(120):
            points[index] = SimpleNamespace(xyz=[float(index % 12), float(index // 12), 0.0], color=[128, 64, 32], error=1.0, track=Track(4))
        points[200] = SimpleNamespace(xyz=[1000.0, 1000.0, 1000.0], color=[255, 0, 0], error=1.0, track=Track(4))
        points[201] = SimpleNamespace(xyz=[0.0, 0.0, 0.0], color=[255, 0, 0], error=20.0, track=Track(1))

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "points3D.ply"
            count = write_filtered_sparse_points_ply(SimpleNamespace(points3D=points), output)
            text = output.read_text(encoding="ascii")

        self.assertEqual(count, 120)
        self.assertIn("element vertex 120", text)
        self.assertNotIn("1000", text)


if __name__ == "__main__":
    unittest.main()
