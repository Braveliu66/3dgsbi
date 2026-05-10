from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class WorkerLoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        from app.config import get_settings

        get_settings.cache_clear()

    def test_task_log_capture_records_python_and_fd_writes(self) -> None:
        from app.worker import TaskLogCapture

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "task.log"
            capture = TaskLogCapture(log_path)
            capture.start()
            print("python stdout line", flush=True)
            os.write(2, b"fd stderr line\n")
            capture.stop()

            data = log_path.read_text(encoding="utf-8")

        self.assertIn("python stdout line", data)
        self.assertIn("fd stderr line", data)

    def test_process_exit_message_explains_sigkill(self) -> None:
        from app.worker import process_exit_message

        self.assertIn("SIGKILL", process_exit_message(-9, "worker"))
        self.assertIn("possible OOM", process_exit_message(137, "worker"))

    def test_lingbot_eta_uses_frame_metrics(self) -> None:
        from app.worker import estimate_lingbot_eta

        task = SimpleNamespace(
            current_stage="lingbot_inference",
            metrics={
                "lingbot_current_frame": 10,
                "lingbot_total_frames": 20,
                "lingbot_seconds_per_frame": 2.5,
            },
        )

        self.assertEqual(estimate_lingbot_eta(task), 25)

    def test_lingbot_eta_ignores_stale_metrics_outside_inference(self) -> None:
        from app.worker import estimate_lingbot_eta

        task = SimpleNamespace(
            current_stage="spz_conversion",
            metrics={"lingbot_inference_eta_seconds": 120},
        )

        self.assertIsNone(estimate_lingbot_eta(task))


if __name__ == "__main__":
    unittest.main()
