from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


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

    def test_preview_worker_uses_resident_process(self) -> None:
        backend_root = Path(__file__).resolve().parents[1]
        source = (backend_root / "app" / "worker.py").read_text(encoding="utf-8")

        self.assertIn("run_preview_task_in_process(task_id, worker_id, redis_client)", source)
        self.assertNotIn("run_task_in_subprocess(task_id, worker_id, redis_client, run_preview_task", source)

    def test_fine_worker_still_uses_task_subprocess(self) -> None:
        backend_root = Path(__file__).resolve().parents[1]
        source = (backend_root / "app" / "fine_worker.py").read_text(encoding="utf-8")

        self.assertIn("run_task_in_subprocess(task_id, worker_id, redis_client, run_fine_task", source)

if __name__ == "__main__":
    unittest.main()
