from __future__ import annotations

import contextlib
import socket
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HTTPX_IMPORT_ERROR: Exception | None = None
try:
    import httpx  # noqa: F401
except Exception as exc:
    HTTPX_IMPORT_ERROR = exc

if HTTPX_IMPORT_ERROR is None:
    from app.preview.weights import ModelDownloadError, ModelWeight, download_model_weight, part_path, weights_for_pipeline  # noqa: E402
else:
    ModelDownloadError = ModelWeight = download_model_weight = part_path = weights_for_pipeline = None


PAYLOAD = b"0123456789abcdefghijklmnopqrstuvwxyz"


class WeightHandler(BaseHTTPRequestHandler):
    payload = PAYLOAD
    requests: list[dict[str, str]] = []
    support_range = True
    fail_after: int | None = None

    def do_GET(self) -> None:
        type(self).requests.append({key.lower(): value for key, value in self.headers.items()})
        body = type(self).payload
        status = 200
        start = 0

        range_header = self.headers.get("Range")
        if range_header and type(self).support_range:
            start = int(range_header.replace("bytes=", "").split("-", 1)[0])
            body = body[start:]
            status = 206

        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{len(type(self).payload) - 1}/{len(type(self).payload)}")
        self.end_headers()

        fail_after = type(self).fail_after
        if fail_after is not None:
            self.wfile.write(body[:fail_after])
            self.wfile.flush()
            with contextlib.suppress(OSError):
                self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


@unittest.skipIf(HTTPX_IMPORT_ERROR is not None, f"httpx unavailable: {HTTPX_IMPORT_ERROR}")
class WeightDownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        WeightHandler.requests = []
        WeightHandler.support_range = True
        WeightHandler.fail_after = None

    def test_empty_cache_downloads_complete_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, run_server() as base_url:
            spec = ModelWeight("weights/file.bin", f"{base_url}/file.bin")

            result = download_model_weight(Path(tmp), spec, prefer_hf_mirror=False, lock_timeout_seconds=2)

            target = Path(tmp) / "weights" / "file.bin"
            self.assertEqual(target.read_bytes(), PAYLOAD)
            self.assertEqual(result["status"], "downloaded")
            self.assertTrue(target.with_name("file.bin.download.json").exists())

    def test_existing_file_skips_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, run_server() as base_url:
            target = Path(tmp) / "weights" / "file.bin"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"already-there")
            spec = ModelWeight("weights/file.bin", f"{base_url}/file.bin")

            result = download_model_weight(Path(tmp), spec, prefer_hf_mirror=False, lock_timeout_seconds=2)

            self.assertEqual(target.read_bytes(), b"already-there")
            self.assertEqual(result["status"], "exists")
            self.assertEqual(WeightHandler.requests, [])

    def test_partial_file_uses_range_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, run_server() as base_url:
            target = Path(tmp) / "weights" / "file.bin"
            target.parent.mkdir(parents=True)
            part_path(target).write_bytes(PAYLOAD[:9])
            spec = ModelWeight("weights/file.bin", f"{base_url}/file.bin")

            result = download_model_weight(Path(tmp), spec, prefer_hf_mirror=False, lock_timeout_seconds=2)

            self.assertEqual(target.read_bytes(), PAYLOAD)
            self.assertEqual(WeightHandler.requests[0].get("range"), "bytes=9-")
            self.assertTrue(result["resumed"])

    def test_server_without_range_restarts_partial_download(self) -> None:
        WeightHandler.support_range = False
        with tempfile.TemporaryDirectory() as tmp, run_server() as base_url:
            target = Path(tmp) / "weights" / "file.bin"
            target.parent.mkdir(parents=True)
            part_path(target).write_bytes(b"stale")
            spec = ModelWeight("weights/file.bin", f"{base_url}/file.bin")

            result = download_model_weight(Path(tmp), spec, prefer_hf_mirror=False, lock_timeout_seconds=2)

            self.assertEqual(target.read_bytes(), PAYLOAD)
            self.assertEqual(WeightHandler.requests[0].get("range"), "bytes=5-")
            self.assertFalse(result["resumed"])

    def test_interrupted_download_keeps_part_and_no_final_file(self) -> None:
        WeightHandler.fail_after = 7
        with tempfile.TemporaryDirectory() as tmp, run_server() as base_url:
            target = Path(tmp) / "weights" / "file.bin"
            spec = ModelWeight("weights/file.bin", f"{base_url}/file.bin")

            with self.assertRaises(ModelDownloadError):
                download_model_weight(Path(tmp), spec, prefer_hf_mirror=False, lock_timeout_seconds=2)

            self.assertFalse(target.exists())
            self.assertTrue(part_path(target).exists())
            self.assertGreater(part_path(target).stat().st_size, 0)

    def test_pipeline_weights_are_task_specific(self) -> None:
        self.assertEqual([item.relative_path for item in weights_for_pipeline("litevggt_spz")], ["litevggt/te_dict.pt"])
        self.assertEqual([item.relative_path for item in weights_for_pipeline("litevggt_edgs")], [])
        self.assertEqual([item.relative_path for item in weights_for_pipeline("lingbot_map_spz")], ["lingbot/lingbot-map-long.pt"])
        self.assertEqual(
            [item.relative_path for item in weights_for_pipeline("mobilegs_lmrs")],
            [
                "amb3r/amb3r.pt",
                "roma/roma_outdoor.pth",
                "roma/roma_indoor.pth",
                "roma/dinov2_vitl14_pretrain.pth",
            ],
        )
        self.assertEqual(
            [item.relative_path for item in weights_for_pipeline("video_artdeco_speed3r")],
            [
                "speed3r_pi3/config.json",
                "speed3r_pi3/model.safetensors",
                "mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
                "mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
                "mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl",
            ],
        )


@contextlib.contextmanager
def run_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), WeightHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


if __name__ == "__main__":
    unittest.main()
