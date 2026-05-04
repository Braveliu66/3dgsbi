from __future__ import annotations

import gzip
import os
import struct
import subprocess
from pathlib import Path

from app.preview.types import PreviewFailure


def convert_ply_to_spz(input_ply: Path, output_spz: Path) -> int:
    """使用项目内置 Spark CLI 把 PLY 转为 Spark 可读 SPZ，并返回 splat 数。"""

    if not input_ply.exists() or input_ply.stat().st_size <= 0:
        raise PreviewFailure("PLY_NOT_FOUND", f"non-empty PLY not found: {input_ply}")

    cli = locate_spark_cli()
    output_spz.parent.mkdir(parents=True, exist_ok=True)
    command = ["node", str(cli), "convert", str(input_ply), str(output_spz)]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8")
    except FileNotFoundError as exc:
        raise PreviewFailure("SPARK_NODE_UNAVAILABLE", "Node.js is required for Spark SPZ conversion") from exc
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise PreviewFailure("SPZ_CONVERSION_FAILED", message or f"Spark converter exited with {completed.returncode}")
    return validate_spz(output_spz)


def validate_spz(path: Path) -> int:
    if not path.exists() or path.stat().st_size <= 0:
        raise PreviewFailure("SPZ_NOT_FOUND", f"non-empty SPZ not found: {path}")
    try:
        with gzip.open(path, "rb") as handle:
            header = handle.read(16)
        magic, version, num_splats = struct.unpack("<III", header[:12])
    except Exception as exc:
        raise PreviewFailure("SPZ_INVALID", f"SPZ header parse failed: {exc}") from exc
    if magic != 0x5053474E or version < 1 or version > 3 or num_splats <= 0:
        raise PreviewFailure("SPZ_INVALID", f"invalid SPZ header: magic={magic:x}, version={version}, splats={num_splats}")
    return int(num_splats)


def locate_spark_cli() -> Path:
    candidates = []
    if os.getenv("SPARK_SPZ_CLI"):
        candidates.append(Path(os.environ["SPARK_SPZ_CLI"]))
    candidates.append(Path(__file__).resolve().parents[1] / "tools" / "spark_transcode_spz.mjs")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise PreviewFailure("SPARK_CLI_NOT_FOUND", "Spark SPZ converter CLI is missing")

