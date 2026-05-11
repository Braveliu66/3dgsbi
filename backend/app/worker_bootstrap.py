from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.preview.weights import LINGBOT_MAP_LONG_WEIGHT, seed_model_weights
from app.worker import main as worker_main


def main() -> None:
    settings = get_settings()
    seed_model_weights(
        Path(settings.model_cache_dir),
        Path("/opt/model-cache-seed"),
        (LINGBOT_MAP_LONG_WEIGHT,),
        log=lambda line: print(line, flush=True),
    )
    if settings.model_auto_download:
        print(f"[worker-bootstrap] task-specific model auto-download enabled in {settings.model_cache_dir}", flush=True)
    else:
        print("[worker-bootstrap] model weight auto-download disabled", flush=True)
    worker_main()


if __name__ == "__main__":
    main()
