from __future__ import annotations

from app.config import get_settings
from app.worker import main as worker_main


def main() -> None:
    settings = get_settings()
    if settings.model_auto_download:
        print(f"[worker-bootstrap] task-specific model auto-download enabled in {settings.model_cache_dir}", flush=True)
    else:
        print("[worker-bootstrap] model weight auto-download disabled", flush=True)
    worker_main()


if __name__ == "__main__":
    main()
