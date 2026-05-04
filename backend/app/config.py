from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "3DGS Reconstruction Platform"
    database_url: str = "sqlite:///./data/app.db"
    redis_url: str = "redis://127.0.0.1:6379/0"
    cors_origins: str = "http://127.0.0.1:3001,http://localhost:3001,http://127.0.0.1:3000,http://localhost:3000"

    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 60 * 24 * 7
    artifact_token_expire_seconds: int = 3600

    admin_username: str = "admin"
    admin_password: str = "admin123"
    admin_email: str = "admin@example.local"

    storage_backend: str = "local"
    storage_root: str = "./data/storage"
    work_root: str = "./data/work"
    s3_endpoint_url: str | None = None
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "three-dgs"
    s3_region: str = "us-east-1"

    repo_cache_dir: str = "../repo-cache"
    model_cache_dir: str = "../model-cache"
    algorithm_strict: bool = True

    preview_queue_name: str = "preview_tasks"
    preview_expected_seconds_litevggt_spz: int = 180
    preview_expected_seconds_litevggt_edgs: int = 480
    preview_expected_seconds_video: int = 420

    litevggt_repo_commit: str = "4767c17f8b6f176bb751566e92f60eb885040033"
    edgs_repo_commit: str = "9a897645eb47c1b24d4f9e4428cd745927bf1ee1"
    lingbot_repo_commit: str = "f720b421c6c50af3adc63272033226aa4811ef42"
    spark_repo_commit: str = "3cf9fa15adb7ac7c47a1e962740db97b9e8a9fdf"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def local_storage_root(self) -> Path:
        return Path(self.storage_root).resolve()

    @property
    def local_work_root(self) -> Path:
        return Path(self.work_root).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    Path(settings.storage_root).mkdir(parents=True, exist_ok=True)
    Path(settings.work_root).mkdir(parents=True, exist_ok=True)
    Path(settings.database_url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True) if settings.database_url.startswith("sqlite:///") else None
    return settings
