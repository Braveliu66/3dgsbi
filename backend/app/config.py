from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent if PACKAGE_DIR.parent.name == "backend" else PACKAGE_DIR.parent


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
    model_auto_download: bool = True
    model_download_prefer_hf_mirror: bool = True
    model_download_lock_timeout_seconds: int = 60 * 60
    algorithm_strict: bool = True

    preview_queue_name: str = "preview_tasks"
    preview_expected_seconds_litevggt_spz: int = 180
    preview_image_max_side: int = 1600
    preview_image_jpeg_quality: int = 90

    fine_queue_name: str = "fine_tasks"
    fine_expected_seconds_images: int = 7200
    fine_expected_seconds_video: int = 14_400
    fine_image_max_side: int = 2400
    fine_iterations: int = 1000

    litevggt_repo_commit: str = "4767c17f8b6f176bb751566e92f60eb885040033"
    amb3r_repo_commit: str = "7aae7fbb77a750651ffa236bb9c3212290c6fc78"
    spark_repo_commit: str = "3cf9fa15adb7ac7c47a1e962740db97b9e8a9fdf"
    fastgs_repo_commit: str = "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f"
    fast_dropgaussian_repo_commit: str = "aba6c08e567bc0b99dfe63f159df3a241562efc4"
    fastergs_repo_commit: str = "12ec29da2a5c0f13974fbe4827c5fc782b817c62"
    freesplatter_repo_commit: str = "c0446c44d9c670c75f2374f4ea32b9588f06723a"
    deblurring_3dgs_repo_commit: str = "e63366b8581c0fde2fda0ab1aea99518da2e2f10"
    three_dgs_lm_repo_commit: str = "d6db64b1844b4303caa2f6e9a0a1ba107b96d6c9"
    lmrs_repo_commit: str = "cb40c7c06c2a60f8314ce095ad7b4513fbb33319"
    lmrs_rasterizer_repo_commit: str = "c2529d3bb13bc38271710785c015a89d9d623237"
    artdeco_repo_commit: str = "bb654395826e50ac9e4671682d901377115a24ce"
    speed3r_repo_commit: str = "5460f7309c87e5daac36385ff6611627de7d7267"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def local_storage_root(self) -> Path:
        return resolve_local_path(self.storage_root)

    @property
    def local_work_root(self) -> Path:
        return resolve_local_path(self.work_root)

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.database_url = resolve_sqlite_url(settings.database_url)
    settings.local_storage_root.mkdir(parents=True, exist_ok=True)
    settings.local_work_root.mkdir(parents=True, exist_ok=True)
    Path(settings.database_url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True) if settings.database_url.startswith("sqlite:///") else None
    return settings


def resolve_local_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def resolve_sqlite_url(value: str) -> str:
    prefix = "sqlite:///"
    if not value.startswith(prefix):
        return value
    raw_path = value[len(prefix) :]
    path = Path(raw_path)
    if path.is_absolute():
        return value
    return f"{prefix}{resolve_local_path(raw_path).as_posix()}"
