import time
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

SCHEMA_LOCK_ID = 330_397_150


def create_database_schema() -> None:
    import app.models  # noqa: F401

    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            # Serialize concurrent startup from backend and workers.
            connection.execute(text("SELECT pg_advisory_xact_lock(:lock_id)"), {"lock_id": SCHEMA_LOCK_ID})
            Base.metadata.create_all(bind=connection)
        return

    Base.metadata.create_all(bind=engine)


def initialize_database_schema(timeout_seconds: float = 120.0, retry_interval_seconds: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            create_database_schema()
            return
        except OperationalError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(retry_interval_seconds)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
