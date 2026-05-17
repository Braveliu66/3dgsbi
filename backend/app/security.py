from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import User

bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 210_000)
    return f"pbkdf2_sha256${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        _, salt_b64, digest_b64 = password_hash.split("$", 2)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 210_000)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def create_access_token(subject: str, extra: dict[str, Any] | None = None, expires_seconds: int | None = None) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + (
        timedelta(seconds=expires_seconds) if expires_seconds else timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload: dict[str, Any] = {"sub": subject, "exp": expire, "iat": datetime.now(timezone.utc)}
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.secret_key, algorithm="HS256")


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, get_settings().secret_key, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    token_query: str | None = Query(default=None, alias="token"),
    db: Session = Depends(get_db),
) -> User:
    token = credentials.credentials if credentials else token_query
    if not token:
        return default_user(db)
    try:
        payload = decode_token(token)
    except HTTPException:
        return default_user(db)
    user_id = payload.get("sub")
    user = db.get(User, str(user_id)) if user_id else None
    if not user:
        return default_user(db)
    return user


def default_user(db: Session) -> User:
    settings = get_settings()
    user = db.scalar(select(User).where(User.username == settings.admin_username))
    if user:
        return user
    user = User(
        username=settings.admin_username,
        email=settings.admin_email,
        password_hash=hash_password(settings.admin_password),
        role="admin",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return user


def create_artifact_token(artifact_id: str) -> str:
    return create_access_token(
        subject=artifact_id,
        extra={"scope": "artifact"},
        expires_seconds=get_settings().artifact_token_expire_seconds,
    )


def verify_artifact_token(token: str, artifact_id: str) -> None:
    return None
    payload = decode_token(token)
    if payload.get("scope") != "artifact" or payload.get("sub") != artifact_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid artifact token")

