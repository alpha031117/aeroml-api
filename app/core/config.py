"""
Project-wide configuration helpers.

Environment variables are loaded via python-dotenv in app.main, so importing this
module anywhere in the codebase will pick up values from `.env` when present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


def _str_to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_database_url() -> str:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "aeroml")
    user = os.getenv("POSTGRES_USER", "aeroml")
    password = os.getenv("POSTGRES_PASSWORD", "change-me-postgres")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db_name}"


@dataclass(frozen=True)
class Settings:
    """Dataclass holding configuration flags for the API runtime."""

    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    minio_model_bucket: str
    minio_session_bucket: str
    database_url: str
    google_client_id: str
    google_client_secret: str
    google_redirect_uri: str
    jwt_secret_key: str
    jwt_algorithm: str
    jwt_access_token_expires_minutes: int
    jwt_cookie_name: str
    frontend_url: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""

    return Settings(
        minio_endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        minio_access_key=os.getenv("MINIO_ACCESS_KEY", "aeroml_local"),
        minio_secret_key=os.getenv("MINIO_SECRET_KEY", "change-me-super-secret"),
        minio_secure=_str_to_bool(os.getenv("MINIO_SECURE"), default=False),
        minio_model_bucket=os.getenv("MINIO_MODEL_BUCKET", "aeroml-models"),
        minio_session_bucket=os.getenv("MINIO_SESSION_BUCKET", "aeroml-session-data"),
        database_url=os.getenv("DATABASE_URL") or _build_database_url(),
        google_client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
        google_client_secret=os.getenv("GOOGLE_CLIENT_SECRET", ""),
        # Defaults target local backend callback; override in production via env.
        google_redirect_uri=os.getenv(
            "GOOGLE_REDIRECT_URI",
            "http://localhost:8000/api/v1/auth/google/callback",
        ),
        jwt_secret_key=os.getenv("JWT_SECRET_KEY", "change-me-in-production"),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        jwt_access_token_expires_minutes=int(
            os.getenv("JWT_ACCESS_TOKEN_EXPIRES_MINUTES", "60")
        ),
        jwt_cookie_name=os.getenv("JWT_COOKIE_NAME", "access_token"),
        # Frontend base URL (e.g., http://localhost:3000) - will append /model-prompt for OAuth redirect
        frontend_url=os.getenv("FRONTEND_URL", "http://localhost:3000"),
    )


