"""MinIO client helpers."""

from __future__ import annotations

from functools import lru_cache

from minio import Minio
from minio.error import S3Error

from app.core.config import get_settings
from app.helper.logger import get_logger

logger = get_logger()


@lru_cache(maxsize=1)
def get_minio_client() -> Minio:
    settings = get_settings()
    try:
        # In minio 7.2.x, all parameters after endpoint must be keyword-only
        client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        logger.info("MinIO client initialized successfully: endpoint=%s, secure=%s", settings.minio_endpoint, settings.minio_secure)
        return client
    except Exception as e:
        logger.error("Failed to initialize MinIO client: %s", e)
        raise


def ensure_bucket(bucket_name: str) -> None:
    client = get_minio_client()
    try:
        # In minio 7.2.x, bucket_name must be a keyword argument
        if not client.bucket_exists(bucket_name=bucket_name):
            client.make_bucket(bucket_name=bucket_name)
            logger.info("Created MinIO bucket: %s", bucket_name)
        else:
            logger.debug("MinIO bucket already exists: %s", bucket_name)
    except S3Error as exc:
        logger.error("Unable to ensure MinIO bucket %s: %s", bucket_name, exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error ensuring bucket %s: %s (type: %s)",
                    bucket_name, exc, type(exc).__name__)
        raise




