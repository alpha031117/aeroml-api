"""Helpers for persisting model and session artifacts in MinIO."""

from __future__ import annotations

import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Optional
import shutil

from minio.error import S3Error

from app.core.config import get_settings
from app.helper.logger import get_logger
from app.services.minio_service import ensure_bucket, get_minio_client

logger = get_logger()
settings = get_settings()


def _zip_path(source_path: Path, label: str) -> Path:
    """Create a temporary zip file for the provided directory."""
    if not source_path.exists():
        raise FileNotFoundError(f"Source path {source_path} does not exist")

    temp_dir = Path(tempfile.mkdtemp())
    zip_path = temp_dir / f"{label}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if source_path.is_file():
            archive.write(source_path, arcname=source_path.name)
        else:
            for file_path in source_path.rglob("*"):
                if file_path.is_file():
                    archive.write(file_path, arcname=file_path.relative_to(source_path))

    return zip_path


def upload_model_artifact(model_path: str, session_id: str, user_id: uuid.UUID) -> Optional[str]:
    """Compress the trained model directory and upload it to MinIO."""
    if not model_path:
        logger.warning("No model path provided for upload")
        return None

    try:
        logger.info("Starting model artifact upload for session %s, model_path=%s", session_id, model_path)

        # Ensure the model path exists
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        ensure_bucket(settings.minio_model_bucket)
        client = get_minio_client()
        object_key = f"models/{user_id}/{session_id}.zip"

        logger.info("Compressing model directory: %s", model_path)
        zip_path = _zip_path(model_path_obj, model_path_obj.name)

        logger.info("Uploading to MinIO: bucket=%s, key=%s", settings.minio_model_bucket, object_key)
        # In minio 7.2.x, use keyword arguments for clarity
        client.fput_object(
            bucket_name=settings.minio_model_bucket,
            object_name=object_key,
            file_path=str(zip_path),
            content_type="application/zip",
        )
        logger.info("Successfully uploaded model artifact for session %s to %s", session_id, object_key)
        return object_key
    except S3Error as exc:
        logger.error("S3Error during model artifact upload for session %s: %s", session_id, exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error during model artifact upload for session %s: %s (type: %s)",
                    session_id, exc, type(exc).__name__)
        raise
    finally:
        if 'zip_path' in locals():
            shutil.rmtree(zip_path.parent, ignore_errors=True)


def upload_session_artifact(session_dir: Path, session_id: str, user_id: uuid.UUID) -> Optional[str]:
    """Compress the session directory and upload it to MinIO."""
    if not session_dir.exists():
        logger.warning("Session directory does not exist: %s", session_dir)
        return None

    try:
        logger.info("Starting session artifact upload for session %s, session_dir=%s", session_id, session_dir)

        ensure_bucket(settings.minio_session_bucket)
        client = get_minio_client()
        object_key = f"sessions/{user_id}/{session_id}.zip"

        logger.info("Compressing session directory: %s", session_dir)
        zip_path = _zip_path(session_dir, session_id)

        logger.info("Uploading to MinIO: bucket=%s, key=%s", settings.minio_session_bucket, object_key)
        # In minio 7.2.x, use keyword arguments for clarity
        client.fput_object(
            bucket_name=settings.minio_session_bucket,
            object_name=object_key,
            file_path=str(zip_path),
            content_type="application/zip",
        )
        logger.info("Successfully uploaded session artifact %s", object_key)
        return object_key
    except S3Error as exc:
        logger.error("S3Error during session artifact upload for session %s: %s", session_id, exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error during session artifact upload for session %s: %s (type: %s)",
                    session_id, exc, type(exc).__name__)
        raise
    finally:
        if 'zip_path' in locals():
            shutil.rmtree(zip_path.parent, ignore_errors=True)


def download_session_artifact(session_object_key: str, destination: Path) -> Path:
    """Download and extract a session artifact from MinIO."""
    ensure_bucket(settings.minio_session_bucket)
    client = get_minio_client()

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
        # In minio 7.2.x, use keyword arguments for clarity
        client.fget_object(
            bucket_name=settings.minio_session_bucket,
            object_name=session_object_key,
            file_path=temp_zip.name
        )
        temp_zip_path = Path(temp_zip.name)

    with zipfile.ZipFile(temp_zip_path, "r") as archive:
        archive.extractall(path=destination)

    temp_zip_path.unlink(missing_ok=True)
    logger.info("Downloaded session artifact %s to %s", session_object_key, destination)
    return destination


def download_model_artifact(model_object_key: str, destination: Path) -> Path:
    """Download and extract a model artifact from MinIO."""
    ensure_bucket(settings.minio_model_bucket)
    client = get_minio_client()

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
        # In minio 7.2.x, use keyword arguments for clarity
        client.fget_object(
            bucket_name=settings.minio_model_bucket,
            object_name=model_object_key,
            file_path=temp_zip.name
        )
        temp_zip_path = Path(temp_zip.name)

    with zipfile.ZipFile(temp_zip_path, "r") as archive:
        archive.extractall(path=destination)

    temp_zip_path.unlink(missing_ok=True)
    logger.info("Downloaded model artifact %s to %s", model_object_key, destination)
    return destination


def list_user_artifacts(user_id: uuid.UUID) -> dict:
    """List all model and session artifacts for a specific user."""
    ensure_bucket(settings.minio_model_bucket)
    ensure_bucket(settings.minio_session_bucket)
    client = get_minio_client()

    artifacts = {
        "models": [],
        "sessions": []
    }

    try:
        # List model artifacts
        model_prefix = f"models/{user_id}/"
        # In minio 7.2.x, prefix must be a keyword argument
        model_objects = client.list_objects(
            bucket_name=settings.minio_model_bucket,
            prefix=model_prefix
        )
        for obj in model_objects:
            artifacts["models"].append({
                "object_key": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                "session_id": obj.object_name.split("/")[-1].replace(".zip", "")
            })
    except S3Error as exc:
        logger.warning("Failed to list model artifacts for user %s: %s", user_id, exc)

    try:
        # List session artifacts
        session_prefix = f"sessions/{user_id}/"
        # In minio 7.2.x, prefix must be a keyword argument
        session_objects = client.list_objects(
            bucket_name=settings.minio_session_bucket,
            prefix=session_prefix
        )
        for obj in session_objects:
            artifacts["sessions"].append({
                "object_key": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                "session_id": obj.object_name.split("/")[-1].replace(".zip", "")
            })
    except S3Error as exc:
        logger.warning("Failed to list session artifacts for user %s: %s", user_id, exc)

    return artifacts


def delete_artifact(bucket: str, object_key: Optional[str]) -> None:
    if not object_key:
        return
    client = get_minio_client()
    try:
        # In minio 7.2.x, use keyword arguments for clarity
        client.remove_object(
            bucket_name=bucket,
            object_name=object_key
        )
        logger.info("Deleted artifact %s from bucket %s", object_key, bucket)
    except S3Error as exc:
        logger.warning("Unable to delete object %s from %s: %s", object_key, bucket, exc)

