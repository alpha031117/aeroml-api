"""Test script to verify MinIO connection and client initialization."""

import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import get_settings
from app.services.minio_service import get_minio_client, ensure_bucket
from app.helper.logger import get_logger

logger = get_logger()


def test_minio_connection():
    """Test MinIO client initialization and connection."""
    print("=" * 60)
    print("Testing MinIO Connection")
    print("=" * 60)

    # Test 1: Get settings
    print("\n[1] Loading configuration...")
    try:
        settings = get_settings()
        print(f"✓ Settings loaded successfully")
        print(f"  - Endpoint: {settings.minio_endpoint}")
        print(f"  - Access Key: {settings.minio_access_key[:4]}***")
        print(f"  - Secure: {settings.minio_secure}")
        print(f"  - Model Bucket: {settings.minio_model_bucket}")
        print(f"  - Session Bucket: {settings.minio_session_bucket}")
    except Exception as e:
        print(f"✗ Failed to load settings: {e}")
        return False

    # Test 2: Initialize MinIO client
    print("\n[2] Initializing MinIO client...")
    try:
        client = get_minio_client()
        print(f"✓ MinIO client initialized successfully")
        print(f"  - Client type: {type(client)}")
    except Exception as e:
        print(f"✗ Failed to initialize MinIO client: {e}")
        print(f"  - Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Test connection by listing buckets
    print("\n[3] Testing connection (listing buckets)...")
    try:
        buckets = client.list_buckets()
        print(f"✓ Connection successful! Found {len(buckets)} bucket(s):")
        for bucket in buckets:
            print(f"  - {bucket.name}")
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        print(f"  - Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Ensure buckets exist
    print("\n[4] Ensuring required buckets exist...")
    try:
        ensure_bucket(settings.minio_model_bucket)
        print(f"✓ Model bucket ready: {settings.minio_model_bucket}")

        ensure_bucket(settings.minio_session_bucket)
        print(f"✓ Session bucket ready: {settings.minio_session_bucket}")
    except Exception as e:
        print(f"✗ Failed to ensure buckets: {e}")
        print(f"  - Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed! MinIO is configured correctly.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_minio_connection()
    sys.exit(0 if success else 1)
