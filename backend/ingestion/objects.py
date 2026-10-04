"""Private, branch-scoped Neon Object Storage access via the S3 wire protocol."""

from functools import lru_cache
import hashlib
from pathlib import Path

from backend.core.config import settings


@lru_cache(maxsize=1)
def _client():
    import boto3
    from botocore.config import Config

    if not settings.NEON_S3_ENDPOINT or not settings.NEON_S3_BUCKET:
        raise RuntimeError("Neon Object Storage endpoint and bucket are required")
    if not settings.NEON_S3_ENDPOINT.startswith("https://"):
        raise ValueError("Neon Object Storage endpoint must use HTTPS")
    return boto3.client(
        "s3",
        endpoint_url=settings.NEON_S3_ENDPOINT,
        region_name=settings.NEON_S3_REGION,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def upload_source(path: Path, key: str, size_bytes: int, kind: str) -> None:
    content_type = {"pdf": "application/pdf", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text": "text/plain; charset=utf-8"}[kind]
    # PutObject is intentional: do not silently activate SDK multipart until a
    # disposable Neon branch proves initiate/complete/abort compatibility.
    with path.open("rb") as source:
        _client().put_object(Bucket=settings.NEON_S3_BUCKET, Key=key, Body=source,
                             ContentLength=size_bytes, ContentType=content_type)


def download_source(key: str, path: Path, expected_digest: str) -> None:
    response = _client().get_object(Bucket=settings.NEON_S3_BUCKET, Key=key)
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("wb") as target:
            while data := response["Body"].read(1024 * 1024):
                size += len(data)
                if size > settings.MAX_UPLOAD_BYTES:
                    raise ValueError("Stored object exceeds configured upload limit")
                digest.update(data)
                target.write(data)
        if digest.hexdigest() != expected_digest:
            raise ValueError("Stored object checksum mismatch")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        response["Body"].close()


def delete_source(key: str) -> None:
    _client().delete_object(Bucket=settings.NEON_S3_BUCKET, Key=key)
