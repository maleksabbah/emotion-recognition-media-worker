"""
S3Client — boto3 wrapper to fetch source media (uploaded video/photo) for
the batch path.

Sync boto3 wrapped in asyncio.to_thread so the event loop stays free.
"""
from __future__ import annotations

import asyncio
import logging

import boto3
from botocore.client import Config as BotoConfig

from app.Config import (
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_INTERNAL_ENDPOINT,
    S3_REGION,
    S3_SECRET_KEY,
)

logger = logging.getLogger("media-worker.s3")


class S3Client:
    def __init__(self) -> None:
        self._client = boto3.client(
            "s3",
            endpoint_url=S3_INTERNAL_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name=S3_REGION,
            config=BotoConfig(signature_version="s3v4"),
        )

    async def fetch_object(self, s3_key: str) -> bytes:
        def _go() -> bytes:
            resp = self._client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            return resp["Body"].read()
        return await asyncio.to_thread(_go)