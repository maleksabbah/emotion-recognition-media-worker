"""
StorageClient — media worker → storage HTTP boundary.

POSTs base64 crops for one face to /internal/save-crops. Storage decodes,
uploads to MinIO, creates 5 FileRecord rows, returns the s3_keys.
"""
from __future__ import annotations

import logging

import httpx

from app.Config import STORAGE_SERVICE_URL

logger = logging.getLogger("media-worker.storage")


class StorageClient:
    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))

    async def stop(self) -> None:
        if self._http:
            await self._http.aclose()

    async def save_crops(
        self,
        session_id: str,
        frame_index: int,
        detection_index: int,
        crops_b64: dict[str, str],
    ) -> dict:
        """Returns {file_ids: {region: {file_id, s3_key}}}."""
        if not self._http:
            raise RuntimeError("Call start() first")

        body = {
            "session_id": session_id,
            "frame_index": frame_index,
            "detection_index": detection_index,
            "crops": crops_b64,
        }
        resp = await self._http.post(
            f"{STORAGE_SERVICE_URL}/internal/save-crops", json=body
        )
        if resp.status_code != 200:
            logger.error("save_crops failed %s: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        return resp.json()