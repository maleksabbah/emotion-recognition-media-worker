"""
RedisRepository — live-mode transport.

Live path:
  - scan_active_sessions() → list of session_ids with pending frames
  - dequeue_media_task(session_id) → MediaTask dict (popped from queue:frames:{sid})
  - fetch_frame(session_id, frame_id) → bytes (from frame:{sid}:{fid})
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import redis.asyncio as redis

from app.Config import LIVE_BLPOP_TIMEOUT_SECONDS, REDIS_URL

logger = logging.getLogger("media-worker.redis")

_FRAME_PREFIX = "frame"
_QUEUE_PREFIX = "queue:frames"


class RedisRepository:
    def __init__(self) -> None:
        self._r: redis.Redis | None = None

    async def start(self) -> None:
        self._r = redis.from_url(REDIS_URL, decode_responses=False)

    async def stop(self) -> None:
        if self._r:
            await self._r.close()

    # ── Discovery ────────────────────────────────────

    async def scan_active_sessions(self) -> list[str]:
        """Find all sessions with pending frames in their queue."""
        if not self._r:
            raise RuntimeError("Call start() first")

        sessions: list[str] = []
        async for key in self._r.scan_iter(match=f"{_QUEUE_PREFIX}:*"):
            key_str = key.decode() if isinstance(key, bytes) else key
            sessions.append(key_str[len(_QUEUE_PREFIX) + 1:])
        return sessions

    # ── Pop next task ────────────────────────────────

    async def dequeue_media_task(self, session_id: str) -> Optional[dict]:
        if not self._r:
            raise RuntimeError("Call start() first")
        raw = await self._r.blpop(
            self._queue_key(session_id), timeout=LIVE_BLPOP_TIMEOUT_SECONDS
        )
        if raw is None:
            return None
        _, payload = raw
        return json.loads(payload)

    # ── Fetch bytes ──────────────────────────────────

    async def fetch_frame(self, session_id: str, frame_id: str) -> Optional[bytes]:
        if not self._r:
            raise RuntimeError("Call start() first")
        return await self._r.get(self._frame_key(session_id, frame_id))

    # ── Keys ─────────────────────────────────────────

    @staticmethod
    def _frame_key(session_id: str, frame_id: str) -> str:
        return f"{_FRAME_PREFIX}:{session_id}:{frame_id}"

    @staticmethod
    def _queue_key(session_id: str) -> str:
        return f"{_QUEUE_PREFIX}:{session_id}"