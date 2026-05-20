"""
Media worker integration tests — single file.
Run from media-worker/ root: `pytest Test.py -v`

Requires:
  pytest.ini with session-scoped loops (see storage)
"""
from __future__ import annotations

import io
import os
import uuid
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio
import redis.asyncio as aioredis
from PIL import Image
from testcontainers.minio import MinioContainer
from testcontainers.redis import RedisContainer


# ══════════════════════════════════════════════
# Containers
# ══════════════════════════════════════════════

@pytest.fixture(scope="session", autouse=True)
def redis_url():
    with RedisContainer() as rc:
        url = f"redis://{rc.get_container_host_ip()}:{rc.get_exposed_port(6379)}"
        os.environ["REDIS_URL"] = url
        yield url


@pytest.fixture(scope="session", autouse=True)
def minio():
    with MinioContainer() as mc:
        host = mc.get_container_host_ip()
        port = mc.get_exposed_port(9000)
        os.environ["S3_INTERNAL_ENDPOINT"] = f"http://{host}:{port}"
        os.environ["S3_ACCESS_KEY"] = mc.access_key
        os.environ["S3_SECRET_KEY"] = mc.secret_key
        os.environ.setdefault("STORAGE_SERVICE_URL", "http://fake-storage:8002")
        client = mc.get_client()
        if not client.bucket_exists("emotion"):
            client.make_bucket("emotion")
        yield mc


# ══════════════════════════════════════════════
# Session-scoped service (Detector model load is slow)
# ══════════════════════════════════════════════

@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def media_service():
    from app.Repositories.RedisRepository import RedisRepository
    from app.Repositories.S3Client import S3Client
    from app.Services.Detector import FaceDetector
    from app.Services.MediaService import MediaService

    redis = RedisRepository()
    await redis.start()

    storage = AsyncMock()
    storage.save_crops = AsyncMock(return_value={
        "file_ids": {r: {"file_id": str(uuid.uuid4()), "s3_key": f"x/{r}.jpg"}
                     for r in ("face", "eyes", "mouth", "cheeks", "forehead")},
    })

    svc = MediaService(
        detector=FaceDetector(),
        redis=redis,
        s3=S3Client(),
        storage=storage,
    )
    yield svc
    await redis.stop()


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def _test_image() -> bytes:
    arr = np.full((300, 300, 3), 255, dtype=np.uint8)
    yy, xx = np.ogrid[:300, :300]
    mask = ((yy - 150) ** 2 / 80**2 + (xx - 150) ** 2 / 60**2) <= 1
    arr[mask] = [220, 180, 150]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


# ══════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════

@pytest.mark.asyncio(loop_scope="session")
async def test_batch_path_processes_s3_source(media_service, minio):
    from app.Dtos.TaskDto.MediaTask import FrameSource, MediaTask

    img = _test_image()
    s3_key = f"sessions/{uuid.uuid4()}/source/img.jpg"
    minio.get_client().put_object(
        "emotion", s3_key, io.BytesIO(img), len(img), content_type="image/jpeg",
    )

    task = MediaTask(
        task_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        mode="video",
        frame_number=0,
        frame_source=FrameSource(type="s3", key=s3_key),
    )
    result = await media_service.process_task(task)

    assert result is not None
    assert result.task_id == task.task_id
    assert isinstance(result.faces, list)


@pytest.mark.asyncio(loop_scope="session")
async def test_live_path_processes_redis_frame(media_service, redis_url):
    from app.Dtos.TaskDto.MediaTask import FrameSource, MediaTask

    sid = str(uuid.uuid4())
    fid = uuid.uuid4().hex

    r = aioredis.from_url(redis_url)
    await r.set(f"frame:{sid}:{fid}", _test_image(), ex=30)
    await r.close()

    task = MediaTask(
        task_id=str(uuid.uuid4()),
        session_id=sid,
        mode="live",
        frame_number=0,
        frame_source=FrameSource(type="redis", key=fid),
    )
    result = await media_service.process_task(task)

    assert result is not None
    assert result.session_id == sid


@pytest.mark.asyncio(loop_scope="session")
async def test_missing_source_returns_none(media_service):
    from app.Dtos.TaskDto.MediaTask import FrameSource, MediaTask

    task = MediaTask(
        task_id=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        mode="live",
        frame_number=0,
        frame_source=FrameSource(type="redis", key="does-not-exist"),
    )
    result = await media_service.process_task(task)
    assert result is None