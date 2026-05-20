"""
Media worker entry point.

Two concurrent loops, both feeding into MediaService.process_task:
  - kafka_loop: pull MediaTask from media_tasks topic   (batch path)
  - redis_loop: scan active sessions, blpop their queue (live path)

Results always publish to Kafka (media_results), so orchestrator's
PipelineService handles both modes uniformly.
"""
from __future__ import annotations

import asyncio
import logging
import signal

from app.Config import LIVE_SCAN_INTERVAL_SECONDS, WORKER_ID
from app.Dtos.TaskDto.MediaTask import MediaTask
from app.Repositories.KafkaConsumer import KafkaConsumer
from app.Repositories.KafkaProducer import KafkaProducer
from app.Repositories.RedisRepository import RedisRepository
from app.Repositories.S3Client import S3Client
from app.Repositories.StorageClient import StorageClient
from app.Services.Detector import FaceDetector
from app.Services.MediaService import MediaService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("media-worker")


# ─── Loop bodies ──────────────────────────────────────────────────────

async def kafka_loop(
    consumer: KafkaConsumer,
    producer: KafkaProducer,
    media: MediaService,
) -> None:
    async for raw in consumer.consume():
        try:
            task = MediaTask.model_validate(raw)
        except Exception as e:
            logger.error("Bad MediaTask from Kafka: %s", e)
            continue
        await _handle(task, media, producer)


async def redis_loop(
    redis_repo: RedisRepository,
    producer: KafkaProducer,
    media: MediaService,
) -> None:
    while True:
        sessions = await redis_repo.scan_active_sessions()
        if not sessions:
            await asyncio.sleep(LIVE_SCAN_INTERVAL_SECONDS)
            continue

        # Round-robin: pop one from each active session per iteration.
        for sid in sessions:
            raw = await redis_repo.dequeue_media_task(sid)
            if raw is None:
                continue
            try:
                task = MediaTask.model_validate(raw)
            except Exception as e:
                logger.error("Bad MediaTask from Redis: %s", e)
                continue
            await _handle(task, media, producer)


async def _handle(
    task: MediaTask, media: MediaService, producer: KafkaProducer
) -> None:
    try:
        result = await media.process_task(task)
    except Exception as e:
        logger.exception("process_task failed: %s", e)
        return
    if result is None:
        return
    await producer.publish_media_result(result.model_dump())


# ─── Entry ────────────────────────────────────────────────────────────

async def run() -> None:
    logger.info("Media worker starting (id=%s)", WORKER_ID)

    detector = FaceDetector()
    redis_repo = RedisRepository()
    s3 = S3Client()
    storage = StorageClient()
    consumer = KafkaConsumer()
    producer = KafkaProducer()

    await redis_repo.start()
    await storage.start()
    await consumer.start()
    await producer.start()
    logger.info("Worker ready")

    media = MediaService(detector=detector, redis=redis_repo, s3=s3, storage=storage)

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    kafka_task = asyncio.create_task(kafka_loop(consumer, producer, media))
    redis_task = asyncio.create_task(redis_loop(redis_repo, producer, media))

    await stop.wait()
    logger.info("Shutting down...")

    kafka_task.cancel()
    redis_task.cancel()
    for t in (kafka_task, redis_task):
        try:
            await t
        except asyncio.CancelledError:
            pass

    await consumer.stop()
    await producer.stop()
    await redis_repo.stop()
    await storage.stop()
    logger.info("Worker stopped")


if __name__ == "__main__":
    asyncio.run(run())