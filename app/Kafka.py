"""
Kafka helpers — consume media_tasks, publish to media_results.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from app.Config import (
    KAFKA_BOOTSTRAP,
    KAFKA_GROUP_ID,
    TOPIC_MEDIA_TASKS,
    TOPIC_MEDIA_RESULTS,
)

logger = logging.getLogger("media-worker.kafka")


async def create_consumer() -> AIOKafkaConsumer:
    consumer = AIOKafkaConsumer(
        TOPIC_MEDIA_TASKS,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=KAFKA_GROUP_ID,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        max_poll_interval_ms=300_000,
    )
    await consumer.start()
    logger.info("Kafka consumer started on topic=%s", TOPIC_MEDIA_TASKS)
    return consumer


async def create_producer() -> AIOKafkaProducer:
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    await producer.start()
    logger.info("Kafka producer started")
    return producer


async def publish_result(producer: AIOKafkaProducer, result: dict[str, Any]) -> None:
    await producer.send_and_wait(
        TOPIC_MEDIA_RESULTS,
        value=result,
        headers=[("worker_id", result.get("worker_id", "unknown").encode())],
    )
    logger.debug("Published result for session=%s frame=%s",
                 result.get("session_id"), result.get("frame_index"))