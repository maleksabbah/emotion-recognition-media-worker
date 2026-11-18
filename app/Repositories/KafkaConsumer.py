"""
KafkaConsumer — pulls media task messages.

Yields decoded dicts. Service layer validates them as MediaTask via Pydantic.
"""
from __future__ import annotations

import ssl
import json
import logging
from typing import AsyncIterator

from aiokafka import AIOKafkaConsumer

from app.Config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_SASL_USERNAME,
    KAFKA_SASL_PASSWORD,
    MEDIA_GROUP_ID,
    MEDIA_TASKS_TOPIC,
)

logger = logging.getLogger("media-worker.kafka-consumer")


class KafkaConsumer:
    def __init__(self) -> None:
        self._consumer: AIOKafkaConsumer | None = None

    async def start(self) -> None:
        kafka_kwargs = {
            "bootstrap_servers": KAFKA_BOOTSTRAP_SERVERS,
            "group_id": MEDIA_GROUP_ID,
            "enable_auto_commit": False,
            "auto_offset_reset": "earliest",
            "session_timeout_ms": 300000,
            "max_poll_interval_ms": 1800000,
        }
        if KAFKA_SASL_USERNAME:
            kafka_kwargs.update({
                "security_protocol": "SASL_SSL",
                "ssl_context": ssl.create_default_context(),
                "sasl_mechanism": "SCRAM-SHA-256",
                "sasl_plain_username": KAFKA_SASL_USERNAME,
                "sasl_plain_password": KAFKA_SASL_PASSWORD,
            })
        self._consumer = AIOKafkaConsumer(MEDIA_TASKS_TOPIC, **kafka_kwargs)
        await self._consumer.start()

    async def stop(self) -> None:
        if self._consumer:
            await self._consumer.stop()

    async def commit(self) -> None:
        if self._consumer:
            await self._consumer.commit()

    async def consume(self) -> AsyncIterator[dict]:
        if not self._consumer:
            raise RuntimeError("Call start() before consume()")
        async for record in self._consumer:
            try:
                yield json.loads(record.value.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.error("Bad JSON from %s: %s", MEDIA_TASKS_TOPIC, e)
                continue