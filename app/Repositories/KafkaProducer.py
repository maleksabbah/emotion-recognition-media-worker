"""
KafkaProducer — publishes MediaResult dicts to media_results topic.
"""
from __future__ import annotations

import ssl
import json
import logging

from aiokafka import AIOKafkaProducer

from app.Config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_SASL_USERNAME,
    KAFKA_SASL_PASSWORD,
    MEDIA_RESULTS_TOPIC,
)

logger = logging.getLogger("media-worker.kafka-producer")


class KafkaProducer:
    def __init__(self) -> None:
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        kafka_kwargs = {"bootstrap_servers": KAFKA_BOOTSTRAP_SERVERS}
        if KAFKA_SASL_USERNAME:
            kafka_kwargs.update({
                "security_protocol": "SASL_SSL",
                "ssl_context": ssl.create_default_context(),
                "sasl_mechanism": "SCRAM-SHA-256",
                "sasl_plain_username": KAFKA_SASL_USERNAME,
                "sasl_plain_password": KAFKA_SASL_PASSWORD,
            })
        self._producer = AIOKafkaProducer(**kafka_kwargs)
        await self._producer.start()

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()

    async def publish_media_result(self, payload: dict) -> None:
        if not self._producer:
            raise RuntimeError("Call start() before publish_*")
        try:
            await self._producer.send_and_wait(
                MEDIA_RESULTS_TOPIC, json.dumps(payload).encode("utf-8")
            )
        except Exception as e:
            logger.error("Kafka publish failed: %s", e)
            raise