"""
Media worker configuration — env vars loaded once.
"""
from __future__ import annotations

import os
import uuid


# ─── Identity ──────────────────────────────────────────────────────────

WORKER_ID = os.getenv("WORKER_ID", f"media-{uuid.uuid4().hex[:8]}")


# ─── Kafka ─────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
MEDIA_TASKS_TOPIC = os.getenv("MEDIA_TASKS_TOPIC", "media_tasks")
MEDIA_RESULTS_TOPIC = os.getenv("MEDIA_RESULTS_TOPIC", "media_results")
MEDIA_GROUP_ID = os.getenv("MEDIA_GROUP_ID", "media-workers")


# ─── Redis (live mode) ─────────────────────────────────────────────────

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}")


# ─── S3 / MinIO (batch mode) ──────────────────────────────────────────

S3_INTERNAL_ENDPOINT = os.getenv("S3_INTERNAL_ENDPOINT", "http://minio:9000")
S3_BUCKET = os.getenv("S3_BUCKET", "emotion")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_REGION = os.getenv("S3_REGION", "us-east-1")


# ─── Storage service (POST crops here for persistence) ───────────────

STORAGE_SERVICE_URL = os.getenv("STORAGE_SERVICE_URL", "http://storage:8002")


# ─── Live polling ──────────────────────────────────────────────────────

# Worker scans active sessions and pops from queue:frames:{sid}. The list
# of active sessions comes from a SCAN of the queue key prefix; the worker
# treats keys as ephemeral — drop when they disappear.
LIVE_BLPOP_TIMEOUT_SECONDS = float(os.getenv("LIVE_BLPOP_TIMEOUT_SECONDS", "0.5"))
LIVE_SCAN_INTERVAL_SECONDS = float(os.getenv("LIVE_SCAN_INTERVAL_SECONDS", "1.0"))