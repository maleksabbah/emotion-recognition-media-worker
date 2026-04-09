"""
Media Worker configuration — loaded from environment variables.
"""
import os

# ── Kafka ──────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "media-worker-group")

TOPIC_MEDIA_TASKS = os.getenv("TOPIC_MEDIA_TASKS", "media_tasks")
TOPIC_MEDIA_RESULTS = os.getenv("TOPIC_MEDIA_RESULTS", "media_results")

# ── Redis ──────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── Face Detection ─────────────────────────────────────
FACE_CONFIDENCE = float(os.getenv("FACE_CONFIDENCE", "0.5"))
FACE_SIZE = int(os.getenv("FACE_SIZE", "224"))       # face crop size
REGION_SIZE = int(os.getenv("REGION_SIZE", "64"))     # region crop size

# ── Worker ─────────────────────────────────────────────
WORKER_ID = os.getenv("WORKER_ID", "media-worker-1")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))