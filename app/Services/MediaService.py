"""
MediaService — fetch frame → detect+crop → persist crops → build MediaResult.

One method, top-down.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from app.Config import WORKER_ID
from app.Dtos.TaskDto.MediaResult import (
    Bbox,
    DetectedFace,
    MediaResult,
    RegionCrops,
)
from app.Dtos.TaskDto.MediaTask import MediaTask
from app.Repositories.RedisRepository import RedisRepository
from app.Repositories.S3Client import S3Client
from app.Repositories.StorageClient import StorageClient
from app.Services.Detector import FaceDetector

logger = logging.getLogger("media-worker.service")


class MediaService:
    def __init__(
        self,
        detector: FaceDetector,
        redis: RedisRepository,
        s3: S3Client,
        storage: StorageClient,
    ):
        self.detector = detector
        self.redis = redis
        self.s3 = s3
        self.storage = storage

    async def process_task(self, task: MediaTask) -> Optional[MediaResult]:
        start = time.perf_counter()

        # 1. Fetch bytes (S3 for batch, Redis for live)
        source = task.frame_source
        if source.type == "s3":
            image_bytes = await self.s3.fetch_object(source.key)
        elif source.type == "redis":
            image_bytes = await self.redis.fetch_frame(task.session_id, source.key)
        else:
            logger.error("Unknown frame_source.type: %s", source.type)
            return None

        if image_bytes is None:
            logger.warning("No bytes for task %s", task.task_id)
            return None

        # 2. Detect + crop in one call (returns base64-encoded faces ready to send)
        faces_with_crops = self.detector.detect_and_crop(image_bytes)

        # 3. Persist crops via storage + wrap into DTOs
        faces: list[DetectedFace] = []
        for i, fc in enumerate(faces_with_crops):
            try:
                saved = await self.storage.save_crops(
                    session_id=task.session_id,
                    frame_index=task.frame_number,
                    detection_index=i,
                    crops_b64={
                        "face":     fc.face_crop,
                        "eyes":     fc.eyes,
                        "mouth":    fc.mouth,
                        "cheeks":   fc.cheeks,
                        "forehead": fc.forehead,
                    },
                )
                file_info = saved.get("file_ids", {}) or {}
                crop_s3_keys = {
                    region: meta.get("s3_key", "")
                    for region, meta in file_info.items()
                }
            except Exception as e:
                logger.error("save_crops failed for face %d: %s", i, e)
                crop_s3_keys = {}

            faces.append(DetectedFace(
                face_index=i,
                track_id=0,
                bbox=Bbox(
                    x=fc.detection.bbox[0],
                    y=fc.detection.bbox[1],
                    w=fc.detection.bbox[2] - fc.detection.bbox[0],
                    h=fc.detection.bbox[3] - fc.detection.bbox[1],
                ),
                # DTO requires Literal["mediapipe", "insightface", "fallback"].
                # `detector` is the composite "mtcnn+<tier>" used for logs;
                # `landmark_tier` is the clean enum value.
                landmark_tier=fc.detection.landmark_tier,
                face_crop=fc.face_crop,
                region_crops=RegionCrops(
                    eyes=fc.eyes,
                    mouth=fc.mouth,
                    cheeks=fc.cheeks,
                    forehead=fc.forehead,
                ),
                crop_s3_keys=crop_s3_keys,
            ))

        # 4. Build the result
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return MediaResult(
            task_id=task.task_id,
            session_id=task.session_id,
            frame_number=task.frame_number,
            timestamp_ms=0.0,
            faces=faces,
            worker_id=WORKER_ID,
            processing_time_ms=elapsed_ms,
        )




