"""
Media Worker — stateless Kafka consumer.

Consumes frames from media_tasks, runs face detection + region cropping,
publishes results to media_results.

Task schema (from orchestrator):
{
    "session_id": str,
    "frame_index": int,
    "image_b64": str,          # base64-encoded JPEG/PNG
    "mode": "live" | "upload",
    "priority": int
}

Result schema (published to media_results):
{
    "session_id": str,
    "frame_index": int,
    "worker_id": str,
    "status": "success" | "no_face" | "error",
    "detections": [
        {
            "detection_index": int,
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "detector": str,
            "landmarks": { name: {x, y} },
            "crops": { face: b64, eyes: b64, mouth: b64, cheeks: b64, forehead: b64 }
        }
    ],
    "error": str | null
}
"""
from __future__ import annotations

import asyncio
import base64
import logging
import signal
import time
from typing import Any

import cv2
import numpy as np

from app.Config import WORKER_ID, MAX_RETRIES
from app.Kafka import create_consumer, create_producer, publish_result
from app.Detector import FaceDetector
from app.Cropper import extract_regions, crops_to_base64

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("media-worker")


class MediaWorker:
    def __init__(self):
        self.detector = FaceDetector()
        self.consumer = None
        self.producer = None
        self._running = False

    async def start(self):
        self.consumer = await create_consumer()
        self.producer = await create_producer()
        self._running = True
        logger.info("Media worker %s started", WORKER_ID)

    async def stop(self):
        self._running = False
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()
        logger.info("Media worker %s stopped", WORKER_ID)

    def _decode_image(self, image_b64: str) -> np.ndarray:
        """Decode base64 image to BGR numpy array."""
        img_bytes = base64.b64decode(image_b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image

    def _process_frame(self, image_b64: str) -> dict[str, Any]:
        """
        Synchronous processing: detect faces → crop regions.
        Returns the detections payload.
        """
        image = self._decode_image(image_b64)
        faces = self.detector.detect(image)

        if not faces:
            return {"status": "no_face", "detections": []}

        detections = []
        for i, face in enumerate(faces):
            regions = extract_regions(image, face)
            if regions is None:
                continue

            crop_data = crops_to_base64(regions)
            landmarks_dict = {
                name: {"x": round(lm.x, 2), "y": round(lm.y, 2)}
                for name, lm in face.landmarks.items()
            }

            detections.append({
                "detection_index": i,
                "bbox": list(face.bbox),
                "confidence": round(face.confidence, 4),
                "detector": face.detector,
                "landmarks": landmarks_dict,
                "crops": crop_data,
            })

        status = "success" if detections else "no_face"
        return {"status": status, "detections": detections}

    async def _handle_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process a single task and return the result message."""
        session_id = task.get("session_id", "unknown")
        frame_index = task.get("frame_index", -1)

        start = time.perf_counter()

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._process_frame, task["image_b64"]
            )
            elapsed = round((time.perf_counter() - start) * 1000, 1)

            return {
                "session_id": session_id,
                "frame_index": frame_index,
                "worker_id": WORKER_ID,
                "status": result["status"],
                "detections": result["detections"],
                "processing_ms": elapsed,
                "error": None,
            }

        except Exception as e:
            elapsed = round((time.perf_counter() - start) * 1000, 1)
            logger.error("Error processing session=%s frame=%d: %s",
                         session_id, frame_index, e)
            return {
                "session_id": session_id,
                "frame_index": frame_index,
                "worker_id": WORKER_ID,
                "status": "error",
                "detections": [],
                "processing_ms": elapsed,
                "error": str(e),
            }

    async def run(self):
        """Main consumer loop."""
        await self.start()

        try:
            async for message in self.consumer:
                if not self._running:
                    break

                task = message.value
                logger.info(
                    "Received task: session=%s frame=%s",
                    task.get("session_id"), task.get("frame_index"),
                )

                result = await self._handle_task(task)
                await publish_result(self.producer, result)
                await self.consumer.commit()

        except asyncio.CancelledError:
            logger.info("Worker loop cancelled")
        finally:
            await self.stop()


async def main():
    worker = MediaWorker()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(worker)))

    await worker.run()


async def _shutdown(worker: MediaWorker):
    logger.info("Shutdown signal received")
    worker._running = False


if __name__ == "__main__":
    asyncio.run(main())