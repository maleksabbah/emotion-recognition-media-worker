# EmotionRecognitionMedia

Face detection and region cropping worker for the Mntis platform.
Consumes `MediaTask` messages, finds faces in the input image, crops
landmark-based regions (eyes / mouth / cheeks / forehead), and
publishes a `MediaResult` containing every crop ready for inference.

Built as a long-running Kafka consumer (no HTTP surface) + a Redis
poller for live-mode frames. Uses MTCNN, MediaPipe FaceMesh, and PIL.

## Architecture

The service follows a clean, layered architecture where each layer has
one responsibility and depends only on the layer beneath it:

* **Main loop** — `main.py` runs two coroutines side by side: a
  Kafka consumer on `media_tasks` (the batch path), and a Redis
  blpop loop that scans active live sessions and pulls their next
  enqueued task (the live path). Both paths funnel into the same
  service. Replaces what would otherwise be HTTP routes.
* **Services** — the business logic. `MediaService` fetches the
  input bytes (from MinIO for batch, from Redis for live), runs
  the detector, persists the crops via the storage service, and
  builds the typed `MediaResult` DTO. `FaceDetector` performs the
  detection + landmark + crop pipeline itself (MTCNN with 30%
  bbox padding → PIL resize to 128 → MediaPipe FaceMesh on the
  128 face → polygon bbox per region with 10px padding → PIL
  resize to 64 → JPEG quality 95 → base64). The cropping path
  matches the training pipeline byte-for-byte so the model sees
  the same input distribution it learned on. Knows nothing about
  Kafka or Redis.
* **Repositories** — data access. `S3Client` reads the source
  image from MinIO. `RedisRepository` reads live frames and
  acknowledges per-session task lists. `StorageClient` writes the
  region crops to MinIO via the storage service (one HTTP call
  per face uploads all five region crops as a bundle). The
  service layer never touches MinIO, Redis, or HTTP directly.
* **Dtos** — `MediaTask` (input — session_id, frame_source,
  frame_number) and `MediaResult` (output — faces with bboxes,
  landmark_tier, base64 crops, and crop s3_keys). Pydantic
  validates every message at the boundary.
* **Config** — wiring: Kafka consumer (group `media-workers`,
  reads `media_tasks`, writes `media_results`), Kafka producer,
  Redis client, httpx client, storage service URL, detector
  confidence threshold, model file paths for the
  `face_landmarker.task` MediaPipe asset.

This separation keeps the queue layer swappable, the business logic
testable in isolation, and the detector free to evolve without
touching the rest.

A few things worth calling out: the detector uses three fallback
tiers — MediaPipe FaceMesh (preferred, gives precise polygons),
InsightFace (5-point keypoints projected into rough boxes),
estimated bbox-fraction (last resort) — and tags each
`DetectedFace.landmark_tier` so downstream callers can filter on
quality. The 30% MTCNN bbox padding was the single biggest
accuracy fix in production: training data was pre-cropped public
dataset headshots, and a tight MTCNN crop on a real photo gave
the model a framing it had never seen.

Part of a multi-service system — see the [platform overview](../EmotionRecognitionDocker)
for the full architecture, pipeline flow, and the other services.
