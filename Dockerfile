# EmotionRecognitionMedia
# Kafka consumer + Redis live frame puller: decode → detect faces → crop regions
# Talks to: Kafka, Redis (live), MinIO (batch), Storage (HTTP for save_crops)
# Scale with: docker compose up --scale media-worker=N

FROM python:3.11-slim

WORKDIR /app

# Runtime + build deps:
#   libgl1, libglib2.0-0, libsm6, libxrender1, libxext6 → OpenCV / mediapipe
#   ffmpeg                                              → video decoding
#   curl                                                → fetch face_landmarker.task model
#   build-essential, python3-dev                        → compile insightface (Cython extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    curl \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Mediapipe Tasks API requires a downloaded .task model file.
# The legacy mp.solutions.face_mesh API was removed in newer mediapipe;
# we use the new mp.tasks.vision.FaceLandmarker which loads from a file.
RUN mkdir -p /app/models && \
    curl -fsSL -o /app/models/face_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

COPY app/ ./app/
COPY main.py .

ENV FACE_LANDMARKER_TASK=/app/models/face_landmarker.task

CMD ["python", "main.py"]