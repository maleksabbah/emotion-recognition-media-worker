# EmotionRecognitionMedia
# Kafka consumer + Redis live frame puller: decode → detect faces → crop regions
# Talks to: Kafka, Redis (live), MinIO (batch), Storage (HTTP for save_crops)
# Scale with: docker compose up --scale media-worker=N

FROM python:3.11-slim

WORKDIR /app

# OpenCV + mediapipe + ffmpeg system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY main.py .

CMD ["python", "main.py"]