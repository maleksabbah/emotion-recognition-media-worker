# EmotionRecognitionMediaWorker
# Stateless Kafka consumer: MTCNN face detection, MediaPipe/InsightFace landmarks, region cropping
# Talks to: Kafka only (media_tasks → media_results)
# No HTTP port — scale with: docker compose up --scale media-worker=3

FROM python:3.11-slim

WORKDIR /app

# OpenCV + MediaPipe system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY main.py .

CMD ["python", "main.py"]
