"""
Media worker services.

  Detector       face detection (MTCNN + landmarks)  ← existing code
  Cropper        region cropping + base64 encoding   ← existing code
  MediaService   process_task(MediaTask) → MediaResult — orchestration
"""
from app.Services.MediaService import MediaService

__all__ = ["MediaService"]