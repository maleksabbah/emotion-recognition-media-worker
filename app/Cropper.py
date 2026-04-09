"""
RegionCropper — extracts face crop (224x224) and 4 region crops (64x64)
from an image using detected landmarks.

Regions:
  - eyes:     bounding box around both eyes
  - mouth:    bounding box around mouth corners
  - cheeks:   bounding box around both cheeks
  - forehead: area above the eyes / top of face
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.Detector import FaceDetection, Landmark

logger = logging.getLogger("media-worker.cropper")

FACE_SIZE = 224
REGION_SIZE = 64


@dataclass
class RegionCrops:
    face: np.ndarray
    eyes: np.ndarray
    mouth: np.ndarray
    cheeks: np.ndarray
    forehead: np.ndarray


def _safe_crop(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """Crop a square region centered at (cx, cy), padded if out of bounds."""
    h, w = image.shape[:2]
    half = size // 2

    x1 = int(cx - half)
    y1 = int(cy - half)
    x2 = x1 + size
    y2 = y1 + size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    crop = image[y1:y2, x1:x2]

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REFLECT_101,
        )

    return cv2.resize(crop, (size, size))


def _midpoint(a: Landmark, b: Landmark) -> tuple[float, float]:
    return (a.x + b.x) / 2, (a.y + b.y) / 2


def extract_regions(
    image_bgr: np.ndarray,
    detection: FaceDetection,
) -> Optional[RegionCrops]:
    """
    Extract face + 4 region crops from a detected face.
    Returns None if essential landmarks are missing.
    """
    lm = detection.landmarks
    x1, y1, x2, y2 = detection.bbox
    h, w = image_bgr.shape[:2]

    # Face crop — expand bbox by 10% for context
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * 0.1), int(bh * 0.1)
    fx1 = max(0, x1 - pad_x)
    fy1 = max(0, y1 - pad_y)
    fx2 = min(w, x2 + pad_x)
    fy2 = min(h, y2 + pad_y)
    face_crop = cv2.resize(image_bgr[fy1:fy2, fx1:fx2], (FACE_SIZE, FACE_SIZE))

    # Region size based on face bbox
    face_w = x2 - x1
    region_span = max(int(face_w * 0.4), REGION_SIZE)

    # Eyes
    if "left_eye" in lm and "right_eye" in lm:
        eye_cx, eye_cy = _midpoint(lm["left_eye"], lm["right_eye"])
    else:
        eye_cx = (x1 + x2) / 2
        eye_cy = y1 + (y2 - y1) * 0.35
    eyes_crop = _safe_crop(image_bgr, eye_cx, eye_cy, region_span)
    eyes_crop = cv2.resize(eyes_crop, (REGION_SIZE, REGION_SIZE))

    # Mouth
    if "mouth_left" in lm and "mouth_right" in lm:
        mouth_cx, mouth_cy = _midpoint(lm["mouth_left"], lm["mouth_right"])
    else:
        mouth_cx = (x1 + x2) / 2
        mouth_cy = y1 + (y2 - y1) * 0.75
    mouth_crop = _safe_crop(image_bgr, mouth_cx, mouth_cy, region_span)
    mouth_crop = cv2.resize(mouth_crop, (REGION_SIZE, REGION_SIZE))

    # Cheeks
    if "left_cheek" in lm and "right_cheek" in lm:
        cheek_cx, cheek_cy = _midpoint(lm["left_cheek"], lm["right_cheek"])
    else:
        cheek_cx = (x1 + x2) / 2
        cheek_cy = y1 + (y2 - y1) * 0.55
    cheeks_crop = _safe_crop(image_bgr, cheek_cx, cheek_cy, region_span)
    cheeks_crop = cv2.resize(cheeks_crop, (REGION_SIZE, REGION_SIZE))

    # Forehead
    if "forehead" in lm:
        fh_cx, fh_cy = lm["forehead"].x, lm["forehead"].y
    else:
        fh_cx = (x1 + x2) / 2
        fh_cy = y1 + (y2 - y1) * 0.15
    forehead_crop = _safe_crop(image_bgr, fh_cx, fh_cy, region_span)
    forehead_crop = cv2.resize(forehead_crop, (REGION_SIZE, REGION_SIZE))

    return RegionCrops(
        face=face_crop,
        eyes=eyes_crop,
        mouth=mouth_crop,
        cheeks=cheeks_crop,
        forehead=forehead_crop,
    )


def crops_to_base64(crops: RegionCrops) -> dict[str, str]:
    """Encode all region crops as base64 JPEG strings for Kafka transport."""
    result = {}
    for name in ("face", "eyes", "mouth", "cheeks", "forehead"):
        img = getattr(crops, name)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result[name] = base64.b64encode(buf.tobytes()).decode("utf-8")
    return result