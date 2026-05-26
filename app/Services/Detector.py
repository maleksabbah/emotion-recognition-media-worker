"""
FaceDetector — MTCNN for face detection + MediaPipe FaceMesh polygon
region cropping that matches the training pipeline exactly.

Pipeline:
  1. MTCNN finds faces -> bounding boxes
  2. For each face, crop face_bbox -> resize to 128x128 (training scale)
  3. Run MediaPipe FaceMesh on the 128 crop, gather full landmark polygons
     for eyes / mouth / cheeks / forehead
  4. Each region = bounding box of its landmark polygon + 10px pad
  5. Resize face to 224 (Predictor expects 224), regions to 64
  6. JPEG-encode + base64

Mirrors `Face_Service._extract_regions_landmarks` from training (same
landmark index sets, same padding). If MediaPipe fails, fall back to
InsightFace 5-point landmarks projected to rough rectangles; if THAT
fails, bbox-fraction estimates. Each tier marks the detection so callers
can filter on quality.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("media-worker.detector")

FACE_SIZE = 224
TRAIN_FACE = 128
REGION_SIZE = 64
REGION_PAD = 10


# ── Landmark index sets — verbatim from training Face_Service ──

LEFT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
]
MOUTH = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95,
]
LEFT_CHEEK = [116, 117, 118, 119, 100, 126, 209, 49, 129, 203]
RIGHT_CHEEK = [345, 346, 347, 348, 329, 355, 429, 279, 358, 423]
FOREHEAD = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]

REGION_LANDMARKS = {
    "eyes":     LEFT_EYE + RIGHT_EYE,
    "mouth":    MOUTH,
    "cheeks":   LEFT_CHEEK + RIGHT_CHEEK,
    "forehead": FOREHEAD,
}


@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    landmark_tier: str = "fallback"   # mediapipe | insightface | fallback
    detector: str = "unknown"


@dataclass
class FaceWithCrops:
    detection: FaceDetection
    face_crop: str
    eyes: str
    mouth: str
    cheeks: str
    forehead: str


class FaceDetector:
    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self._mtcnn = None
        self._mp = None
        self._insight = None

    # ── MTCNN ──────────────────────────────────────────

    def _get_mtcnn(self):
        if self._mtcnn is None:
            try:
                from mtcnn import MTCNN
                self._mtcnn = MTCNN(min_face_size=40)
                logger.info("MTCNN loaded")
            except ImportError:
                logger.warning("MTCNN not installed")
                self._mtcnn = False
        return self._mtcnn if self._mtcnn is not False else None

    def _detect_faces_mtcnn(self, image_rgb: np.ndarray) -> list[dict]:
        mtcnn = self._get_mtcnn()
        if mtcnn is None:
            return []
        results = mtcnn.detect_faces(image_rgb)
        faces = []
        for r in results:
            if r["confidence"] < self.min_confidence:
                continue
            x, y, w, h = r["box"]
            faces.append({
                "bbox": (max(0, x), max(0, y), x + w, y + h),
                "confidence": r["confidence"],
            })
        return faces

    # ── MediaPipe (polygon landmarks) ──────────────────

    def _get_mediapipe(self):
        if self._mp is None:
            try:
                import mediapipe as mp
                self._mp = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.3,
                )
                logger.info("MediaPipe FaceMesh loaded")
            except ImportError:
                logger.warning("MediaPipe not installed")
                self._mp = False
        return self._mp if self._mp is not False else None

    def _region_boxes_from_mediapipe(
        self, face128_rgb: np.ndarray,
    ) -> Optional[dict[str, tuple[int, int, int, int]]]:
        mp_detector = self._get_mediapipe()
        if mp_detector is None:
            return None
        results = mp_detector.process(face128_rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = face128_rgb.shape[:2]
        boxes: dict[str, tuple[int, int, int, int]] = {}
        for region, indices in REGION_LANDMARKS.items():
            xs = [int(landmarks[i].x * w) for i in indices]
            ys = [int(landmarks[i].y * h) for i in indices]
            if not xs or not ys:
                continue
            rx1 = max(0, min(xs) - REGION_PAD)
            ry1 = max(0, min(ys) - REGION_PAD)
            rx2 = min(w, max(xs) + REGION_PAD)
            ry2 = min(h, max(ys) + REGION_PAD)
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            boxes[region] = (rx1, ry1, rx2, ry2)
        if len(boxes) < 4:
            return None
        return boxes

    # ── InsightFace fallback ───────────────────────────

    def _get_insightface(self):
        if self._insight is None:
            try:
                from insightface.app import FaceAnalysis
                self._insight = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CPUExecutionProvider"],
                )
                self._insight.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("InsightFace loaded")
            except (ImportError, Exception) as e:
                logger.warning("InsightFace unavailable: %s", e)
                self._insight = False
        return self._insight if self._insight is not False else None

    def _region_boxes_from_insightface(
        self, face128_bgr: np.ndarray,
    ) -> Optional[dict[str, tuple[int, int, int, int]]]:
        app = self._get_insightface()
        if app is None:
            return None
        faces = app.get(face128_bgr)
        if not faces or faces[0].kps is None or len(faces[0].kps) < 5:
            return None
        kps = faces[0].kps
        h, w = face128_bgr.shape[:2]

        def box_around(cx, cy, side):
            half = side // 2
            return (
                max(0, int(cx - half)), max(0, int(cy - half)),
                min(w, int(cx + half)), min(h, int(cy + half)),
            )

        eye_cx = (kps[0][0] + kps[1][0]) / 2
        eye_cy = (kps[0][1] + kps[1][1]) / 2
        eye_side = max(int(abs(kps[1][0] - kps[0][0]) * 1.6), REGION_SIZE)
        mouth_cx = (kps[3][0] + kps[4][0]) / 2
        mouth_cy = (kps[3][1] + kps[4][1]) / 2
        mouth_side = max(int(abs(kps[4][0] - kps[3][0]) * 1.6), REGION_SIZE)

        return {
            "eyes":     box_around(eye_cx, eye_cy, eye_side),
            "mouth":    box_around(mouth_cx, mouth_cy, mouth_side),
            "cheeks":   box_around(kps[2][0], kps[2][1] + h * 0.1, int(w * 0.55)),
            "forehead": box_around(eye_cx, eye_cy - h * 0.2, int(w * 0.6)),
        }

    # ── Estimated fallback ─────────────────────────────

    @staticmethod
    def _region_boxes_estimated(face_w: int, face_h: int) -> dict[str, tuple[int, int, int, int]]:
        def b(cx_f, cy_f, w_f, h_f):
            cx = face_w * cx_f
            cy = face_h * cy_f
            hw = (face_w * w_f) / 2
            hh = (face_h * h_f) / 2
            return (
                max(0, int(cx - hw)), max(0, int(cy - hh)),
                min(face_w, int(cx + hw)), min(face_h, int(cy + hh)),
            )
        return {
            "eyes":     b(0.50, 0.36, 0.70, 0.20),
            "mouth":    b(0.50, 0.77, 0.45, 0.20),
            "cheeks":   b(0.50, 0.58, 0.78, 0.22),
            "forehead": b(0.50, 0.16, 0.70, 0.20),
        }

    # ── Public API ─────────────────────────────────────

    def detect_and_crop(self, image_bytes: bytes) -> list[FaceWithCrops]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            logger.error("Failed to decode image bytes")
            return []

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_bgr.shape[:2]

        faces = self._detect_faces_mtcnn(image_rgb)
        if not faces:
            logger.debug("MTCNN found nothing — falling back to center crop")
            faces = [self._fallback_centered_face(img_w, img_h)]

        results: list[FaceWithCrops] = []
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            face_bgr_128 = cv2.resize(image_bgr[y1:y2, x1:x2], (TRAIN_FACE, TRAIN_FACE))
            face_rgb_128 = cv2.cvtColor(face_bgr_128, cv2.COLOR_BGR2RGB)

            tier = "mediapipe"
            region_boxes = self._region_boxes_from_mediapipe(face_rgb_128)
            if region_boxes is None:
                region_boxes = self._region_boxes_from_insightface(face_bgr_128)
                tier = "insightface"
            if region_boxes is None:
                region_boxes = self._region_boxes_estimated(TRAIN_FACE, TRAIN_FACE)
                tier = "fallback"

            crops: dict[str, np.ndarray] = {}
            for region, (rx1, ry1, rx2, ry2) in region_boxes.items():
                patch = face_bgr_128[ry1:ry2, rx1:rx2]
                if patch.size == 0:
                    patch = face_bgr_128
                crops[region] = cv2.resize(patch, (REGION_SIZE, REGION_SIZE))
            crops["face"] = cv2.resize(face_bgr_128, (FACE_SIZE, FACE_SIZE))

            b64: dict[str, str] = {}
            for name, img in crops.items():
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                b64[name] = base64.b64encode(buf.tobytes()).decode("utf-8")

            det = FaceDetection(
                bbox=(x1, y1, x2, y2),
                confidence=face["confidence"],
                landmark_tier=tier,
                detector=f"mtcnn+{tier}",
            )
            results.append(FaceWithCrops(
                detection=det,
                face_crop=b64["face"],
                eyes=b64["eyes"],
                mouth=b64["mouth"],
                cheeks=b64["cheeks"],
                forehead=b64["forehead"],
            ))

        return results

    @staticmethod
    def _fallback_centered_face(img_w: int, img_h: int) -> dict:
        side = min(img_h, img_w)
        cx, cy = img_w // 2, img_h // 2
        half = side // 2
        return {
            "bbox": (cx - half, cy - half, cx + half, cy + half),
            "confidence": 0.1,
        }