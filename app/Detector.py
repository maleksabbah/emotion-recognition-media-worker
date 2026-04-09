"""
FaceDetector — MTCNN for face detection + landmark extraction.

Pipeline:
  1. MTCNN detects faces → bounding boxes
  2. MediaPipe FaceMesh extracts landmarks on each detected face (9 key points)
  3. If MediaPipe fails → InsightFace for 5 landmarks
  4. If both fail → estimate landmarks from bounding box
  5. If MTCNN finds nothing → fallback center crop
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("media-worker.detector")


@dataclass
class Landmark:
    x: float
    y: float


@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]           # x1, y1, x2, y2
    confidence: float
    landmarks: dict[str, Landmark] = field(default_factory=dict)
    detector: str = "unknown"


class FaceDetector:
    """
    MTCNN for face detection, MediaPipe/InsightFace for landmark extraction.
    Each dependency is lazy-loaded on first use.
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self._mtcnn: Optional[object] = None
        self._mp_detector: Optional[object] = None
        self._insight_app: Optional[object] = None

    # ── MTCNN (face detection) ─────────────────────────

    def _get_mtcnn(self):
        if self._mtcnn is None:
            try:
                from mtcnn import MTCNN
                self._mtcnn = MTCNN(min_face_size=40)
                logger.info("MTCNN loaded")
            except ImportError:
                logger.warning("MTCNN not installed — face detection unavailable")
                self._mtcnn = False
        return self._mtcnn if self._mtcnn is not False else None

    def _detect_faces_mtcnn(self, image_rgb: np.ndarray) -> list[dict]:
        """
        Detect faces using MTCNN.
        Returns list of {bbox: (x1,y1,x2,y2), confidence: float}
        """
        mtcnn = self._get_mtcnn()
        if mtcnn is None:
            return []

        results = mtcnn.detect_faces(image_rgb)
        faces = []
        for r in results:
            if r["confidence"] < self.min_confidence:
                continue
            x, y, w, h = r["box"]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = x + w
            y2 = y + h
            faces.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": r["confidence"],
            })

        return faces

    # ── MediaPipe (landmark extraction) ────────────────

    def _get_mediapipe(self):
        if self._mp_detector is None:
            try:
                import mediapipe as mp
                self._mp_detector = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.3,
                )
                logger.info("MediaPipe FaceMesh loaded")
            except ImportError:
                logger.warning("MediaPipe not installed — skipping landmark tier 1")
                self._mp_detector = False
        return self._mp_detector if self._mp_detector is not False else None

    def _extract_landmarks_mediapipe(
        self, face_crop_rgb: np.ndarray, offset_x: int, offset_y: int
    ) -> dict[str, Landmark]:
        """
        Run MediaPipe on a cropped face region.
        Returns landmarks in original image coordinates (offset applied).
        """
        detector = self._get_mediapipe()
        if detector is None:
            return {}

        results = detector.process(face_crop_rgb)
        if not results.multi_face_landmarks:
            return {}

        face_lm = results.multi_face_landmarks[0]
        h, w = face_crop_rgb.shape[:2]

        return {
            "left_eye": Landmark(
                face_lm.landmark[33].x * w + offset_x,
                face_lm.landmark[33].y * h + offset_y,
            ),
            "right_eye": Landmark(
                face_lm.landmark[263].x * w + offset_x,
                face_lm.landmark[263].y * h + offset_y,
            ),
            "nose_tip": Landmark(
                face_lm.landmark[1].x * w + offset_x,
                face_lm.landmark[1].y * h + offset_y,
            ),
            "mouth_left": Landmark(
                face_lm.landmark[61].x * w + offset_x,
                face_lm.landmark[61].y * h + offset_y,
            ),
            "mouth_right": Landmark(
                face_lm.landmark[291].x * w + offset_x,
                face_lm.landmark[291].y * h + offset_y,
            ),
            "chin": Landmark(
                face_lm.landmark[152].x * w + offset_x,
                face_lm.landmark[152].y * h + offset_y,
            ),
            "forehead": Landmark(
                face_lm.landmark[10].x * w + offset_x,
                face_lm.landmark[10].y * h + offset_y,
            ),
            "left_cheek": Landmark(
                face_lm.landmark[234].x * w + offset_x,
                face_lm.landmark[234].y * h + offset_y,
            ),
            "right_cheek": Landmark(
                face_lm.landmark[454].x * w + offset_x,
                face_lm.landmark[454].y * h + offset_y,
            ),
        }

    # ── InsightFace (fallback landmark extraction) ─────

    def _get_insightface(self):
        if self._insight_app is None:
            try:
                from insightface.app import FaceAnalysis
                self._insight_app = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CPUExecutionProvider"],
                )
                self._insight_app.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("InsightFace loaded")
            except (ImportError, Exception) as e:
                logger.warning("InsightFace unavailable: %s", e)
                self._insight_app = False
        return self._insight_app if self._insight_app is not False else None

    def _extract_landmarks_insightface(
        self, face_crop_bgr: np.ndarray, offset_x: int, offset_y: int
    ) -> dict[str, Landmark]:
        """
        Run InsightFace on a cropped face region.
        Returns 5 landmarks in original image coordinates.
        """
        app = self._get_insightface()
        if app is None:
            return {}

        faces = app.get(face_crop_bgr)
        if not faces:
            return {}

        face = faces[0]
        if face.kps is None or len(face.kps) < 5:
            return {}

        return {
            "left_eye": Landmark(float(face.kps[0][0]) + offset_x, float(face.kps[0][1]) + offset_y),
            "right_eye": Landmark(float(face.kps[1][0]) + offset_x, float(face.kps[1][1]) + offset_y),
            "nose_tip": Landmark(float(face.kps[2][0]) + offset_x, float(face.kps[2][1]) + offset_y),
            "mouth_left": Landmark(float(face.kps[3][0]) + offset_x, float(face.kps[3][1]) + offset_y),
            "mouth_right": Landmark(float(face.kps[4][0]) + offset_x, float(face.kps[4][1]) + offset_y),
        }

    # ── Estimated landmarks (last resort) ──────────────

    def _estimate_landmarks(self, bbox: tuple[int, int, int, int]) -> dict[str, Landmark]:
        """Estimate landmark positions from bounding box using average face proportions."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2

        return {
            "left_eye": Landmark(x1 + w * 0.3, y1 + h * 0.35),
            "right_eye": Landmark(x1 + w * 0.7, y1 + h * 0.35),
            "nose_tip": Landmark(cx, y1 + h * 0.55),
            "mouth_left": Landmark(x1 + w * 0.35, y1 + h * 0.75),
            "mouth_right": Landmark(x1 + w * 0.65, y1 + h * 0.75),
            "chin": Landmark(cx, y1 + h * 0.95),
            "forehead": Landmark(cx, y1 + h * 0.1),
            "left_cheek": Landmark(x1 + w * 0.2, y1 + h * 0.55),
            "right_cheek": Landmark(x1 + w * 0.8, y1 + h * 0.55),
        }

    # ── Fallback center crop ───────────────────────────

    def _detect_fallback(self, image: np.ndarray) -> list[FaceDetection]:
        """Last resort — assume a face is centered in the image."""
        h, w = image.shape[:2]
        size = min(h, w)
        x1 = (w - size) // 2
        y1 = (h - size) // 2
        bbox = (x1, y1, x1 + size, y1 + size)

        landmarks = self._estimate_landmarks(bbox)

        return [FaceDetection(
            bbox=bbox,
            confidence=0.1,
            landmarks=landmarks,
            detector="fallback",
        )]

    # ── Public API ─────────────────────────────────────

    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        """
        Detect faces and extract landmarks.

        Pipeline:
          1. MTCNN detects faces (bounding boxes)
          2. For each face, extract landmarks:
             a. MediaPipe FaceMesh (9 landmarks)
             b. InsightFace fallback (5 landmarks)
             c. Estimated from bbox (9 landmarks, low quality)
          3. If MTCNN finds nothing → fallback center crop

        Input: BGR image (OpenCV format).
        Returns: list of FaceDetection.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_bgr.shape[:2]

        # Step 1: Detect faces with MTCNN
        faces = self._detect_faces_mtcnn(image_rgb)

        if not faces:
            logger.debug("MTCNN found nothing — fallback center crop")
            return self._detect_fallback(image_bgr)

        logger.debug("MTCNN found %d face(s)", len(faces))

        # Step 2: Extract landmarks for each detected face
        detections = []
        for face in faces:
            bbox = face["bbox"]
            x1, y1, x2, y2 = bbox

            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            # Crop the face region with padding for landmark detection
            pad = int((x2 - x1) * 0.15)
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(img_w, x2 + pad)
            cy2 = min(img_h, y2 + pad)

            face_crop_bgr = image_bgr[cy1:cy2, cx1:cx2]
            face_crop_rgb = image_rgb[cy1:cy2, cx1:cx2]

            # Try MediaPipe first
            landmarks = self._extract_landmarks_mediapipe(face_crop_rgb, cx1, cy1)
            detector_name = "mtcnn+mediapipe"

            # Try InsightFace if MediaPipe failed
            if not landmarks:
                landmarks = self._extract_landmarks_insightface(face_crop_bgr, cx1, cy1)
                detector_name = "mtcnn+insightface"

            # Estimate from bbox if both failed
            if not landmarks:
                landmarks = self._estimate_landmarks((x1, y1, x2, y2))
                detector_name = "mtcnn+estimated"

            detections.append(FaceDetection(
                bbox=(x1, y1, x2, y2),
                confidence=face["confidence"],
                landmarks=landmarks,
                detector=detector_name,
            ))

        return detections