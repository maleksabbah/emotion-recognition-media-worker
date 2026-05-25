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
        # mtcnn==1.0.0 (TensorFlow-based PyPI package) dropped the
        # min_face_size kwarg from __init__ — it's no longer accepted.
        # Smaller-face filtering is done after detection via bbox area.
        if self._mtcnn is None:
            try:
                from mtcnn import MTCNN
                self._mtcnn = MTCNN()
                logger.info("MTCNN loaded")
            except ImportError:
                logger.warning("MTCNN not installed — face detection unavailable")
                self._mtcnn = False
        return self._mtcnn if self._mtcnn is not False else None

    def _detect_faces_mtcnn(self, image_rgb: np.ndarray) -> list[dict]:
        """
        Detect faces using MTCNN.

        mtcnn==1.0.0 returns a list of dicts with keys 'box' [x, y, w, h]
        and 'confidence'. Older versions may key the score as 'score' so we
        accept either. Faces below MIN_FACE_PX in the larger side are
        dropped (the old min_face_size=40 enforced this).

        Returns list of {bbox: (x1,y1,x2,y2), confidence: float}.
        """
        MIN_FACE_PX = 40

        mtcnn = self._get_mtcnn()
        if mtcnn is None:
            return []

        try:
            results = mtcnn.detect_faces(image_rgb)
        except Exception as e:
            logger.error("MTCNN detect_faces failed: %s", e)
            return []

        faces: list[dict] = []
        for r in results or []:
            # 1.0.0 → 'confidence', older → 'score'; accept either.
            conf = r.get("confidence", r.get("score", 0.0))
            if conf < self.min_confidence:
                continue

            box = r.get("box") or r.get("bbox")
            if not box or len(box) < 4:
                continue
            x, y, w, h = box

            # Drop tiny faces — replaces the old min_face_size= kwarg.
            if max(w, h) < MIN_FACE_PX:
                continue

            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = int(x + w)
            y2 = int(y + h)
            faces.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(conf),
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


# ──────────────────────────────────────────────────────────────────────
# Cropping — one method, one loop
# ──────────────────────────────────────────────────────────────────────

import base64

FACE_SIZE = 224
REGION_SIZE = 64

# (landmark_a, landmark_b, vertical_fraction_fallback)
# For forehead, only one landmark; second slot is None.
_REGION_SPECS = {
    "eyes":     ("left_eye",   "right_eye",    0.35),
    "mouth":    ("mouth_left", "mouth_right",  0.75),
    "cheeks":   ("left_cheek", "right_cheek",  0.55),
    "forehead": ("forehead",   None,           0.15),
}


@dataclass
class FaceWithCrops:
    """A detected face plus its base64-encoded crops, ready to send over Kafka."""
    detection: "FaceDetection"
    face_crop: str
    eyes: str
    mouth: str
    cheeks: str
    forehead: str


def _detect_and_crop(self, image_bytes: bytes) -> list[FaceWithCrops]:
    """
    Decode → detect → crop 5 regions → base64 encode.
    Returns one FaceWithCrops per detected face.
    """
    # Decode JPEG bytes
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        logger.error("Failed to decode image bytes")
        return []

    detections = self.detect(image)
    h, w = image.shape[:2]
    results: list[FaceWithCrops] = []

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        lm = det.landmarks

        # Face crop — bbox + 10% padding, resize to 224
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        face = cv2.resize(
            image[max(0, y1-pad_y):min(h, y2+pad_y),
                  max(0, x1-pad_x):min(w, x2+pad_x)],
            (FACE_SIZE, FACE_SIZE),
        )

        # Region crops — square side scales with face width, min 64px
        side = max(int((x2 - x1) * 0.4), REGION_SIZE)
        half = side // 2

        crops: dict[str, np.ndarray] = {"face": face}
        for name, (lm_a, lm_b, frac) in _REGION_SPECS.items():
            # Pick center: landmark midpoint if available, else bbox fraction
            if lm_a in lm and (lm_b is None or lm_b in lm):
                if lm_b is None:
                    cx, cy = lm[lm_a].x, lm[lm_a].y
                else:
                    cx = (lm[lm_a].x + lm[lm_b].x) / 2
                    cy = (lm[lm_a].y + lm[lm_b].y) / 2
            else:
                cx = (x1 + x2) / 2
                cy = y1 + (y2 - y1) * frac

            # Slice — clamp to image, reflect-pad any overhang, resize
            rx1, ry1 = int(cx - half), int(cy - half)
            rx2, ry2 = rx1 + side, ry1 + side
            pad_l = max(0, -rx1); pad_t = max(0, -ry1)
            pad_r = max(0, rx2 - w); pad_b = max(0, ry2 - h)
            sl = image[max(0, ry1):min(h, ry2), max(0, rx1):min(w, rx2)]
            if pad_l or pad_t or pad_r or pad_b:
                sl = cv2.copyMakeBorder(
                    sl, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101,
                )
            crops[name] = cv2.resize(sl, (REGION_SIZE, REGION_SIZE))

        # JPEG-encode + base64
        b64: dict[str, str] = {}
        for name, img in crops.items():
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            b64[name] = base64.b64encode(buf.tobytes()).decode("utf-8")

        results.append(FaceWithCrops(
            detection=det,
            face_crop=b64["face"],
            eyes=b64["eyes"],
            mouth=b64["mouth"],
            cheeks=b64["cheeks"],
            forehead=b64["forehead"],
        ))

    return results


FaceDetector.detect_and_crop = _detect_and_crop