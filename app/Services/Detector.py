"""
FaceDetector — detection + cropping identical to the training pipeline.

Training reference (Face_Service in UNetProject):
    1. MTCNN finds faces in the original image.
    2. Crop the face bbox via PIL .crop, resize to 128x128 with PIL.
    3. Run MediaPipe FaceMesh on the 128x128 face (numpy RGB).
    4. For each region: collect ~20 landmark points, take bbox of points
       with 10px pad, PIL.crop from the 128x128 face.
    5. PIL.resize each region to 64x64. Save as JPEG quality 95.

Production used to do all of this with cv2, which:
  - Read JPEGs as BGR,
  - Used cv2.resize (different interpolation default than PIL),
  - cv2.imencode'd the BGR array as JPEG — which decoders then read as RGB,
    flipping red and blue.

That is the most likely reason the prediction was wrong even though the
checkpoint loads cleanly and labels match. This file rewrites the detector
to use PIL everywhere the training pipeline did.

cv2 is still used for one thing: MTCNN/InsightFace need numpy arrays in
BGR/RGB and that's the conventional way to pass them. We convert to/from
PIL right at the edges.
"""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("media-worker.detector")

FACE_SIZE = 224       # model expects 224 for the face stream
TRAIN_FACE = 128      # training extracted regions from a 128x128 face crop
REGION_SIZE = 64
REGION_PAD = 10       # px padding around landmark polygon (matches training)


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
    bbox: tuple[int, int, int, int]           # x1, y1, x2, y2 (orig image)
    confidence: float
    landmark_tier: str = "fallback"           # mediapipe | insightface | fallback
    detector: str = "unknown"


@dataclass
class FaceWithCrops:
    detection: FaceDetection
    face_crop: str   # b64 JPEG, 224x224 RGB
    eyes: str        # b64 JPEG, 64x64 RGB
    mouth: str
    cheeks: str
    forehead: str


# ── Helpers ─────────────────────────────────────────────

def _pil_to_b64_jpeg(img: Image.Image, quality: int = 95) -> str:
    """Encode a PIL image to base64 JPEG (RGB)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _crop_with_pad(
    img: Image.Image,
    points: list[tuple[int, int]],
    padding: int,
) -> Optional[Image.Image]:
    """Tight bbox over points, padded, clamped to image bounds — PIL crop."""
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    w, h = img.size
    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(w, max(xs) + padding)
    y2 = min(h, max(ys) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return img.crop((x1, y1, x2, y2))


class FaceDetector:
    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self._mtcnn = None
        self._mp = None
        self._insight = None

    # ── MTCNN — face detection ─────────────────────────

    def _get_mtcnn(self):
        if self._mtcnn is None:
            try:
                from mtcnn import MTCNN
                # MTCNN 1.0+ dropped min_face_size from __init__.
                self._mtcnn = MTCNN()
                logger.info("MTCNN loaded")
            except ImportError:
                logger.warning("MTCNN not installed")
                self._mtcnn = False
        return self._mtcnn if self._mtcnn is not False else None

    def _detect_faces_mtcnn(self, rgb_np: np.ndarray) -> list[dict]:
        mtcnn = self._get_mtcnn()
        if mtcnn is None:
            return []
        results = mtcnn.detect_faces(rgb_np)
        faces = []
        for r in results:
            if r["confidence"] < self.min_confidence:
                continue
            x, y, w, h = r["box"]
            if w < 40 or h < 40:
                continue
            faces.append({
                "bbox": (max(0, x), max(0, y), x + w, y + h),
                "confidence": r["confidence"],
            })
        return faces

    # ── MediaPipe FaceMesh — polygon landmark extraction ─

    def _get_mediapipe(self):
        if self._mp is None:
            try:
                import mediapipe as mp
                # static_image_mode + max_num_faces=1 matches training defaults.
                self._mp = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    min_detection_confidence=0.5,
                )
                logger.info("MediaPipe FaceMesh loaded")
            except ImportError:
                logger.warning("MediaPipe not installed")
                self._mp = False
        return self._mp if self._mp is not False else None

    def _region_pil_crops_from_mediapipe(
        self, face_pil_128: Image.Image,
    ) -> Optional[dict[str, Image.Image]]:
        """
        Run MediaPipe on the 128x128 face crop and return PIL crops for each
        region — bbox of landmark polygon + 10px pad — identical to training.
        Returns None if MediaPipe finds nothing or any region fails.
        """
        mp_detector = self._get_mediapipe()
        if mp_detector is None:
            return None

        rgb_np = np.array(face_pil_128)
        results = mp_detector.process(rgb_np)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        w, h = face_pil_128.size

        def points_for(indices):
            return [
                (int(landmarks[i].x * w), int(landmarks[i].y * h))
                for i in indices
            ]

        crops: dict[str, Image.Image] = {}
        for region, idxs in REGION_LANDMARKS.items():
            patch = _crop_with_pad(face_pil_128, points_for(idxs), REGION_PAD)
            if patch is None:
                return None
            crops[region] = patch
        return crops

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

    def _region_pil_crops_from_insightface(
        self, face_pil_128: Image.Image,
    ) -> Optional[dict[str, Image.Image]]:
        """Fallback when MediaPipe fails — coarse boxes around 5 keypoints."""
        app = self._get_insightface()
        if app is None:
            return None
        # InsightFace expects BGR numpy
        rgb_np = np.array(face_pil_128)
        bgr_np = rgb_np[:, :, ::-1].copy()
        faces = app.get(bgr_np)
        if not faces or faces[0].kps is None or len(faces[0].kps) < 5:
            return None

        kps = faces[0].kps  # left_eye, right_eye, nose, mouth_left, mouth_right
        w, h = face_pil_128.size

        def box(cx: float, cy: float, side: int) -> tuple[int, int, int, int]:
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

        boxes = {
            "eyes":     box(eye_cx, eye_cy, eye_side),
            "mouth":    box(mouth_cx, mouth_cy, mouth_side),
            "cheeks":   box(kps[2][0], kps[2][1] + h * 0.1, int(w * 0.55)),
            "forehead": box(eye_cx, eye_cy - h * 0.2, int(w * 0.6)),
        }
        return {name: face_pil_128.crop(b) for name, b in boxes.items()}

    # ── Estimated fallback ─────────────────────────────

    @staticmethod
    def _region_pil_crops_estimated(face_pil_128: Image.Image) -> dict[str, Image.Image]:
        w, h = face_pil_128.size

        def b(cx_f: float, cy_f: float, w_f: float, h_f: float) -> tuple[int, int, int, int]:
            cx = w * cx_f
            cy = h * cy_f
            hw = (w * w_f) / 2
            hh = (h * h_f) / 2
            return (
                max(0, int(cx - hw)), max(0, int(cy - hh)),
                min(w, int(cx + hw)), min(h, int(cy + hh)),
            )

        return {
            "eyes":     face_pil_128.crop(b(0.50, 0.36, 0.70, 0.20)),
            "mouth":    face_pil_128.crop(b(0.50, 0.77, 0.45, 0.20)),
            "cheeks":   face_pil_128.crop(b(0.50, 0.58, 0.78, 0.22)),
            "forehead": face_pil_128.crop(b(0.50, 0.16, 0.70, 0.20)),
        }

    # ── Public API ─────────────────────────────────────

    def detect_and_crop(self, image_bytes: bytes) -> list[FaceWithCrops]:
        # Decode through PIL — handles JPEG/PNG/WebP, gives RGB directly.
        try:
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.error("Failed to decode image bytes: %s", e)
            return []

        rgb_np = np.array(image_pil)
        img_h, img_w = rgb_np.shape[:2]

        faces = self._detect_faces_mtcnn(rgb_np)
        if not faces:
            logger.debug("MTCNN found nothing — falling back to center crop")
            side = min(img_h, img_w)
            cx, cy = img_w // 2, img_h // 2
            half = side // 2
            faces = [{
                "bbox": (cx - half, cy - half, cx + half, cy + half),
                "confidence": 0.1,
            }]

        results: list[FaceWithCrops] = []
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Crop face from original PIL, resize to 128 (training scale)
            face_pil_128 = image_pil.crop((x1, y1, x2, y2)).resize((TRAIN_FACE, TRAIN_FACE))

            # Region crops — 3 tiers
            tier = "mediapipe"
            region_crops = self._region_pil_crops_from_mediapipe(face_pil_128)
            if region_crops is None:
                region_crops = self._region_pil_crops_from_insightface(face_pil_128)
                tier = "insightface"
            if region_crops is None:
                region_crops = self._region_pil_crops_estimated(face_pil_128)
                tier = "fallback"

            # Resize regions to 64x64 (PIL — same as training)
            region_crops = {
                name: img.resize((REGION_SIZE, REGION_SIZE))
                for name, img in region_crops.items()
            }

            # Face for the model: send at 128x128 (training storage size).
            # Predictor's FACE_TRANSFORM.Resize(224) does the upsample,
            # matching training's DataLoader exactly.
            # JPEG-encode + base64 each (RGB, quality 95 — matches training save)
            b64 = {
                "face":     _pil_to_b64_jpeg(face_pil_128),
                "eyes":     _pil_to_b64_jpeg(region_crops["eyes"]),
                "mouth":    _pil_to_b64_jpeg(region_crops["mouth"]),
                "cheeks":   _pil_to_b64_jpeg(region_crops["cheeks"]),
                "forehead": _pil_to_b64_jpeg(region_crops["forehead"]),
            }

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