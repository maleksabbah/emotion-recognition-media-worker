"""
Media Worker Test Suite
Run: pytest Test.py -v -o asyncio_mode=auto -o python_files=Test.py -o python_classes=Test
"""
from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest

# ══════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════

@pytest.fixture
def sample_image_bgr():
    """A 480x640 BGR test image with a centered bright region (fake face)."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a face-like ellipse in the center
    cv2.ellipse(img, (320, 240), (80, 100), 0, 0, 360, (180, 160, 140), -1)
    # Draw eyes
    cv2.circle(img, (290, 210), 10, (255, 255, 255), -1)
    cv2.circle(img, (350, 210), 10, (255, 255, 255), -1)
    # Draw mouth
    cv2.ellipse(img, (320, 280), (25, 10), 0, 0, 360, (0, 0, 200), -1)
    return img


@pytest.fixture
def sample_image_b64(sample_image_bgr):
    """Base64-encoded JPEG of the sample image."""
    _, buf = cv2.imencode(".jpg", sample_image_bgr)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@pytest.fixture
def sample_landmarks():
    """Landmarks dict matching what the detector would return."""
    from app.Detector import Landmark
    return {
        "left_eye": Landmark(290, 210),
        "right_eye": Landmark(350, 210),
        "nose_tip": Landmark(320, 240),
        "mouth_left": Landmark(295, 280),
        "mouth_right": Landmark(345, 280),
        "chin": Landmark(320, 320),
        "forehead": Landmark(320, 160),
        "left_cheek": Landmark(260, 240),
        "right_cheek": Landmark(380, 240),
    }


@pytest.fixture
def sample_detection(sample_landmarks):
    from app.Detector import FaceDetection
    return FaceDetection(
        bbox=(240, 140, 400, 340),
        confidence=0.95,
        landmarks=sample_landmarks,
        detector="test",
    )


@pytest.fixture
def mock_consumer():
    consumer = AsyncMock()
    consumer.start = AsyncMock()
    consumer.stop = AsyncMock()
    consumer.commit = AsyncMock()
    return consumer


@pytest.fixture
def mock_producer():
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock()
    return producer


# ══════════════════════════════════════════════
# Detector tests
# ══════════════════════════════════════════════

class TestFaceDetector:
    def test_fallback_returns_detection(self, sample_image_bgr):
        from app.Detector import FaceDetector
        detector = FaceDetector()
        # Force fallback by making MTCNN unavailable
        detector._mtcnn = False

        detections = detector.detect(sample_image_bgr)
        assert len(detections) == 1
        assert detections[0].detector == "fallback"
        assert detections[0].confidence == 0.1

    def test_fallback_has_landmarks(self, sample_image_bgr):
        from app.Detector import FaceDetector
        detector = FaceDetector()
        detector._mtcnn = False

        detection = detector.detect(sample_image_bgr)[0]
        assert "left_eye" in detection.landmarks
        assert "right_eye" in detection.landmarks
        assert "mouth_left" in detection.landmarks
        assert "forehead" in detection.landmarks
        assert "left_cheek" in detection.landmarks
        assert "right_cheek" in detection.landmarks

    def test_fallback_bbox_is_square(self, sample_image_bgr):
        from app.Detector import FaceDetector
        detector = FaceDetector()
        detector._mtcnn = False

        detection = detector.detect(sample_image_bgr)[0]
        x1, y1, x2, y2 = detection.bbox
        assert (x2 - x1) == (y2 - y1)  # square crop

    def test_estimated_landmarks_all_present(self):
        from app.Detector import FaceDetector
        detector = FaceDetector()
        landmarks = detector._estimate_landmarks((100, 100, 300, 400))
        assert len(landmarks) == 9
        for name in ("left_eye", "right_eye", "nose_tip", "mouth_left",
                      "mouth_right", "chin", "forehead", "left_cheek", "right_cheek"):
            assert name in landmarks

    def test_mtcnn_lazy_load_sets_false_on_import_error(self):
        from app.Detector import FaceDetector
        detector = FaceDetector()
        with patch.dict("sys.modules", {"mtcnn": None}):
            detector._mtcnn = None
            result = detector._get_mtcnn()
            assert result is None

    def test_mediapipe_lazy_load_sets_false_on_import_error(self):
        from app.Detector import FaceDetector
        detector = FaceDetector()
        with patch.dict("sys.modules", {"mediapipe": None}):
            detector._mp_detector = None
            result = detector._get_mediapipe()
            assert result is None

    def test_insightface_lazy_load_sets_false_on_import_error(self):
        from app.Detector import FaceDetector
        detector = FaceDetector()
        with patch.dict("sys.modules", {"insightface": None, "insightface.app": None}):
            detector._insight_app = None
            result = detector._get_insightface()
            assert result is None


# ══════════════════════════════════════════════
# Cropper tests
# ══════════════════════════════════════════════

class TestRegionCropper:
    def test_extract_regions_returns_all_crops(self, sample_image_bgr, sample_detection):
        from app.Cropper import extract_regions
        crops = extract_regions(sample_image_bgr, sample_detection)
        assert crops is not None
        assert crops.face.shape == (224, 224, 3)
        assert crops.eyes.shape == (64, 64, 3)
        assert crops.mouth.shape == (64, 64, 3)
        assert crops.cheeks.shape == (64, 64, 3)
        assert crops.forehead.shape == (64, 64, 3)

    def test_extract_regions_without_landmarks(self, sample_image_bgr):
        """Should use estimated positions when landmarks are missing."""
        from app.Detector import FaceDetection
        from app.Cropper import extract_regions

        detection = FaceDetection(
            bbox=(200, 100, 440, 380),
            confidence=0.8,
            landmarks={},
            detector="test",
        )
        crops = extract_regions(sample_image_bgr, detection)
        assert crops is not None
        assert crops.face.shape == (224, 224, 3)
        assert crops.eyes.shape == (64, 64, 3)

    def test_crops_to_base64(self, sample_image_bgr, sample_detection):
        from app.Cropper import extract_regions, crops_to_base64
        crops = extract_regions(sample_image_bgr, sample_detection)
        b64 = crops_to_base64(crops)

        assert set(b64.keys()) == {"face", "eyes", "mouth", "cheeks", "forehead"}
        for key, val in b64.items():
            decoded = base64.b64decode(val)
            assert len(decoded) > 100  # valid JPEG data

    def test_safe_crop_handles_edge_bbox(self):
        """Crop near image edge should pad instead of crash."""
        from app.Cropper import _safe_crop
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = _safe_crop(img, 5, 5, 50)  # top-left corner
        assert crop.shape == (50, 50, 3)

    def test_safe_crop_handles_overflow_bbox(self):
        """Crop extending past image should pad instead of crash."""
        from app.Cropper import _safe_crop
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = _safe_crop(img, 95, 95, 50)  # bottom-right corner
        assert crop.shape == (50, 50, 3)


# ══════════════════════════════════════════════
# Worker tests
# ══════════════════════════════════════════════

class TestMediaWorker:
    def test_decode_image(self, sample_image_b64):
        from app.Worker import MediaWorker
        worker = MediaWorker()
        img = worker._decode_image(sample_image_b64)
        assert img.shape[2] == 3  # BGR
        assert img.shape[0] > 0
        assert img.shape[1] > 0

    def test_decode_image_invalid(self):
        from app.Worker import MediaWorker
        worker = MediaWorker()
        with pytest.raises(ValueError, match="Failed to decode"):
            worker._decode_image(base64.b64encode(b"not an image").decode())

    def test_process_frame_returns_detections(self, sample_image_b64):
        from app.Worker import MediaWorker
        worker = MediaWorker()
        # Force fallback detector (no MTCNN)
        worker.detector._mtcnn = False

        result = worker._process_frame(sample_image_b64)
        assert result["status"] in ("success", "no_face")
        if result["status"] == "success":
            assert len(result["detections"]) >= 1
            det = result["detections"][0]
            assert "bbox" in det
            assert "crops" in det
            assert set(det["crops"].keys()) == {"face", "eyes", "mouth", "cheeks", "forehead"}

    def test_process_frame_crops_are_valid_b64(self, sample_image_b64):
        from app.Worker import MediaWorker
        worker = MediaWorker()
        worker.detector._mtcnn = False

        result = worker._process_frame(sample_image_b64)
        if result["status"] == "success":
            for crop_name, crop_b64 in result["detections"][0]["crops"].items():
                img_bytes = base64.b64decode(crop_b64)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                assert img is not None, f"{crop_name} crop is not valid JPEG"

    @pytest.mark.asyncio
    async def test_handle_task_success(self, sample_image_b64):
        from app.Worker import MediaWorker
        worker = MediaWorker()
        worker.detector._mtcnn = False

        task = {
            "session_id": "test-session",
            "frame_index": 0,
            "image_b64": sample_image_b64,
            "mode": "upload",
        }
        result = await worker._handle_task(task)
        assert result["session_id"] == "test-session"
        assert result["frame_index"] == 0
        assert result["status"] in ("success", "no_face")
        assert result["error"] is None
        assert result["processing_ms"] > 0

    @pytest.mark.asyncio
    async def test_handle_task_error(self):
        from app.Worker import MediaWorker
        worker = MediaWorker()

        task = {
            "session_id": "test-session",
            "frame_index": 0,
            "image_b64": "not-valid-base64!!!",
        }
        result = await worker._handle_task(task)
        assert result["status"] == "error"
        assert result["error"] is not None


# ══════════════════════════════════════════════
# Kafka integration tests
# ══════════════════════════════════════════════

class TestKafkaIntegration:
    @pytest.mark.asyncio
    async def test_publish_result(self, mock_producer):
        from app.Kafka import publish_result
        result = {
            "session_id": "s1",
            "frame_index": 0,
            "worker_id": "test-worker",
            "status": "success",
        }
        await publish_result(mock_producer, result)
        mock_producer.send_and_wait.assert_called_once()
        call_kwargs = mock_producer.send_and_wait.call_args
        assert call_kwargs[1]["value"] == result

    @pytest.mark.asyncio
    async def test_publish_result_includes_worker_header(self, mock_producer):
        from app.Kafka import publish_result
        result = {"worker_id": "w1", "session_id": "s1", "frame_index": 0}
        await publish_result(mock_producer, result)
        headers = mock_producer.send_and_wait.call_args[1]["headers"]
        assert headers == [("worker_id", b"w1")]


# ══════════════════════════════════════════════
# End-to-end pipeline test
# ══════════════════════════════════════════════

class TestEndToEnd:
    def test_full_pipeline_detect_crop_encode(self, sample_image_bgr):
        """Full path: image → detect → crop → base64 → decode → verify shapes."""
        from app.Detector import FaceDetector
        from app.Cropper import extract_regions, crops_to_base64

        detector = FaceDetector()
        detector._mtcnn = False

        # Detect
        detections = detector.detect(sample_image_bgr)
        assert len(detections) >= 1

        # Crop
        crops = extract_regions(sample_image_bgr, detections[0])
        assert crops is not None

        # Encode
        b64 = crops_to_base64(crops)

        # Decode and verify
        for name, expected_size in [("face", 224), ("eyes", 64), ("mouth", 64),
                                     ("cheeks", 64), ("forehead", 64)]:
            img_bytes = base64.b64decode(b64[name])
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            assert img is not None
            # JPEG compression may slightly change size, but should be close
            assert abs(img.shape[0] - expected_size) <= 2
            assert abs(img.shape[1] - expected_size) <= 2