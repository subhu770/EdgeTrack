"""Integration tests for EdgeTrack FastAPI REST API contracts and schema validation."""

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import HealthResponse, MetricsResponse, TrackResponse, DetectionResponse

client = TestClient(app)


def _generate_test_image_bytes(width: int = 640, height: int = 480) -> bytes:
    """Create a synthetic in-memory JPEG image for upload testing."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw simple shapes
    cv2.circle(img, (width // 2, height // 2), 50, (255, 255, 255), -1)
    success, buffer = cv2.imencode(".jpg", img)
    assert success
    return buffer.tobytes()


def test_health_endpoint() -> None:
    """Test GET /health returns 200 and valid HealthResponse schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    # Validate against Pydantic schema
    health = HealthResponse(**data)
    assert health.status == "healthy"
    assert health.model_loaded is True
    assert health.execution_provider == "CPUExecutionProvider"
    assert health.cpu_threads > 0


def test_metrics_endpoint() -> None:
    """Test GET /metrics returns 200 and valid MetricsResponse schema."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()

    metrics = MetricsResponse(**data)
    assert metrics.total_requests >= 0
    assert metrics.total_frames_processed >= 0


def test_reset_endpoint() -> None:
    """Test POST /api/v1/reset returns 200 and success status."""
    response = client.post("/api/v1/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_detect_endpoint() -> None:
    """Test POST /api/v1/detect with synthetic image upload."""
    img_bytes = _generate_test_image_bytes(640, 480)
    files = {"file": ("test_frame.jpg", io.BytesIO(img_bytes), "image/jpeg")}

    response = client.post("/api/v1/detect", files=files)
    assert response.status_code == 200
    data = response.json()

    det_resp = DetectionResponse(**data)
    assert det_resp.image_width == 640
    assert det_resp.image_height == 480
    assert det_resp.metrics.total_ms > 0
    assert det_resp.metrics.fps > 0


def test_track_endpoint() -> None:
    """Test POST /api/v1/track with consecutive image uploads to verify tracking."""
    img_bytes = _generate_test_image_bytes(640, 480)

    # Frame 1
    files_1 = {"file": ("frame_001.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    resp_1 = client.post("/api/v1/track", files=files_1)
    assert resp_1.status_code == 200
    data_1 = resp_1.json()

    track_resp_1 = TrackResponse(**data_1)
    assert track_resp_1.image_width == 640
    assert track_resp_1.image_height == 480
    assert track_resp_1.metrics.track_ms >= 0

    # Frame 2
    files_2 = {"file": ("frame_002.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    resp_2 = client.post("/api/v1/track", files=files_2)
    assert resp_2.status_code == 200
    data_2 = resp_2.json()

    track_resp_2 = TrackResponse(**data_2)
    assert track_resp_2.active_tracks_count >= 1


def test_invalid_image_upload_returns_400() -> None:
    """Test that corrupt image binary returns 400 Bad Request."""
    corrupt_bytes = b"NOT_A_VALID_IMAGE_DATA_HEADER"
    files = {"file": ("corrupt.jpg", io.BytesIO(corrupt_bytes), "image/jpeg")}

    response = client.post("/api/v1/track", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data

