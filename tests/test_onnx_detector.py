"""Unit tests for ONNX Runtime detector, letterboxing, NMS, and INT8 quantization."""

import os
import numpy as np
import pytest
from app.services.onnx_detector import ONNXDetector


def test_letterbox_aspect_ratio_preservation() -> None:
    """Test letterbox resizing preserves aspect ratio and calculates correct padding."""
    # Create test image 400x800 (height=400, width=800 -> aspect ratio 2:1)
    orig_img = np.zeros((400, 800, 3), dtype=np.uint8)

    padded, ratio, (dw, dh) = ONNXDetector.letterbox(orig_img, new_shape=(640, 640))

    assert padded.shape == (640, 640, 3)
    # Scale ratio = min(640/400, 640/800) = min(1.6, 0.8) = 0.8
    assert pytest.approx(ratio, 0.001) == 0.8

    # Unpadded width = 800 * 0.8 = 640 -> dw = 0
    # Unpadded height = 400 * 0.8 = 320 -> dh = (640 - 320) / 2 = 160
    assert pytest.approx(dw, 1.0) == 0.0
    assert pytest.approx(dh, 1.0) == 160.0


def test_preprocess_tensor_contract() -> None:
    """Test preprocess outputs float32 NCHW tensor normalized to [0, 1]."""
    detector = ONNXDetector(model_path="models/test_model.onnx")
    img = np.full((480, 640, 3), 255, dtype=np.uint8)

    blob, ratio, pad = detector.preprocess(img)

    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32
    assert float(np.min(blob)) >= 0.0
    assert float(np.max(blob)) <= 1.0
    assert blob.flags["C_CONTIGUOUS"]


def test_postprocess_yolov8_parsing() -> None:
    """Test YOLOv8 tensor shape (1, 84, 8400) parsing, coordinate unletterboxing, and NMS."""
    detector = ONNXDetector(
        model_path="models/test_model.onnx",
        conf_threshold=0.5,
        nms_iou_threshold=0.45,
    )

    # Synthetic raw output tensor: (1, 84, 8400)
    raw_tensor = np.zeros((1, 84, 8400), dtype=np.float32)

    # Box 1: [cx=320, cy=320, w=100, h=100], class 0 (person) = 0.90
    raw_tensor[0, 0, 0] = 320.0
    raw_tensor[0, 1, 0] = 320.0
    raw_tensor[0, 2, 0] = 100.0
    raw_tensor[0, 3, 0] = 100.0
    raw_tensor[0, 4, 0] = 0.90

    # Box 2: Duplicate overlapping box with lower score (should be suppressed by NMS)
    raw_tensor[0, 0, 1] = 322.0
    raw_tensor[0, 1, 1] = 321.0
    raw_tensor[0, 2, 1] = 98.0
    raw_tensor[0, 3, 1] = 102.0
    raw_tensor[0, 4, 1] = 0.75

    # Original image 480x640 letterboxed into 640x640:
    # ratio = 1.0, pad = (0, 80)
    orig_shape = (480, 640)
    ratio = 1.0
    pad = (0.0, 80.0)

    detections = detector.postprocess(raw_tensor, orig_shape, ratio, pad)

    # Due to NMS, duplicate box must be suppressed -> exactly 1 detection kept
    assert len(detections) == 1
    det = detections[0]
    assert det.class_id == 0
    assert det.confidence == 0.90
    assert det.class_name == "person"

    # Coordinates unletterboxed:
    # y1_letterbox = 320 - 50 = 270 -> y1_orig = 270 - 80 = 190
    # y2_letterbox = 320 + 50 = 370 -> y2_orig = 370 - 80 = 290
    assert pytest.approx(det.bbox.y1, abs=1.0) == 190.0
    assert pytest.approx(det.bbox.y2, abs=1.0) == 290.0


def test_dynamic_int8_quantization(tmp_path) -> None:
    """Test dynamic INT8 model quantization utility."""
    src_model = "models/test_model.onnx"
    out_model = str(tmp_path / "model_int8.onnx")

    success = ONNXDetector.quantize_model_to_int8(src_model, out_model)
    assert success is True
    assert os.path.exists(out_model)
    assert os.path.getsize(out_model) > 0

