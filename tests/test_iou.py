"""Unit tests for BoundingBox schemas and vectorized IoU calculations."""

import numpy as np
import pytest
from app.models.schemas import BoundingBox
from app.services.bytetrack import bbox_ious


def test_bounding_box_properties() -> None:
    """Test computed bounding box geometry properties."""
    bbox = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0)
    assert bbox.width == 40.0
    assert bbox.height == 60.0
    assert bbox.area == 2400.0
    assert pytest.approx(bbox.aspect_ratio, 0.001) == 40.0 / 60.0

    # Conversions
    xywh = bbox.to_xywh()
    assert xywh == (10.0, 20.0, 40.0, 60.0)

    xyah = bbox.to_xyah()
    assert pytest.approx(xyah[0], 0.01) == 30.0  # center_x
    assert pytest.approx(xyah[1], 0.01) == 50.0  # center_y
    assert pytest.approx(xyah[2], 0.01) == 40.0 / 60.0  # aspect_ratio
    assert pytest.approx(xyah[3], 0.01) == 60.0  # height


def test_bounding_box_coordinate_reordering() -> None:
    """Test that inverted coordinates (e.g. x2 < x1) are automatically corrected."""
    bbox = BoundingBox(x1=100.0, y1=150.0, x2=50.0, y2=80.0)
    assert bbox.x1 == 50.0
    assert bbox.x2 == 100.0
    assert bbox.y1 == 80.0
    assert bbox.y2 == 150.0


def test_iou_identical_boxes() -> None:
    """Test that identical boxes have an IoU of exactly 1.0."""
    boxes_a = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
    boxes_b = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)

    iou = bbox_ious(boxes_a, boxes_b)
    assert iou.shape == (1, 1)
    assert pytest.approx(float(iou[0, 0]), 1e-4) == 1.0


def test_iou_disjoint_boxes() -> None:
    """Test that non-overlapping boxes have an IoU of 0.0."""
    boxes_a = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    boxes_b = np.array([[100.0, 100.0, 120.0, 120.0]], dtype=np.float32)

    iou = bbox_ious(boxes_a, boxes_b)
    assert float(iou[0, 0]) == 0.0


def test_iou_partial_overlap() -> None:
    """Test known partial overlap IoU calculation.

    Box A: [0, 0, 10, 10] -> Area = 100
    Box B: [5, 0, 15, 10] -> Area = 100
    Intersection: [5, 0, 10, 10] -> Area = 50
    Union: 100 + 100 - 50 = 150
    Expected IoU: 50 / 150 = 1/3
    """
    boxes_a = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    boxes_b = np.array([[5.0, 0.0, 15.0, 10.0]], dtype=np.float32)

    iou = bbox_ious(boxes_a, boxes_b)
    assert pytest.approx(float(iou[0, 0]), 1e-4) == 1.0 / 3.0


def test_vectorized_iou_matrix_dimensions() -> None:
    """Test that NxM IoU matrix shape and values match pairwise results."""
    boxes_a = np.array([
        [0.0, 0.0, 20.0, 20.0],
        [10.0, 10.0, 30.0, 30.0],
    ], dtype=np.float32)

    boxes_b = np.array([
        [0.0, 0.0, 20.0, 20.0],
        [50.0, 50.0, 70.0, 70.0],
        [10.0, 10.0, 30.0, 30.0],
    ], dtype=np.float32)

    iou_matrix = bbox_ious(boxes_a, boxes_b)
    assert iou_matrix.shape == (2, 3)

    # Box A0 and Box B0 are identical -> 1.0
    assert pytest.approx(float(iou_matrix[0, 0]), 1e-4) == 1.0
    # Box A0 and Box B1 are disjoint -> 0.0
    assert float(iou_matrix[0, 1]) == 0.0
    # Box A1 and Box B2 are identical -> 1.0
    assert pytest.approx(float(iou_matrix[1, 2]), 1e-4) == 1.0


def test_iou_empty_inputs() -> None:
    """Test that empty box arrays yield empty matrices without exceptions."""
    empty_boxes = np.empty((0, 4), dtype=np.float32)
    boxes = np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32)

    assert bbox_ious(empty_boxes, boxes).shape == (0, 1)
    assert bbox_ious(boxes, empty_boxes).shape == (1, 0)
    assert bbox_ious(empty_boxes, empty_boxes).shape == (0, 0)

