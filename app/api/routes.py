"""EdgeTrack REST API endpoints for detection, tracking, health, and metrics."""

import time
from typing import Optional
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.logging import logger
from app.models.schemas import (
    DetectionResponse,
    HealthResponse,
    LatencyMetrics,
    MetricsResponse,
    TrackResponse,
)
from app.services.bytetrack import ByteTracker
from app.services.onnx_detector import ONNXDetector

router = APIRouter()

# Global state / singleton service instances
_detector: Optional[ONNXDetector] = None
_tracker: Optional[ByteTracker] = None
_start_time = time.time()

# Aggregated metrics counters
_metrics = {
    "total_requests": 0,
    "total_frames": 0,
    "sum_preprocess_ms": 0.0,
    "sum_inference_ms": 0.0,
    "sum_postprocess_ms": 0.0,
    "sum_track_ms": 0.0,
    "sum_total_ms": 0.0,
}


def get_detector() -> ONNXDetector:
    """Retrieve or initialize the singleton ONNX detector."""
    global _detector
    if _detector is None:
        settings = get_settings()
        model_path = settings.INT8_MODEL_PATH if settings.USE_INT8 else settings.MODEL_PATH
        _detector = ONNXDetector(
            model_path=model_path,
            conf_threshold=settings.CONF_THRESHOLD,
            nms_iou_threshold=settings.NMS_IOU_THRESHOLD,
            input_size=(settings.INPUT_WIDTH, settings.INPUT_HEIGHT),
            num_threads=settings.ORT_NUM_THREADS,
        )
    return _detector


def get_tracker() -> ByteTracker:
    """Retrieve or initialize the singleton ByteTracker."""
    global _tracker
    if _tracker is None:
        settings = get_settings()
        _tracker = ByteTracker(
            track_high_thresh=settings.TRACK_HIGH_THRESH,
            track_low_thresh=settings.TRACK_LOW_THRESH,
            new_track_thresh=settings.NEW_TRACK_THRESH,
            match_thresh=settings.MATCH_THRESH,
            match_thresh_second=settings.MATCH_THRESH_SECOND,
            match_thresh_unconfirmed=settings.MATCH_THRESH_UNCONFIRMED,
            max_lost_frames=settings.MAX_LOST_FRAMES,
            min_consecutive_frames=settings.MIN_CONSECUTIVE_FRAMES,
        )
    return _tracker


async def _decode_image_upload(file: UploadFile) -> np.ndarray:
    """Read and decode uploaded image file into BGR NumPy array."""
    if not file.content_type or not file.content_type.startswith("image/"):
        # Allow octet-stream fallback or check file extension
        pass

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file received.",
            )
        np_arr = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("cv2.imdecode returned None")
        return image
    except Exception as exc:
        logger.error(f"Failed to decode uploaded image: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format or corrupt file data.",
        ) from exc


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def healthcheck() -> HealthResponse:
    """System health check and inference engine status."""
    settings = get_settings()
    detector = get_detector()
    is_loaded = detector.session is not None
    uptime = time.time() - _start_time

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        version=settings.VERSION,
        uptime_seconds=round(uptime, 2),
        model_loaded=is_loaded,
        model_path=detector.model_path,
        execution_provider="CPUExecutionProvider",
        cpu_threads=detector.num_threads,
    )


@router.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
def get_metrics() -> MetricsResponse:
    """Aggregated latency and throughput metrics."""
    frames = max(1, _metrics["total_frames"])
    avg_total = _metrics["sum_total_ms"] / frames
    fps = round(1000.0 / avg_total, 2) if avg_total > 0 else 0.0

    tracker = get_tracker()
    active_count = len([t for t in tracker.tracked_stracks if t.is_activated])

    return MetricsResponse(
        total_requests=_metrics["total_requests"],
        total_frames_processed=_metrics["total_frames"],
        avg_preprocess_ms=round(_metrics["sum_preprocess_ms"] / frames, 3),
        avg_inference_ms=round(_metrics["sum_inference_ms"] / frames, 3),
        avg_postprocess_ms=round(_metrics["sum_postprocess_ms"] / frames, 3),
        avg_tracking_ms=round(_metrics["sum_track_ms"] / frames, 3),
        avg_total_ms=round(avg_total, 3),
        effective_fps=fps,
        current_active_tracks=active_count,
    )


@router.post("/api/v1/reset", tags=["Tracking"])
def reset_tracker() -> dict:
    """Reset tracker state and restart track ID sequences."""
    tracker = get_tracker()
    tracker.reset()
    logger.info("ByteTracker state reset successfully.")
    return {"status": "success", "message": "Tracker state reset successfully."}


@router.post("/api/v1/detect", response_model=DetectionResponse, tags=["Inference"])
async def detect_frame(file: UploadFile = File(...)) -> DetectionResponse:
    """Run ONNX object detection on a single image frame."""
    t_start = time.perf_counter()
    image = await _decode_image_upload(file)
    height, width = image.shape[:2]

    detector = get_detector()
    detections, latencies = detector.detect(image)

    t_end = time.perf_counter()
    total_ms = round((t_end - t_start) * 1000.0, 3)
    fps = round(1000.0 / max(total_ms, 0.001), 2)

    metrics = LatencyMetrics(
        preprocess_ms=latencies["preprocess_ms"],
        inference_ms=latencies["inference_ms"],
        postprocess_ms=latencies["postprocess_ms"],
        track_ms=0.0,
        total_ms=total_ms,
        fps=fps,
    )

    _metrics["total_requests"] += 1
    _metrics["total_frames"] += 1
    _metrics["sum_preprocess_ms"] += latencies["preprocess_ms"]
    _metrics["sum_inference_ms"] += latencies["inference_ms"]
    _metrics["sum_postprocess_ms"] += latencies["postprocess_ms"]
    _metrics["sum_total_ms"] += total_ms

    return DetectionResponse(
        frame_id=file.filename,
        image_width=width,
        image_height=height,
        detections_count=len(detections),
        detections=detections,
        metrics=metrics,
    )


@router.post("/api/v1/track", response_model=TrackResponse, tags=["Tracking"])
async def track_frame(file: UploadFile = File(...)) -> TrackResponse:
    """Execute end-to-end detection and ByteTrack multi-object tracking on a frame."""
    t_start = time.perf_counter()
    image = await _decode_image_upload(file)
    height, width = image.shape[:2]

    detector = get_detector()
    tracker = get_tracker()

    # 1. Vision Detection
    detections, det_latencies = detector.detect(image)

    # 2. ByteTrack State Estimation & Association
    t_track_start = time.perf_counter()
    active_tracks = tracker.update(detections)
    t_track_end = time.perf_counter()
    track_ms = round((t_track_end - t_track_start) * 1000.0, 3)

    t_end = time.perf_counter()
    total_ms = round((t_end - t_start) * 1000.0, 3)
    fps = round(1000.0 / max(total_ms, 0.001), 2)

    metrics = LatencyMetrics(
        preprocess_ms=det_latencies["preprocess_ms"],
        inference_ms=det_latencies["inference_ms"],
        postprocess_ms=det_latencies["postprocess_ms"],
        track_ms=track_ms,
        total_ms=total_ms,
        fps=fps,
    )

    # Update runtime metrics
    _metrics["total_requests"] += 1
    _metrics["total_frames"] += 1
    _metrics["sum_preprocess_ms"] += det_latencies["preprocess_ms"]
    _metrics["sum_inference_ms"] += det_latencies["inference_ms"]
    _metrics["sum_postprocess_ms"] += det_latencies["postprocess_ms"]
    _metrics["sum_track_ms"] += track_ms
    _metrics["sum_total_ms"] += total_ms

    return TrackResponse(
        frame_id=file.filename,
        image_width=width,
        image_height=height,
        active_tracks_count=len(active_tracks),
        tracks=active_tracks,
        metrics=metrics,
    )

