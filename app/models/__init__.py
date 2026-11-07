"""Pydantic v2 domain schemas."""
from app.models.schemas import (
    BoundingBox,
    Detection,
    TrackState,
    TrackedObject,
    LatencyMetrics,
    DetectionResponse,
    TrackResponse,
    HealthResponse,
    MetricsResponse,
)

__all__ = [
    "BoundingBox",
    "Detection",
    "TrackState",
    "TrackedObject",
    "LatencyMetrics",
    "DetectionResponse",
    "TrackResponse",
    "HealthResponse",
    "MetricsResponse",
]

