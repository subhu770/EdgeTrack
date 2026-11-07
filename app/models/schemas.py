"""Pydantic v2 schemas for bounding boxes, detections, tracks, analytics, and API responses."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, computed_field, model_validator


class TrackState(str, Enum):
    """ByteTrack tracklet lifecycle states."""
    NEW = "New"
    TRACKED = "Tracked"
    LOST = "Lost"
    REMOVED = "Removed"


class BoundingBox(BaseModel):
    """Bounding box in (x1, y1, x2, y2) pixel coordinates."""
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        """Ensure coordinate consistency."""
        if self.x2 < self.x1:
            self.x1, self.x2 = self.x2, self.x1
        if self.y2 < self.y1:
            self.y1, self.y2 = self.y2, self.y1
        return self

    @computed_field
    @property
    def width(self) -> float:
        """Width of bounding box."""
        return round(self.x2 - self.x1, 2)

    @computed_field
    @property
    def height(self) -> float:
        """Height of bounding box."""
        return round(self.y2 - self.y1, 2)

    @computed_field
    @property
    def area(self) -> float:
        """Area in square pixels."""
        return round((self.x2 - self.x1) * (self.y2 - self.y1), 2)

    @computed_field
    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (width / height)."""
        h = max(self.y2 - self.y1, 1e-6)
        return round((self.x2 - self.x1) / h, 4)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Return (top_left_x, top_left_y, width, height)."""
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def to_xyah(self) -> Tuple[float, float, float, float]:
        """Return (center_x, center_y, aspect_ratio, height)."""
        w = self.x2 - self.x1
        h = max(self.y2 - self.y1, 1e-6)
        cx = self.x1 + w / 2.0
        cy = self.y1 + h / 2.0
        return (cx, cy, w / h, h)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BoundingBox":
        """Instantiate from top-left (x, y) and dimensions (w, h)."""
        return cls(x1=x, y1=y, x2=x + w, y2=y + h)

    @classmethod
    def from_xyah(cls, cx: float, cy: float, a: float, h: float) -> "BoundingBox":
        """Instantiate from center (cx, cy), aspect ratio a, and height h."""
        w = a * h
        return cls(
            x1=cx - w / 2.0,
            y1=cy - h / 2.0,
            x2=cx + w / 2.0,
            y2=cy + h / 2.0,
        )


class Detection(BaseModel):
    """Raw object detection from vision model."""
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_id: int = Field(..., ge=0)
    class_name: Optional[str] = None


class Velocity(BaseModel):
    """Estimated velocity in pixels per frame."""
    vx: float = Field(..., description="Horizontal velocity")
    vy: float = Field(..., description="Vertical velocity")


class TrackedObject(BaseModel):
    """Persistent tracked object across frames."""
    track_id: int
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_id: int
    class_name: Optional[str] = None
    state: TrackState
    age: int = Field(..., description="Total frames since track was initialized")
    time_since_update: int = Field(..., description="Frames elapsed since last detection match")
    velocity: Optional[Velocity] = None


class LatencyMetrics(BaseModel):
    """Detailed pipeline execution breakdown in milliseconds."""
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    track_ms: float
    total_ms: float
    fps: float


class DetectionResponse(BaseModel):
    """Response schema for raw detection endpoint."""
    frame_id: Optional[str] = None
    image_width: int
    image_height: int
    detections_count: int
    detections: List[Detection]
    metrics: LatencyMetrics


class TrackResponse(BaseModel):
    """Response schema for multi-object tracking endpoint."""
    frame_id: Optional[str] = None
    image_width: int
    image_height: int
    active_tracks_count: int
    tracks: List[TrackedObject]
    metrics: LatencyMetrics


class HealthResponse(BaseModel):
    """Response schema for system healthcheck."""
    status: str
    version: str
    uptime_seconds: float
    model_loaded: bool
    model_path: str
    execution_provider: str
    cpu_threads: int


class MetricsResponse(BaseModel):
    """Aggregated runtime performance statistics."""
    total_requests: int
    total_frames_processed: int
    avg_preprocess_ms: float
    avg_inference_ms: float
    avg_postprocess_ms: float
    avg_tracking_ms: float
    avg_total_ms: float
    effective_fps: float
    current_active_tracks: int


class TrajectoryPoint(BaseModel):
    """Individual timestamped position along a track's trajectory."""
    frame_idx: int
    bbox: BoundingBox
    velocity: Optional[Velocity] = None


class VideoTrackSummary(BaseModel):
    """Response schema for batch video stream tracking."""
    filename: str
    total_frames: int
    video_width: int
    video_height: int
    nominal_fps: float
    unique_tracks_count: int
    trajectories: Dict[str, List[TrajectoryPoint]]
    average_processing_fps: float
    total_processing_ms: float
