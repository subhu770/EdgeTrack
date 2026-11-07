"""Application configuration using Pydantic Settings."""

import os
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """EdgeTrack application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Project metadata
    PROJECT_NAME: str = "EdgeTrack"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "High-performance edge multi-object tracking engine utilizing ONNX Runtime INT8 "
        "inference and pure NumPy/SciPy ByteTrack state estimation."
    )
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # ONNX Runtime Inference settings
    MODEL_PATH: str = "models/yolov8n.onnx"
    INT8_MODEL_PATH: str = "models/yolov8n_int8.onnx"
    USE_INT8: bool = False
    INPUT_WIDTH: int = 640
    INPUT_HEIGHT: int = 640
    CONF_THRESHOLD: float = 0.25
    NMS_IOU_THRESHOLD: float = 0.45
    ORT_NUM_THREADS: int = max(1, os.cpu_count() or 4)
    ORT_INTER_OP_THREADS: int = 1
    ORT_GRAPH_OPTIMIZATION: str = "ORT_ENABLE_ALL"

    # ByteTrack Tracking parameters
    TRACK_HIGH_THRESH: float = 0.5
    TRACK_LOW_THRESH: float = 0.1
    NEW_TRACK_THRESH: float = 0.6
    MATCH_THRESH: float = 0.8
    MATCH_THRESH_SECOND: float = 0.5
    MATCH_THRESH_UNCONFIRMED: float = 0.7
    MAX_LOST_FRAMES: int = 30
    MIN_CONSECUTIVE_FRAMES: int = 2

    # Standard COCO 80 Classes
    COCO_CLASSES: List[str] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]


@lru_cache()
def get_settings() -> Settings:
    """Return cached instance of application settings."""
    return Settings()

