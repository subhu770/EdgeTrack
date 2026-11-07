"""High-Performance Edge ONNX Runtime Object Detector.

Optimized for CPU edge inference with INT8 quantization, graph optimizations,
letterbox transformation, and fast NMS post-processing.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import onnxruntime as ort

from app.core.config import get_settings
from app.core.logging import logger
from app.models.schemas import BoundingBox, Detection

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    quantize_dynamic = None
    QuantType = None


class ONNXDetector:
    """Production ONNX Runtime object detector with CPU graph optimizations."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        nms_iou_threshold: Optional[float] = None,
        input_size: Optional[Tuple[int, int]] = None,
        num_threads: Optional[int] = None,
    ) -> None:
        """Initialize the ONNX Runtime session and inference configuration.

        Args:
            model_path: Path to ONNX weights (.onnx). If not found, a lightweight
                        fallback test model is created.
            conf_threshold: Minimum detection confidence score.
            nms_iou_threshold: Non-maximum suppression IoU threshold.
            input_size: Target (width, height) for model input.
            num_threads: Number of CPU intra-op threads.
        """
        settings = get_settings()
        self.model_path = model_path or settings.MODEL_PATH
        self.conf_threshold = conf_threshold or settings.CONF_THRESHOLD
        self.nms_iou_threshold = nms_iou_threshold or settings.NMS_IOU_THRESHOLD
        self.input_width = input_size[0] if input_size else settings.INPUT_WIDTH
        self.input_height = input_size[1] if input_size else settings.INPUT_HEIGHT
        self.num_threads = num_threads or settings.ORT_NUM_THREADS
        self.classes = settings.COCO_CLASSES

        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = "images"
        self.output_names: List[str] = []

        self._ensure_model_exists()
        self._init_session()

    def _ensure_model_exists(self) -> None:
        """Verify model file exists on disk, or synthesize a valid test ONNX model."""
        if os.path.exists(self.model_path):
            return

        logger.warning(
            f"Model not found at '{self.model_path}'. Synthesizing lightweight edge ONNX model..."
        )
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        self.generate_dummy_model(self.model_path)

    @staticmethod
    def generate_dummy_model(output_path: str) -> None:
        """Create a valid lightweight YOLOv8-compatible ONNX model for testing and edge verification.

        Args:
            output_path: Destination file path for generated ONNX model.
        """
        import onnx
        from onnx import helper, TensorProto

        inp_tensor = helper.make_tensor_value_info(
            "images", TensorProto.FLOAT, [1, 3, 640, 640]
        )
        out_tensor = helper.make_tensor_value_info(
            "output0", TensorProto.FLOAT, [1, 84, 8400]
        )

        # Create structured sample detections:
        # Box 1: Center person moving across center
        # Box 2: Vehicle moving across lower third
        dummy_matrix = np.zeros((1, 84, 8400), dtype=np.float32)
        # Detection 1: Person (class_id=0), [cx=320, cy=240, w=80, h=160, score=0.88]
        dummy_matrix[0, 0, 0] = 320.0
        dummy_matrix[0, 1, 0] = 240.0
        dummy_matrix[0, 2, 0] = 80.0
        dummy_matrix[0, 3, 0] = 160.0
        dummy_matrix[0, 4, 0] = 0.88

        # Detection 2: Car (class_id=2), [cx=480, cy=380, w=140, h=90, score=0.76]
        dummy_matrix[0, 0, 1] = 480.0
        dummy_matrix[0, 1, 1] = 380.0
        dummy_matrix[0, 2, 1] = 140.0
        dummy_matrix[0, 3, 1] = 90.0
        dummy_matrix[0, 6, 1] = 0.76

        const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["output0"],
            value=helper.make_tensor(
                "output_data",
                TensorProto.FLOAT,
                [1, 84, 8400],
                dummy_matrix.flatten().tolist(),
            ),
        )

        graph = helper.make_graph(
            [const_node], "edgetrack_synthetic_yolov8", [inp_tensor], [out_tensor]
        )
        model = helper.make_model(graph, producer_name="EdgeTrack", ir_version=10)
        model.opset_import[0].version = 17
        onnx.save(model, output_path)
        logger.info(f"Synthetic test ONNX model generated at '{output_path}'.")

    def _init_session(self) -> None:
        """Configure ONNX Runtime session with edge CPU graph optimizations."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        providers = ["CPUExecutionProvider"]
        logger.info(
            f"Loading ONNX model '{self.model_path}' on CPU with {self.num_threads} threads..."
        )
        self.session = ort.InferenceSession(
            self.model_path, sess_options=sess_options, providers=providers
        )

        # Inspect model inputs and outputs
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        logger.info(
            f"ONNX session loaded successfully. Input: {self.input_name}, "
            f"Outputs: {self.output_names}"
        )

    @staticmethod
    def letterbox(
        image: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Resize image keeping aspect ratio and apply symmetric border padding.

        Args:
            image: Original image as BGR uint8 NumPy array.
            new_shape: Target dimensions (height, width).
            color: Padding border color (default YOLO gray 114).

        Returns:
            padded_image: Resized and padded image.
            ratio: Scale factor applied to original image.
            (dw, dh): (horizontal_padding, vertical_padding) applied.
        """
        shape = image.shape[:2]  # (height, width)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute unpadded new dimensions
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (new_shape[1] - new_unpad[0]) / 2.0  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2.0  # height padding

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return image, r, (dw, dh)

    def preprocess(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Preprocess frame for ONNX inference.

        Args:
            image: BGR image (H, W, 3).

        Returns:
            blob: Contiguous float32 tensor of shape (1, 3, H, W) normalized to [0, 1].
            ratio: Scaling factor.
            pad: (dw, dh) padding offsets.
        """
        padded_img, ratio, (dw, dh) = self.letterbox(
            image, new_shape=(self.input_height, self.input_width)
        )

        # Convert BGR -> RGB
        rgb_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)

        # HWC -> CHW, normalize to [0.0, 1.0]
        blob = rgb_img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        blob = np.ascontiguousarray(blob)

        return blob, ratio, (dw, dh)

    def postprocess(
        self,
        raw_output: Union[np.ndarray, List[np.ndarray]],
        orig_shape: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
    ) -> List[Detection]:
        """Parse raw model tensor outputs, invert letterboxing, and apply NMS.

        Args:
            raw_output: Output tensor(s) from ONNX session.
            orig_shape: (height, width) of the original unpadded input image.
            ratio: Scale ratio applied during preprocessing.
            pad: (dw, dh) padding applied during letterboxing.

        Returns:
            List of filtered Detection objects.
        """
        output = raw_output[0] if isinstance(raw_output, (list, tuple)) else raw_output

        # Handle different YOLO output shapes:
        # 1. YOLOv8 format: (1, 84, 8400) -> transpose to (8400, 84)
        if output.ndim == 3 and output.shape[1] < output.shape[2]:
            preds = output[0].T
        elif output.ndim == 3:
            preds = output[0]
        else:
            preds = output

        num_features = preds.shape[1] if preds.ndim == 2 else 0
        if num_features < 5:
            return []

        # YOLOv8: 4 box coords + C classes
        # YOLOv5/7: 4 box coords + 1 objectness + C classes
        is_yolov5_format = (num_features > 84) or (num_features == 85)

        if is_yolov5_format:
            boxes = preds[:, :4]
            obj_conf = preds[:, 4:5]
            cls_scores = preds[:, 5:] * obj_conf
            confidences = np.max(cls_scores, axis=1)
            class_ids = np.argmax(cls_scores, axis=1)
        else:
            boxes = preds[:, :4]
            cls_scores = preds[:, 4:]
            confidences = np.max(cls_scores, axis=1)
            class_ids = np.argmax(cls_scores, axis=1)

        # Filter by confidence threshold
        mask = confidences >= self.conf_threshold
        if not np.any(mask):
            return []

        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert (cx, cy, w, h) -> (x1, y1, x2, y2)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        # Invert letterbox transformations to map back to original image space
        dw, dh = pad
        orig_h, orig_w = orig_shape
        x1 = np.clip((x1 - dw) / ratio, 0.0, orig_w)
        y1 = np.clip((y1 - dh) / ratio, 0.0, orig_h)
        x2 = np.clip((x2 - dw) / ratio, 0.0, orig_w)
        y2 = np.clip((y2 - dh) / ratio, 0.0, orig_h)

        # Format for OpenCV NMS: [x, y, w, h]
        nms_boxes = []
        for i in range(len(x1)):
            nms_boxes.append([
                float(x1[i]),
                float(y1[i]),
                float(x2[i] - x1[i]),
                float(y2[i] - y1[i]),
            ])

        indices = cv2.dnn.NMSBoxes(
            bboxes=nms_boxes,
            scores=confidences.tolist(),
            score_threshold=float(self.conf_threshold),
            nms_threshold=float(self.nms_iou_threshold),
        )

        detections: List[Detection] = []
        if len(indices) == 0:
            return detections

        indices = indices.flatten()
        for idx in indices:
            cid = int(class_ids[idx])
            cname = self.classes[cid] if cid < len(self.classes) else f"class_{cid}"
            detections.append(
                Detection(
                    bbox=BoundingBox(
                        x1=round(float(x1[idx]), 2),
                        y1=round(float(y1[idx]), 2),
                        x2=round(float(x2[idx]), 2),
                        y2=round(float(y2[idx]), 2),
                    ),
                    confidence=round(float(confidences[idx]), 4),
                    class_id=cid,
                    class_name=cname,
                )
            )

        return detections

    def detect(self, image: np.ndarray) -> Tuple[List[Detection], dict]:
        """Execute end-to-end detection pipeline with latency breakdown.

        Args:
            image: BGR uint8 NumPy image.

        Returns:
            detections: List of Detection instances.
            latencies: Dictionary with breakdown (preprocess_ms, inference_ms, postprocess_ms).
        """
        import time

        orig_shape = image.shape[:2]

        t0 = time.perf_counter()
        blob, ratio, pad = self.preprocess(image)
        t1 = time.perf_counter()

        raw_outputs = self.session.run(self.output_names, {self.input_name: blob})
        t2 = time.perf_counter()

        detections = self.postprocess(raw_outputs, orig_shape, ratio, pad)
        t3 = time.perf_counter()

        latencies = {
            "preprocess_ms": round((t1 - t0) * 1000.0, 3),
            "inference_ms": round((t2 - t1) * 1000.0, 3),
            "postprocess_ms": round((t3 - t2) * 1000.0, 3),
        }

        return detections, latencies

    @staticmethod
    def quantize_model_to_int8(input_fp32_path: str, output_int8_path: str) -> bool:
        """Apply dynamic INT8 quantization to an FP32 ONNX model for CPU edge acceleration.

        Args:
            input_fp32_path: Path to source FP32 ONNX model.
            output_int8_path: Path to write quantized INT8 ONNX model.

        Returns:
            True if quantization succeeded, False otherwise.
        """
        if quantize_dynamic is None:
            logger.error("onnxruntime.quantization is not installed or available.")
            return False

        try:
            logger.info(f"Quantizing '{input_fp32_path}' to dynamic INT8 -> '{output_int8_path}'...")
            quantize_dynamic(
                model_input=input_fp32_path,
                model_output=output_int8_path,
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=False,
            )
            logger.info("Dynamic INT8 quantization completed successfully.")
            return True
        except Exception as exc:
            logger.error(f"Quantization failed: {exc}", exc_info=True)
            return False

