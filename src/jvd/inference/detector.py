"""
detector.py — YOLO26 object detector with TensorRT / ONNX runtime routing.

Layer: inference/  (imports from core/ and inference/compiler)

Loads TensorRT .engine by default, fallback to compile .pt -> .engine,
then fallback to ONNX. Maintains VRAM cache with periodic flushes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import torch

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore[assignment,misc]

from jvd.core.datamodels import BoundingBox, DetectionEvent, VehicleClass
from jvd.core.exceptions import PipelineConfigError
from jvd.inference.compiler import (
    InferenceFormat,
    compile_to_tensorrt,
    export_to_onnx,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_COCO_VEHICLE_IDS  = [2, 3, 5, 7]        # car, motorcycle, bus, truck
_CONF_THRESHOLD    = 0.4
_CACHE_FLUSH_INTERVAL = 10_000           # frames between empty_cache() calls
_IMGSZ             = 640


class ObjectDetector:
    """YOLO26 inference engine with automatic TensorRT → ONNX fallback.

    Usage::
        detector = ObjectDetector("models/yolo26n.pt", device=dm.device)
        detector.load()
        events = detector.predict(frame, frame_id=42, timestamp=1.5)
    """

    def __init__(
        self,
        model_path: str | Path,
        device: torch.device,
        conf: float = _CONF_THRESHOLD,
        imgsz: int = _IMGSZ,
    ) -> None:
        self._model_path = Path(model_path)
        self._device     = device
        self._conf       = conf
        self._imgsz      = imgsz
        self._model      = None
        self._format     = InferenceFormat.PYTORCH
        self._frame_count: int = 0

    # ── Model loading ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the inference model using the best available backend.

        Cascade: TRT .engine -> Compile TRT -> ONNX -> PipelineConfigError
        """
        # ── Attempt 1 & 2: TensorRT ──────────────────────────────────────────
        try:
            engine_path = self._model_path.with_suffix(".engine")
            if not engine_path.exists():
                engine_path = compile_to_tensorrt(
                    self._model_path, self._imgsz, workspace_gb=2
                )
            logger.info(f"Loading TensorRT engine: {engine_path}")
            self._model  = YOLO(str(engine_path))
            self._format = InferenceFormat.TENSORRT
            logger.info("Backend: TensorRT (FP16, static alloc)")
            return

        except Exception as trt_err:
            logger.warning(
                f"TensorRT load/compile failed ({type(trt_err).__name__}: {trt_err}). "
                "Attempting ONNX fallback …"
            )

        # ── Attempt 3: ONNX ──────────────────────────────────────────────────
        try:
            onnx_path = export_to_onnx(self._model_path, self._imgsz)
            logger.info(f"Loading ONNX model: {onnx_path}")
            self._model  = YOLO(str(onnx_path))
            self._format = InferenceFormat.ONNX
            logger.info("Backend: ONNX (FP16, ONNXRuntime-GPU)")
            return

        except Exception as onnx_err:
            logger.error(f"ONNX fallback also failed: {onnx_err}")

        # ── No viable backend ─────────────────────────────────────────────────
        raise PipelineConfigError(
            f"Cannot load model '{self._model_path}' via TensorRT or ONNX. "
            "Refusing to use PyTorch FP32 to avoid VRAM OOM on < 3 GB GPU."
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        frame,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> List[DetectionEvent]:
        """Run YOLO26 NMS-Free inference on one BGR frame.

        Flushes the CUDA allocator cache every ``_CACHE_FLUSH_INTERVAL`` frames
        via ``torch.cuda.empty_cache()`` to prevent long-session OOM.

        Returns:
            List of ``DetectionEvent`` objects (one per detected vehicle).

        Raises:
            PipelineConfigError: If ``load()`` was never called.
        """
        if self._model is None:
            raise PipelineConfigError(
                "ObjectDetector.predict() called before load(). "
                "Call detector.load() once during pipeline startup."
            )

        # ── Periodic VRAM cache flush ─────────────────────────────────────────
        self._frame_count += 1
        if self._frame_count % _CACHE_FLUSH_INTERVAL == 0:
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
                logger.debug(
                    f"[frame {self._frame_count}] "
                    "VRAM allocator cache flushed (empty_cache)."
                )

        # ── YOLO26 inference (End-to-End NMS-Free) ────────────────────────────
        results = self._model.predict(
            source=frame,
            conf=self._conf,
            classes=_COCO_VEHICLE_IDS,
            imgsz=self._imgsz,
            verbose=False,
            device=self._device,
        )

        return self._parse_results(results, frame_id, timestamp)

    # ── Result parsing ────────────────────────────────────────────────────────

    def _parse_results(
        self,
        results: list,
        frame_id: int,
        timestamp: float,
    ) -> List[DetectionEvent]:
        """Convert raw YOLO output tensors → ``DetectionEvent`` list.

        YOLO26 is NMS-Free (End-to-End), so no duplicate suppression is needed.
        Malformed boxes are logged and skipped rather than crashing the pipeline.
        """
        events: List[DetectionEvent] = []
        if not results or results[0].boxes is None:
            return events

        boxes = results[0].boxes
        for i in range(len(boxes)):
            try:
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf     = float(boxes.conf[i])
                coco_id  = int(boxes.cls[i])
                vehicle  = VehicleClass.from_coco_id(coco_id)
                event = DetectionEvent(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=conf,
                    class_label=vehicle,
                )
                events.append(event)
            except (IndexError, ValueError) as exc:
                logger.warning(f"Skipping malformed detection box [{i}]: {exc}")

        return events

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def backend(self) -> InferenceFormat: return self._format

    @property
    def is_ready(self) -> bool: return self._model is not None

    @property
    def raw_model(self): return self._model

    @property
    def device(self) -> "torch.device": return self._device

    @property
    def conf(self) -> float: return self._conf

    @property
    def imgsz(self) -> int: return self._imgsz
