"""
detector.py — Nhận diện đối tượng bằng YOLO26 hỗ trợ TensorRT / ONNX.

Layer: inference/

Load TensorRT .engine mặc định, nếu không có sẽ tự build từ .pt,
nếu lỗi sẽ fallback về ONNX. Tự động giải phóng VRAM định kỳ.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
)

logger = logging.getLogger(__name__)

# ── Cấu hình Model ────────────────────────────────────────────────────────────
_IMGSZ             = 1024        # Độ phân giải cao cho vật thể xa
_CONF_THRESH       = 0.35        # Ngưỡng tự tin cao để tránh nhiễu
_IOU_THRESH        = 0.3         # NMS chặt để tránh hộp trùng lặp
_COCO_VEHICLE_IDS  = [2, 3, 5, 7]        # xe hơi, xe máy, xe buýt, xe tải
_CACHE_FLUSH_INTERVAL = 10_000           # Tần suất dọn VRAM (số frame)


class ObjectDetector:
    """Engine suy luận YOLO26 với tự động fallback TensorRT → ONNX.

    Cách dùng::
        detector = ObjectDetector("models/yolo26n.pt", device=dm.device)
        detector.load()
        events = detector.predict(frame, frame_id=42, timestamp=1.5)
    """

    def __init__(
        self,
        model_path: str | Path,
        device: torch.device,
        conf: float = _CONF_THRESH,
        imgsz: int = _IMGSZ,
    ) -> None:
        self._model_path = Path(model_path)
        self._device     = device
        self._conf       = conf
        self._imgsz      = imgsz
        self._model      = None
        self._format     = InferenceFormat.PYTORCH
        self._frame_count: int = 0

    # ── Load model ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model sử dụng backend tốt nhất có thể.

        Ưu tiên: TRT .engine -> Compile TRT -> ONNX -> PipelineConfigError
        """
        # ── Cách 1 & 2: TensorRT (chỉ GPU) ──────────────────────────────
        if self._device.type == "cuda":
            try:
                engine_path = self._model_path.with_suffix(".engine")
                if not engine_path.exists():
                    engine_path = compile_to_tensorrt(
                        self._model_path, self._imgsz, workspace_gb=2
                    )
                logger.info(f"Đang load TensorRT engine: {engine_path}")
                self._model = YOLO(str(engine_path), task="detect")
                self._format = InferenceFormat.TENSORRT
                logger.info("Backend: TensorRT (FP16, cấp phát tĩnh)")
                return

            except Exception as trt_err:
                logger.warning(
                    f"Load/compile TensorRT lỗi ({type(trt_err).__name__}: {trt_err}). "
                    "Đang thử fallback sang ONNX …"
                )
        else:
            logger.info("Phát hiện CPU; bỏ qua TensorRT, dùng ONNX fallback.")

        # ── Cách 3: PyTorch gốc (.pt) ───────────────────────────────────────────
        try:
            logger.info(f"Đang load PyTorch model: {self._model_path}")
            self._model = YOLO(str(self._model_path), task="detect")

            # Kiểm tra CUDA kernel bằng cách chạy thử
            import numpy as np
            dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
            self._model.predict(source=dummy, imgsz=self._imgsz, device=self._device, verbose=False)

            self._format = InferenceFormat.PYTORCH
            logger.info(f"Backend: PyTorch ({self._device.type.upper()} FP32)")
            return
        except Exception as pt_err:
            logger.error(f"Load PyTorch lỗi trên {self._device}: {pt_err}")

        # ── Không có backend phù hợp ───────────────────────────────────────────
        raise PipelineConfigError(
            f"Không thể load model '{self._model_path}' trên thiết bị '{self._device}'. "
            f"Nếu dùng GPU, đảm bảo PyTorch hỗ trợ kiến trúc GPU của bạn "
            f"(chạy: python -c \"import torch; print(torch.cuda.get_arch_list())\"). "
            f"Xem https://pytorch.org/get-started/locally/ để biết phiên bản tương thích."
        )


    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        frame,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> list[DetectionEvent]:
        """Chạy suy luận YOLO26 NMS-Free trên 1 frame BGR.

        Xóa cache CUDA định kỳ để tránh OOM khi chạy lâu.

        Returns:
            Danh sách đối tượng ``DetectionEvent`` (mỗi xe một đối tượng).

        Raises:
            PipelineConfigError: Nếu chưa gọi ``load()``.
        """
        if self._model is None:
            raise PipelineConfigError(
                "Đã gọi ObjectDetector.predict() trước load(). "
                "Gọi detector.load() 1 lần khi khởi động pipeline."
            )

        # ── Xóa cache VRAM định kỳ ─────────────────────────────────────────────
        self._frame_count += 1
        if self._frame_count % _CACHE_FLUSH_INTERVAL == 0:
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
                logger.debug(
                    f"[frame {self._frame_count}] "
                    "Đã xóa bộ nhớ đệm VRAM (empty_cache)."
                )

        # ── YOLO26 inference (End-to-End NMS-Free) ────────────────────────────
        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,               # Ngưỡng IOU rõ ràng
            classes=_COCO_VEHICLE_IDS,
            imgsz=self._imgsz,
            verbose=False,
            device=self._device,
            agnostic_nms=True,           # Gộp hộp không phân biệt class
        )

        return self._parse_results(results, frame_id, timestamp)

    # ── Xử lý kết quả ────────────────────────────────────────────────────────

    def _parse_results(
        self,
        results: list,
        frame_id: int,
        timestamp: float,
    ) -> list[DetectionEvent]:
        """Chuyển tensor kết quả YOLO → danh sách ``DetectionEvent``.

        YOLO26 là NMS-Free nên không cần lọc hộp trùng lặp.
        Bỏ qua các hộp lỗi thay vì làm sập pipeline.
        """
        events: list[DetectionEvent] = []
        if not results or results[0].boxes is None:
            return events

        boxes = results[0].boxes
        for i in range(len(boxes)):
            try:
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf     = float(boxes.conf[i])
                coco_id  = int(boxes.cls[i])
                vehicle  = VehicleClass.from_coco_id(coco_id)

                # --- Lọc nâng cao: Loại bỏ người ---
                # Hộp đứng (cao >> rộng) có thể là người, không phải xe.
                w_box = x2 - x1
                h_box = y2 - y1
                if w_box > 0 and (h_box / w_box) > 2.0:
                    # Bỏ qua nếu giống người đang đứng
                    continue

                event = DetectionEvent(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=conf,
                    class_label=vehicle,
                )
                events.append(event)
            except (IndexError, ValueError) as exc:
                logger.warning(f"Bỏ qua hộp lỗi [{i}]: {exc}")

        return events

    # ── Chẩn đoán ───────────────────────────────────────────────────────────

    @property
    def backend(self) -> InferenceFormat: return self._format

    @property
    def is_ready(self) -> bool: return self._model is not None

    @property
    def raw_model(self): return self._model

    @property
    def device(self) -> torch.device: return self._device

    @property
    def conf(self) -> float: return self._conf

    @property
    def imgsz(self) -> int: return self._imgsz
