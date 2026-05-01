"""
compiler.py — Biên dịch TensorRT với fallback ONNX.

Layer: inference/ (chỉ import từ core/)

So sánh các định dạng inference:
+-------------------+----------+----------+---------------------------+
| Định dạng         | Độ chính | VRAM     | Ghi chú                   |
+-------------------+----------+----------+---------------------------+
| PyTorch .pt       | FP32     | Rất cao  | Dễ OOM trên < 3 GB        |
| ONNX (.onnx)      | FP16     | Trung bình | Đa nền tảng, nhanh      |
| TensorRT .engine  | FP16     | Rất thấp | Tối ưu layer, alloc tĩnh  |
+-------------------+----------+----------+---------------------------+

Tham số biên dịch TensorRT:
- half=True      : Chuyển FP32 → FP16; tăng gấp đôi thông lượng Tensor Core
- dynamic=False  : Input shape cố định (1024×1024); TRT cấp phát VRAM khi build
- workspace=2    : Giới hạn bộ nhớ scratch của trình tối ưu TRT ở 2 GB
- imgsz=1024     : Độ phân giải input YOLO (phải khớp với cấu hình detector)
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover — cần ultralytics khi chạy thực tế
    YOLO = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ── Enum định dạng inference ──────────────────────────────────────────────────

class InferenceFormat(Enum):
    """Backend inference được chọn sau khi kiểm tra phần cứng."""
    TENSORRT = auto()   # .engine — nhanh nhất, chỉ chạy trên NVIDIA
    ONNX     = auto()   # .onnx   — fallback đa nền tảng qua ONNXRuntime
    PYTORCH  = auto()   # .pt     — phương án cuối (tốn VRAM nhất)


# ── Biên dịch TensorRT ────────────────────────────────────────────────────────

def compile_to_tensorrt(
    model_path: str | Path,
    imgsz: int = 1024,
    workspace_gb: int = 2,
) -> Path:
    """
    Biên dịch model YOLO .pt sang file TensorRT .engine.

    File .engine được đặt cùng thư mục với .pt và được tái sử dụng
    ở các lần chạy sau (không biên dịch lại nếu đã tồn tại).

    Tham số biên dịch:
    - ``format="engine"``     : Dùng backend TensorRT của NVIDIA.
    - ``half=True``           : FP16 mixed precision (2× thông lượng trên Ampere+).
    - ``dynamic=False``       : Input shape cố định; TRT cấp phát VRAM khi build,
                                không bị phân mảnh bộ nhớ khi chạy.
    - ``workspace=workspace_gb``: Giới hạn scratch buffer của trình tối ưu.
                                Đặt 2 GB để còn >= 200 MB cho OS trên card 3 GB.
    - ``imgsz=imgsz``         : Phải khớp với độ phân giải dùng khi inference.

    Args:
        model_path:   Đường dẫn đến file weight YOLO26 .pt.
        imgsz:        Độ phân giải input (vuông). Mặc định 1024.
        workspace_gb: Giới hạn bộ nhớ scratch TRT (GB). Mặc định 2.

    Returns:
        Đường dẫn đến file .engine đã biên dịch.

    Raises:
        FileNotFoundError: Nếu model_path không tồn tại.
        RuntimeError:      Nếu biên dịch TRT thất bại.
    """
    from ultralytics import YOLO as _YOLO  # noqa: F401

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")

    engine_path = model_path.with_suffix(".engine")
    if engine_path.exists():
        logger.info(f"TensorRT engine đã tồn tại, bỏ qua biên dịch: {engine_path}")
        return engine_path

    logger.info(
        f"Đang biên dịch {model_path.name} → TensorRT engine "
        f"(FP16, imgsz={imgsz}, workspace={workspace_gb} GB) …"
    )
    model = YOLO(str(model_path))
    model.export(
        format="engine",
        half=True,
        dynamic=False,
        imgsz=imgsz,
        workspace=workspace_gb,
    )
    if not engine_path.exists():
        raise RuntimeError(
            f"Biên dịch TensorRT hoàn tất nhưng không tìm thấy file: {engine_path}"
        )
    logger.info(f"TensorRT engine sẵn sàng: {engine_path}")
    return engine_path


# ── Export ONNX (fallback) ────────────────────────────────────────────────────

def export_to_onnx(
    model_path: str | Path,
    imgsz: int = 1024,
) -> Path:
    """
    Export model YOLO .pt sang ONNX FP16 để dùng khi không có TensorRT.

    ONNX dùng ONNXRuntime-GPU để inference, tránh yêu cầu biên dịch phụ thuộc
    phần cứng như TRT, nhưng VRAM cao hơn TRT và thấp hơn PyTorch FP32.

    Args:
        model_path: Đường dẫn đến file weight YOLO26 .pt.
        imgsz:      Độ phân giải input (vuông). Mặc định 1024.

    Returns:
        Đường dẫn đến file .onnx đã export.
    """
    model_path = Path(model_path)
    onnx_path = model_path.with_suffix(".onnx")
    if onnx_path.exists():
        logger.info(f"ONNX model đã tồn tại, bỏ qua export: {onnx_path}")
        return onnx_path

    logger.info(f"Đang export {model_path.name} → ONNX (FP16, imgsz={imgsz}) …")
    model = YOLO(str(model_path))
    model.export(format="onnx", half=True, dynamic=False, imgsz=imgsz)
    logger.info(f"ONNX model sẵn sàng: {onnx_path}")
    return onnx_path
