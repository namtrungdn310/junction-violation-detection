"""
compiler.py — TensorRT cross-compilation with ONNX fallback.

Layer: inference/  (imports from core/ only)

Inference format strategy
~~~~~~~~~~~~~~~~~~~~~~~~~
+-------------------+----------+----------+---------------------------+
| Format            | Precision| VRAM Use | Notes                     |
+-------------------+----------+----------+---------------------------+
| PyTorch .pt       | FP32     | Very High| OOM risk on < 3 GB        |
| ONNX (.onnx)      | FP16     | Medium   | Cross-platform, fast      |
| TensorRT .engine  | FP16     | Very Low | Layer-fused, static alloc |
+-------------------+----------+----------+---------------------------+

Compilation flags (TensorRT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- half=True      : FP32 → FP16 weight conversion; doubles Tensor Core throughput
- dynamic=False  : Static input shape (640×640); TRT pre-allocates memory at
                   build time → no runtime fragmentation
- workspace=2    : Caps TRT optimiser scratch memory at 2 GB; leaves headroom
                   for OS and inference context on ≤ 3 GB VRAM cards
- imgsz=640      : Standard YOLO input resolution
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover — ultralytics required at runtime
    YOLO = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ── Format enum ───────────────────────────────────────────────────────────────

class InferenceFormat(Enum):
    """Runtime inference backend selected after hardware probing."""
    TENSORRT = auto()   # .engine — fastest, NVIDIA-specific
    ONNX     = auto()   # .onnx   — portable fallback via ONNXRuntime
    PYTORCH  = auto()   # .pt     — last resort (high VRAM cost)


# ── TensorRT compilation ──────────────────────────────────────────────────────

def compile_to_tensorrt(
    model_path: str | Path,
    imgsz: int = 640,
    workspace_gb: int = 2,
) -> Path:
    """
    Cross-compile a YOLO .pt model to a TensorRT .engine file.

    The compiled engine is placed alongside the source .pt file and reused
    on subsequent runs; compilation is skipped if the .engine already exists.

    Compilation parameters
    ~~~~~~~~~~~~~~~~~~~~~~
    - ``format="engine"``   : Selects NVIDIA TensorRT backend.
    - ``half=True``         : Enables FP16 mixed precision.  On Ampere+ GPUs
                              Tensor Cores deliver 2× throughput over FP32.
    - ``dynamic=False``     : Fixes input shape to (1, 3, imgsz, imgsz).
                              TRT pre-allocates VRAM at compile time, eliminating
                              runtime fragmentation on memory-constrained cards.
    - ``workspace=workspace_gb``: Caps the TRT optimiser scratch buffer.
                              Set to 2 GB to leave ≥ 200 MB for the OS on a
                              3 GB card.  Reducing this may slow compilation
                              but will never cause inference OOM.
    - ``imgsz=imgsz``       : Must match the input resolution used at runtime.

    Args:
        model_path:   Path to the YOLO26 .pt weight file.
        imgsz:        Input resolution (square). Default 640.
        workspace_gb: TRT optimiser scratch memory cap in GB. Default 2.

    Returns:
        Path to the compiled .engine file.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        RuntimeError:      If TRT compilation fails (caller should catch and
                           fall back to ONNX).
    """
    from ultralytics import YOLO as _YOLO  # noqa: F401 — re-import for local alias

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    engine_path = model_path.with_suffix(".engine")
    if engine_path.exists():
        logger.info(f"TensorRT engine already exists, skipping compilation: {engine_path}")
        return engine_path

    logger.info(
        f"Compiling {model_path.name} → TensorRT engine "
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
            f"TensorRT export completed but engine file not found: {engine_path}"
        )
    logger.info(f"TensorRT engine ready: {engine_path}")
    return engine_path


# ── ONNX fallback export ──────────────────────────────────────────────────────

def export_to_onnx(
    model_path: str | Path,
    imgsz: int = 640,
) -> Path:
    """
    Export a YOLO .pt model to ONNX FP16 format as a fallback backend.

    ONNX uses ONNXRuntime-GPU for inference, which avoids TRT's hardware-
    specific compilation while still offering GPU acceleration and FP16
    support.  VRAM usage is higher than TRT but substantially lower than
    the native PyTorch FP32 graph.

    Args:
        model_path: Path to the YOLO26 .pt weight file.
        imgsz:      Input resolution (square). Default 640.

    Returns:
        Path to the exported .onnx file.
    """
    model_path = Path(model_path)
    onnx_path = model_path.with_suffix(".onnx")
    if onnx_path.exists():
        logger.info(f"ONNX model already exists, skipping export: {onnx_path}")
        return onnx_path

    logger.info(f"Exporting {model_path.name} → ONNX (FP16, imgsz={imgsz}) …")
    model = YOLO(str(model_path))
    model.export(format="onnx", half=True, dynamic=False, imgsz=imgsz)
    logger.info(f"ONNX model ready: {onnx_path}")
    return onnx_path
