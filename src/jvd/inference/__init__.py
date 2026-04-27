"""
Inference layer: AI model wrappers for detection and OCR.

Modules in this layer may import from core/ and utils/
but never from pipeline/.

Public API
~~~~~~~~~~
- ObjectDetector        : YOLO26 detector with TensorRT / ONNX runtime routing
- InferenceFormat       : Enum describing the active backend
- compile_to_tensorrt   : Cross-compile .pt → TensorRT .engine
- export_to_onnx        : Export .pt → ONNX FP16 fallback
- ByteTrackSession      : Object tracking using ByteTrack
- VehicleTrackerManager : Trajectory history and management
- LicensePlateRecognizer: Multiprocessed OCR
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ObjectDetector",
    "InferenceFormat",
    "compile_to_tensorrt",
    "export_to_onnx",
    "ByteTrackSession",
    "VehicleTrackerManager",
    "LicensePlateRecognizer",
]

_EXPORT_MAP = {
    "ObjectDetector": ("jvd.inference.detector", "ObjectDetector"),
    "InferenceFormat": ("jvd.inference.compiler", "InferenceFormat"),
    "compile_to_tensorrt": ("jvd.inference.compiler", "compile_to_tensorrt"),
    "export_to_onnx": ("jvd.inference.compiler", "export_to_onnx"),
    "ByteTrackSession": ("jvd.inference.tracker", "ByteTrackSession"),
    "VehicleTrackerManager": ("jvd.inference.tracker", "VehicleTrackerManager"),
    "LicensePlateRecognizer": ("jvd.inference.ocr", "LicensePlateRecognizer"),
}


def __getattr__(name: str):
    """Lazy-load inference symbols to keep import-time overhead minimal."""
    target = _EXPORT_MAP.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = target
    module = import_module(module_name)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
