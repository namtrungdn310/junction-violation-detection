"""
Inference layer: AI model wrappers for detection and OCR.

Modules in this layer may import from core/ and utils/
but never from pipeline/.

Public API
~~~~~~~~~~
- ObjectDetector   : YOLO26 detector with TensorRT / ONNX runtime routing
- InferenceFormat  : Enum describing the active backend
- compile_to_tensorrt : Cross-compile .pt → TensorRT .engine
- export_to_onnx      : Export .pt → ONNX FP16 fallback
"""

from jvd.inference.compiler import InferenceFormat, compile_to_tensorrt, export_to_onnx
from jvd.inference.detector import ObjectDetector

__all__ = [
    "ObjectDetector",
    "InferenceFormat",
    "compile_to_tensorrt",
    "export_to_onnx",
]
