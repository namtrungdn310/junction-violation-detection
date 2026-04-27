"""
Unit tests for Phase 3: YOLO26 ObjectDetector and TensorRT compiler.

Tests cover:
    compiler.py : compile_to_tensorrt (cache hit, missing file, failure),
                  export_to_onnx (cache hit)
    detector.py : ObjectDetector load cascade (TRT → ONNX → error),
                  predict() → DetectionEvent parsing, VRAM flush trigger,
                  predict-before-load guard, COCO class mapping
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch

from jvd.core.datamodels import BoundingBox, DetectionEvent, VehicleClass
from jvd.core.exceptions import PipelineConfigError
from jvd.inference.compiler import InferenceFormat, compile_to_tensorrt, export_to_onnx
from jvd.inference.detector import ObjectDetector, _CACHE_FLUSH_INTERVAL


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture()
def dummy_pt(tmp_path: Path) -> Path:
    """Create a fake .pt file to satisfy existence checks."""
    pt = tmp_path / "yolo26n.pt"
    pt.write_bytes(b"fake_weights")
    return pt


@pytest.fixture()
def dummy_engine(tmp_path: Path) -> Path:
    """Create a fake .engine file to simulate a pre-cached compilation."""
    eng = tmp_path / "yolo26n.engine"
    eng.write_bytes(b"fake_engine")
    return eng


@pytest.fixture()
def dummy_onnx(tmp_path: Path) -> Path:
    """Create a fake .onnx file."""
    onnx = tmp_path / "yolo26n.onnx"
    onnx.write_bytes(b"fake_onnx")
    return onnx


def _make_bgr(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_mock_results(
    boxes_xyxy: List[List[float]],
    confs: List[float],
    cls_ids: List[int],
) -> list:
    """Build a fake ultralytics Results object for predict() testing."""
    result = MagicMock()
    n = len(boxes_xyxy)
    result.boxes.xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32)
    result.boxes.conf = torch.tensor(confs, dtype=torch.float32)
    result.boxes.cls  = torch.tensor(cls_ids, dtype=torch.float32)
    result.boxes.__len__ = MagicMock(return_value=n)
    return [result]


# ── compiler.py tests ─────────────────────────────────────────────────────────

class TestCompileToTensorRT:
    """compile_to_tensorrt() behaviour."""

    def test_raises_on_missing_pt(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Model weights not found"):
            compile_to_tensorrt(tmp_path / "nonexistent.pt")

    def test_cache_hit_skips_compilation(self, dummy_pt, dummy_engine):
        """If .engine already exists next to .pt, skip compilation entirely."""
        # Make engine sit next to the .pt
        engine_copy = dummy_pt.with_suffix(".engine")
        engine_copy.write_bytes(b"cached_engine")

        with patch("jvd.inference.compiler.YOLO") as mock_yolo:
            result = compile_to_tensorrt(dummy_pt)

        mock_yolo.assert_not_called()
        assert result == engine_copy

    def test_compilation_called_when_no_engine(self, dummy_pt):
        """If no .engine exists, YOLO.export() must be invoked."""
        engine_path = dummy_pt.with_suffix(".engine")

        mock_model = MagicMock()
        # Simulate export() creating the .engine file
        def fake_export(**kwargs):
            engine_path.write_bytes(b"compiled_engine")

        mock_model.export.side_effect = fake_export

        with patch("jvd.inference.compiler.YOLO", return_value=mock_model):
            result = compile_to_tensorrt(dummy_pt, workspace_gb=2)

        mock_model.export.assert_called_once_with(
            format="engine", half=True, dynamic=False, imgsz=640, workspace=2
        )
        assert result == engine_path

    def test_raises_when_engine_not_produced(self, dummy_pt):
        """RuntimeError if export() does not produce the .engine file."""
        mock_model = MagicMock()
        mock_model.export.return_value = None   # engine file never created

        with patch("jvd.inference.compiler.YOLO", return_value=mock_model):
            with pytest.raises(RuntimeError, match="engine file not found"):
                compile_to_tensorrt(dummy_pt)


class TestExportToONNX:
    """export_to_onnx() behaviour."""

    def test_cache_hit_skips_export(self, dummy_pt):
        onnx_path = dummy_pt.with_suffix(".onnx")
        onnx_path.write_bytes(b"cached_onnx")

        with patch("jvd.inference.compiler.YOLO") as mock_yolo:
            result = export_to_onnx(dummy_pt)

        mock_yolo.assert_not_called()
        assert result == onnx_path


# ── detector.py — load cascade tests ─────────────────────────────────────────

class TestObjectDetectorLoadCascade:
    """Nested try-except loading strategy."""

    def test_loads_tensorrt_when_engine_present(self, dummy_pt, cpu_device):
        engine_path = dummy_pt.with_suffix(".engine")
        engine_path.write_bytes(b"engine")

        mock_model = MagicMock()
        with patch("jvd.inference.detector.YOLO", return_value=mock_model):
            det = ObjectDetector(dummy_pt, device=cpu_device)
            det.load()

        assert det.backend == InferenceFormat.TENSORRT
        assert det.is_ready

    def test_falls_back_to_onnx_on_trt_failure(self, dummy_pt, cpu_device):
        """If TRT load raises, detector must fall back to ONNX without crashing."""
        onnx_path = dummy_pt.with_suffix(".onnx")
        onnx_path.write_bytes(b"onnx_model")

        call_count = [0]
        def mock_yolo_factory(path_str):
            call_count[0] += 1
            if ".engine" in path_str:
                raise RuntimeError("GPU architecture mismatch — engine invalid")
            return MagicMock()

        with patch("jvd.inference.detector.YOLO", side_effect=mock_yolo_factory):
            with patch("jvd.inference.detector.compile_to_tensorrt",
                       side_effect=RuntimeError("TRT compile failed")):
                with patch("jvd.inference.detector.export_to_onnx",
                           return_value=onnx_path):
                    det = ObjectDetector(dummy_pt, device=cpu_device)
                    det.load()

        assert det.backend == InferenceFormat.ONNX
        assert det.is_ready

    def test_raises_pipeline_error_when_all_backends_fail(self, dummy_pt, cpu_device):
        """No silent FP32 fallback — must raise PipelineConfigError."""
        with patch("jvd.inference.detector.compile_to_tensorrt",
                   side_effect=RuntimeError("TRT failed")):
            with patch("jvd.inference.detector.export_to_onnx",
                       side_effect=RuntimeError("ONNX failed")):
                det = ObjectDetector(dummy_pt, device=cpu_device)
                with pytest.raises(PipelineConfigError):
                    det.load()

    def test_predict_before_load_raises(self, dummy_pt, cpu_device):
        det = ObjectDetector(dummy_pt, device=cpu_device)
        with pytest.raises(PipelineConfigError, match="load\\(\\)"):
            det.predict(_make_bgr())


# ── detector.py — predict() + parsing tests ──────────────────────────────────

class TestObjectDetectorPredict:
    """predict() result parsing and DetectionEvent construction."""

    def _make_loaded_detector(self, dummy_pt, cpu_device) -> ObjectDetector:
        det = ObjectDetector(dummy_pt, device=cpu_device)
        mock_model = MagicMock()
        det._model  = mock_model
        det._format = InferenceFormat.TENSORRT
        return det

    def test_returns_detection_events(self, dummy_pt, cpu_device):
        det = self._make_loaded_detector(dummy_pt, cpu_device)
        mock_results = _make_mock_results(
            boxes_xyxy=[[10, 20, 100, 150]],
            confs=[0.85],
            cls_ids=[2],   # car
        )
        det._model.predict.return_value = mock_results

        events = det.predict(_make_bgr(), frame_id=5, timestamp=1.0)
        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, DetectionEvent)
        assert ev.frame_id == 5
        assert ev.timestamp == 1.0
        assert ev.class_label == VehicleClass.CAR
        assert abs(ev.confidence - 0.85) < 1e-4

    def test_coco_id_mapping(self, dummy_pt, cpu_device):
        det = self._make_loaded_detector(dummy_pt, cpu_device)
        test_cases = [
            (2, VehicleClass.CAR),
            (3, VehicleClass.MOTORBIKE),
            (5, VehicleClass.BUS),
            (7, VehicleClass.TRUCK),
        ]
        for coco_id, expected_class in test_cases:
            mock_results = _make_mock_results([[0, 0, 50, 50]], [0.9], [coco_id])
            det._model.predict.return_value = mock_results
            events = det.predict(_make_bgr())
            assert events[0].class_label == expected_class

    def test_returns_empty_on_no_detections(self, dummy_pt, cpu_device):
        det = self._make_loaded_detector(dummy_pt, cpu_device)
        result = MagicMock()
        result.boxes = None
        det._model.predict.return_value = [result]
        events = det.predict(_make_bgr())
        assert events == []

    def test_bbox_coordinates_preserved(self, dummy_pt, cpu_device):
        det = self._make_loaded_detector(dummy_pt, cpu_device)
        mock_results = _make_mock_results([[15, 25, 200, 300]], [0.7], [7])
        det._model.predict.return_value = mock_results
        events = det.predict(_make_bgr())
        bbox = events[0].bbox
        assert bbox.x1 == pytest.approx(15.0)
        assert bbox.y1 == pytest.approx(25.0)
        assert bbox.x2 == pytest.approx(200.0)
        assert bbox.y2 == pytest.approx(300.0)


class TestVRAMCacheFlush:
    """Verify periodic empty_cache() is called at correct interval."""

    def test_empty_cache_called_at_interval(self, dummy_pt):
        cuda_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        det = ObjectDetector(dummy_pt, device=cuda_device)
        mock_model = MagicMock()
        mock_model.predict.return_value = [MagicMock(boxes=None)]
        det._model  = mock_model
        det._format = InferenceFormat.TENSORRT

        # Set counter just before flush point
        det._frame_count = _CACHE_FLUSH_INTERVAL - 1

        with patch("jvd.inference.detector.torch.cuda.empty_cache") as mock_cache:
            det.predict(_make_bgr())   # This call hits the flush point
            if cuda_device.type == "cuda":
                mock_cache.assert_called_once()
