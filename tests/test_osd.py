"""
Unit tests for Phase 8: OSDRenderer and ViolationReporter.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

from jvd.core.datamodels import BoundingBox, DetectionEvent, VehicleClass
from jvd.pipeline.analyzer import RegionOfInterest, TrackState
from jvd.pipeline.osd import OSDRenderer
from jvd.pipeline.reporter import ViolationReporter

def _make_event(tid: int) -> DetectionEvent:
    return DetectionEvent(
        frame_id=1,
        timestamp=0.0,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
        track_id=tid,
        class_label=VehicleClass.CAR
    )

class TestOSDRenderer:
    def test_draw_roi_and_boxes(self):
        renderer = OSDRenderer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        roi = RegionOfInterest([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        
        # 1 normal, 1 emergency, 1 blocked, 1 violation
        ev_norm = _make_event(1)
        ev_emerg = _make_event(2)
        ev_block = _make_event(3)
        ev_viol = _make_event(4)
        
        states = {
            3: TrackState(is_blocked=True),
            4: TrackState(violation_triggered=True)
        }
        emergency_ids = {2}
        ocr_results = {4: "43A-12345"}
        
        out = renderer.draw(
            frame, 
            [ev_norm, ev_emerg, ev_block, ev_viol],
            roi,
            states,
            emergency_ids,
            ocr_results
        )
        
        # Verify it doesn't mutate original
        assert np.array_equal(frame, np.zeros((100, 100, 3), dtype=np.uint8))
        # Ensure something was drawn
        assert out.max() > 0

class TestViolationReporter:
    def test_reporter_lifecycle(self, tmp_path: Path):
        reporter = ViolationReporter(export_dir=str(tmp_path), fps=30)
        reporter.pre_event_len = 2
        reporter.post_event_len = 2
        
        # Add pre-event frames
        f1 = np.zeros((10, 10, 3), dtype=np.uint8)
        f2 = np.ones((10, 10, 3), dtype=np.uint8) * 255
        reporter.add_frame(f1)
        reporter.add_frame(f2)
        
        # Trigger violation
        wide_shot = np.ones((10, 10, 3), dtype=np.uint8)
        lp_crop = np.zeros((5, 5, 3), dtype=np.uint8)
        reporter.trigger_violation(99, 10.0, "29A-1111", wide_shot, lp_crop)
        
        assert 99 in reporter._pending_exports
        
        # Add post-event frames
        f3 = np.ones((10, 10, 3), dtype=np.uint8) * 100
        reporter.add_frame(f3)
        assert 99 in reporter._pending_exports
        
        with patch("cv2.VideoWriter") as mock_writer:
            reporter.add_frame(f3) # This should complete the job
            
        assert 99 not in reporter._pending_exports
        
        # Check files
        base_name = "violation_99_10"
        json_path = tmp_path / f"{base_name}.json"
        wide_path = tmp_path / f"{base_name}_wide.jpg"
        crop_path = tmp_path / f"{base_name}_crop.jpg"
        
        assert json_path.exists()
        assert wide_path.exists()
        assert crop_path.exists()
        
        # Verify JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert data["license_plate"] == "29A-1111"
            assert data["violation_type"] == "Hatched_Marking_Stop"
            assert data["camera_id"] == "CAM_01"
