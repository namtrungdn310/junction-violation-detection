"""
Unit tests for Phase 4: ByteTrackSession and VehicleTrackerManager.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from jvd.core.datamodels import BoundingBox, DetectionEvent, VehicleClass
from jvd.core.exceptions import PipelineConfigError
from jvd.inference.tracker import ByteTrackSession, VehicleTrackerManager


# ── ByteTrackSession Tests ────────────────────────────────────────────────────

def _make_bgr(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)

def _make_mock_detector(is_ready: bool = True):
    det = MagicMock()
    det.is_ready = is_ready
    det.conf = 0.4
    det.imgsz = 640
    det.device = "cpu"
    
    raw_model = MagicMock()
    det.raw_model = raw_model
    return det, raw_model

def _make_track_results(
    boxes_xyxy: List[List[float]],
    confs: List[float],
    cls_ids: List[int],
    track_ids: List[int] | None = None,
) -> list:
    result = MagicMock()
    if len(boxes_xyxy) == 0:
        result.boxes = None
        return [result]
        
    result.boxes.xyxy = np.array(boxes_xyxy)
    result.boxes.conf = np.array(confs)
    result.boxes.cls = np.array(cls_ids)
    if track_ids is not None:
        result.boxes.id = np.array(track_ids)
    else:
        result.boxes.id = None
    result.boxes.__len__ = MagicMock(return_value=len(boxes_xyxy))
    return [result]

class TestByteTrackSession:
    def test_requires_loaded_detector(self):
        det, _ = _make_mock_detector(is_ready=False)
        with pytest.raises(PipelineConfigError, match="loaded detector"):
            ByteTrackSession(det)

    def test_track_calls_model_track_with_persist(self):
        det, raw_model = _make_mock_detector()
        session = ByteTrackSession(det)
        
        raw_model.track.return_value = _make_track_results([], [], [], [])
        frame = _make_bgr()
        
        events = session.track(frame, frame_id=1, timestamp=0.1)
        
        raw_model.track.assert_called_once()
        kwargs = raw_model.track.call_args.kwargs
        assert kwargs["source"] is frame
        assert kwargs["persist"] is True
        assert kwargs["conf"] == 0.4
        assert kwargs["imgsz"] == 640
        assert kwargs["device"] == "cpu"
        assert "tracker" in kwargs

    def test_track_parsing_with_ids(self):
        det, raw_model = _make_mock_detector()
        session = ByteTrackSession(det)
        
        # 1 car, 1 motorcycle
        raw_model.track.return_value = _make_track_results(
            boxes_xyxy=[[10, 10, 50, 50], [100, 100, 120, 120]],
            confs=[0.9, 0.8],
            cls_ids=[2, 3],
            track_ids=[1, 2]
        )
        
        events = session.track(_make_bgr(), frame_id=10, timestamp=1.5)
        
        assert len(events) == 2
        
        ev1 = events[0]
        assert ev1.track_id == 1
        assert ev1.class_label == VehicleClass.CAR
        assert ev1.frame_id == 10
        assert ev1.timestamp == 1.5
        
        ev2 = events[1]
        assert ev2.track_id == 2
        assert ev2.class_label == VehicleClass.MOTORBIKE

    def test_track_parsing_no_ids(self):
        """If tracker returns None for ids, events should have track_id=None."""
        det, raw_model = _make_mock_detector()
        session = ByteTrackSession(det)
        
        raw_model.track.return_value = _make_track_results(
            boxes_xyxy=[[10, 10, 50, 50]], confs=[0.9], cls_ids=[2], track_ids=None
        )
        events = session.track(_make_bgr())
        assert len(events) == 1
        assert events[0].track_id is None

# ── VehicleTrackerManager Tests ───────────────────────────────────────────────

class TestVehicleTrackerManager:
    def _make_event(self, tid: int | None, cx: float, cy: float) -> DetectionEvent:
        # We just need bbox.center to equal (cx, cy)
        # So we can set x1=cx-10, x2=cx+10, y1=cy-10, y2=cy+10
        bbox = BoundingBox(x1=cx-10, y1=cy-10, x2=cx+10, y2=cy+10)
        return DetectionEvent(
            frame_id=1, timestamp=0.0, bbox=bbox, track_id=tid, class_label=VehicleClass.CAR
        )

    def test_ignores_events_without_track_id(self):
        mgr = VehicleTrackerManager()
        ev = self._make_event(None, 100, 100)
        mgr.update([ev], frame_id=1)
        assert mgr.track_count == 0

    def test_records_history_centers(self):
        mgr = VehicleTrackerManager(history_len=3)
        # Track 1 moves (100,100) -> (110,110) -> (120,120)
        mgr.update([self._make_event(1, 100, 100)], frame_id=1)
        mgr.update([self._make_event(1, 110, 110)], frame_id=2)
        mgr.update([self._make_event(1, 120, 120)], frame_id=3)
        
        hist = mgr.get_history(1)
        assert len(hist) == 3
        assert hist[0] == (100.0, 100.0)
        assert hist[2] == (120.0, 120.0)

    def test_history_length_capped(self):
        mgr = VehicleTrackerManager(history_len=2)
        mgr.update([self._make_event(1, 10, 10)], frame_id=1)
        mgr.update([self._make_event(1, 20, 20)], frame_id=2)
        mgr.update([self._make_event(1, 30, 30)], frame_id=3)
        
        hist = mgr.get_history(1)
        assert len(hist) == 2
        assert hist[0] == (20.0, 20.0)
        assert hist[1] == (30.0, 30.0)

    def test_velocity_calculation(self):
        mgr = VehicleTrackerManager()
        assert mgr.get_velocity(1) is None
        
        mgr.update([self._make_event(1, 100, 100)], frame_id=1)
        assert mgr.get_velocity(1) is None  # Only 1 point
        
        mgr.update([self._make_event(1, 115, 120)], frame_id=2)
        vx, vy = mgr.get_velocity(1)
        assert vx == pytest.approx(15.0)
        assert vy == pytest.approx(20.0)

    def test_cleanup_stale(self):
        mgr = VehicleTrackerManager(max_stale_frames=5)
        mgr.update([self._make_event(1, 10, 10), self._make_event(2, 20, 20)], frame_id=10)
        mgr.update([self._make_event(2, 25, 25)], frame_id=12)  # Update track 2
        
        # Frame 15: difference is 5 for track 1, which is <= 5, so not stale yet
        assert mgr.cleanup_stale(15) == 0
        assert mgr.track_count == 2
        
        # Frame 16: diff for track 1 is 6 > 5. Track 1 evicted. Track 2 diff is 4 <= 5.
        assert mgr.cleanup_stale(16) == 1
        assert mgr.track_count == 1
        assert mgr.active_track_ids == {2}
