"""
Unit tests for Phase 5: ViolationAnalyzer and RegionOfInterest.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jvd.core.datamodels import BoundingBox, DetectionEvent, VehicleClass
from jvd.pipeline.analyzer import RegionOfInterest, ViolationAnalyzer


def test_roi_contains():
    # A simple square ROI from (0.2, 0.2) to (0.8, 0.8)
    norm_pts = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    roi = RegionOfInterest(norm_pts)
    
    w, h = 1000, 1000
    # Inside point
    assert roi.contains((500, 500), w, h) is True
    # Edge point
    assert roi.contains((200, 500), w, h) is True
    # Outside point
    assert roi.contains((100, 500), w, h) is False

def test_roi_requires_polygon():
    with pytest.raises(ValueError, match="at least 3 points"):
        RegionOfInterest([(0.0, 0.0), (1.0, 1.0)])

def _make_event(tid: int, timestamp: float, x1: float, y1: float, x2: float, y2: float) -> DetectionEvent:
    return DetectionEvent(
        frame_id=1,
        timestamp=timestamp,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        track_id=tid,
        class_label=VehicleClass.CAR
    )

class TestViolationAnalyzer:
    def setup_method(self):
        # ROI covers the whole frame for simplicity
        self.analyzer = ViolationAnalyzer([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        self.mock_tracker = MagicMock()
        self.w, self.h = 100, 100

    def test_moving_vehicle_no_violation(self):
        ev = _make_event(1, 0.0, 10, 10, 30, 30)
        self.mock_tracker.get_velocity.return_value = (5.0, 5.0)  # fast moving
        
        violations = self.analyzer.analyze([ev], self.mock_tracker, self.w, self.h)
        assert len(violations) == 0
        assert 1 not in self.analyzer.states or self.analyzer.states[1].first_stop_time is None

    def test_stopped_vehicle_triggers_violation_after_time(self):
        self.mock_tracker.get_velocity.return_value = (0.5, 0.5)  # stopped
        
        # Frame 1: Stop begins
        ev1 = _make_event(1, 1.0, 10, 10, 30, 30)
        viols1 = self.analyzer.analyze([ev1], self.mock_tracker, self.w, self.h)
        assert len(viols1) == 0
        assert self.analyzer.states[1].first_stop_time == 1.0
        
        # Frame 2: Still stopped but only 2 seconds passed
        ev2 = _make_event(1, 3.0, 10, 10, 30, 30)
        viols2 = self.analyzer.analyze([ev2], self.mock_tracker, self.w, self.h)
        assert len(viols2) == 0
        
        # Frame 3: 3.5 seconds passed -> Violation
        ev3 = _make_event(1, 4.5, 10, 10, 30, 30)
        viols3 = self.analyzer.analyze([ev3], self.mock_tracker, self.w, self.h)
        assert len(viols3) == 1
        assert viols3[0].dwell_time_seconds == 3.5
        assert self.analyzer.states[1].violation_triggered is True
        
        # Frame 4: Does not trigger again
        ev4 = _make_event(1, 5.5, 10, 10, 30, 30)
        viols4 = self.analyzer.analyze([ev4], self.mock_tracker, self.w, self.h)
        assert len(viols4) == 0

    def test_congestion_exemption_blocked_by_traffic(self):
        self.mock_tracker.get_velocity.return_value = (0.0, 0.0)  # all stopped
        
        # Car 1 is behind Car 2
        # Car 2 (blocking): y1=30, y2=60
        # Car 1 (blocked): y1=65, y2=90
        # They have horizontal overlap
        
        ev_blocked = _make_event(1, 10.0, 10, 65, 40, 90)
        ev_blocking = _make_event(2, 10.0, 15, 30, 35, 60)
        
        self.analyzer.analyze([ev_blocked, ev_blocking], self.mock_tracker, self.w, self.h)
        
        assert self.analyzer.states[1].is_blocked is True
        assert self.analyzer.states[2].is_blocked is False  # Front car is not blocked

        # Jump forward 5 seconds
        ev_blocked_late = _make_event(1, 15.0, 10, 65, 40, 90)
        ev_blocking_late = _make_event(2, 15.0, 15, 30, 35, 60)
        
        violations = self.analyzer.analyze([ev_blocked_late, ev_blocking_late], self.mock_tracker, self.w, self.h)
        # Only car 2 should violate, car 1 is exempted
        assert len(violations) == 1
        assert violations[0].event.track_id == 2

    def test_stale_track_eviction(self):
        ev = _make_event(1, 1.0, 10, 10, 30, 30)
        self.mock_tracker.get_velocity.return_value = (0.0, 0.0)
        
        self.analyzer.analyze([ev], self.mock_tracker, self.w, self.h)
        assert 1 in self.analyzer.states
        
        # Next frame without track 1
        self.analyzer.analyze([], self.mock_tracker, self.w, self.h)
        assert 1 not in self.analyzer.states

    def test_vehicle_outside_roi(self):
        # Make an ROI on the right side of the image
        analyzer = ViolationAnalyzer([(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)])
        self.mock_tracker.get_velocity.return_value = (0.0, 0.0)
        
        # Vehicle on the left (x=10..30)
        ev_out = _make_event(1, 1.0, 10, 10, 30, 30)
        analyzer.analyze([ev_out], self.mock_tracker, 100, 100)
        assert 1 not in analyzer.states
        
        # Vehicle on the right (x=60..80)
        ev_in = _make_event(2, 1.0, 60, 10, 80, 30)
        analyzer.analyze([ev_in], self.mock_tracker, 100, 100)
        assert 2 in analyzer.states
