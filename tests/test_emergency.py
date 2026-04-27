"""
Unit tests for Phase 6: EmergencyVehicleDetector.
"""

from __future__ import annotations

import numpy as np

from jvd.core.datamodels import BoundingBox, DetectionEvent, VehicleClass
from jvd.pipeline.emergency import EmergencyVehicleDetector


def _make_event(tid: int, x1: float, y1: float, x2: float, y2: float) -> DetectionEvent:
    return DetectionEvent(
        frame_id=1,
        timestamp=0.0,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        track_id=tid,
        class_label=VehicleClass.CAR
    )

def _make_frame(color_bgr: tuple, h: int = 100, w: int = 100) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color_bgr
    return frame

class TestEmergencyVehicleDetector:
    def test_extract_beacon_roi(self):
        detector = EmergencyVehicleDetector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Valid bbox
        ev = _make_event(1, 10, 10, 50, 90) # Height 80, Width 40
        roi = detector._extract_beacon_roi(frame, ev.bbox)
        assert roi is not None
        assert roi.shape == (20, 40, 3) # Top 25% of 80 is 20
        
        # Too small bbox
        ev_small = _make_event(2, 10, 10, 12, 12)
        roi_small = detector._extract_beacon_roi(frame, ev_small.bbox)
        assert roi_small is None

    def test_count_active_pixels_red(self):
        detector = EmergencyVehicleDetector()
        # Create a red frame (BGR)
        red_roi = _make_frame((0, 0, 255), 20, 40)
        red_count, blue_count = detector._count_active_pixels(red_roi)
        assert red_count > 0
        assert blue_count == 0

    def test_count_active_pixels_blue(self):
        detector = EmergencyVehicleDetector()
        # Create a blue frame (BGR)
        blue_roi = _make_frame((255, 0, 0), 20, 40)
        red_count, blue_count = detector._count_active_pixels(blue_roi)
        assert blue_count > 0
        assert red_count == 0

    def test_detect_frequency(self):
        detector = EmergencyVehicleDetector(fps=30.0)
        from collections import deque
        
        # Simulate a 2Hz square wave (15 frames low, 15 frames high)
        signal = deque(maxlen=60)
        for i in range(60):
            if (i // 15) % 2 == 0:
                signal.append(0)
            else:
                signal.append(100) # Peak
                
        hz = detector._detect_frequency(signal)
        # Should have 2 peaks in 60 frames (2 seconds) -> 1 Hz
        # Wait: 0-14(L), 15-29(H peak1), 30-44(L), 45-59(H peak2) -> 2 peaks / 2 sec = 1.0 Hz
        assert 0.9 <= hz <= 1.1

    def test_process_identifies_emergency_vehicle(self):
        detector = EmergencyVehicleDetector(fps=30.0)
        ev = _make_event(1, 10, 10, 50, 90)
        
        # Feed 60 frames
        # Frame 0-14: Red, Frame 15-29: Black, Frame 30-44: Red, Frame 45-59: Black
        red_frame = _make_frame((0, 0, 255), 100, 100)
        black_frame = _make_frame((0, 0, 0), 100, 100)
        
        for i in range(60):
            if (i // 15) % 2 == 1:
                emergencies = detector.process(red_frame, [ev])
            else:
                emergencies = detector.process(black_frame, [ev])
                
        # By frame 60, it should have detected the 1Hz signal
        assert 1 in emergencies
        assert 1 in detector.confirmed_emergencies
        
        # Ensure cleanup works
        detector.process(black_frame, []) # track 1 disappeared
        assert 1 not in detector._signals # Memory cleaned
        assert 1 in detector.confirmed_emergencies # Still permanently marked
