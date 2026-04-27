"""
emergency.py — Optical heuristic detection for emergency vehicles.

Layer: pipeline/

Uses CPU-based computer vision (HSV thresholding, CLAHE, and peak detection)
to analyze the top 25% of vehicle bounding boxes for alternating Red/Blue
flashing beacon lights at standard frequencies (1Hz - 4Hz).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List, Set, Tuple

import cv2
import numpy as np

from jvd.core.datamodels import DetectionEvent

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_HISTORY_LEN = 60
_MIN_HZ      = 1.0
_MAX_HZ      = 4.0
_NOISE_FLOOR = 10     # Minimum pixel count to be considered a valid flash peak

# HSV Color Ranges
_RED_LOWER1  = np.array([0, 120, 70], dtype=np.uint8)
_RED_UPPER1  = np.array([10, 255, 255], dtype=np.uint8)
_RED_LOWER2  = np.array([170, 120, 70], dtype=np.uint8)
_RED_UPPER2  = np.array([180, 255, 255], dtype=np.uint8)

_BLUE_LOWER  = np.array([100, 150, 0], dtype=np.uint8)
_BLUE_UPPER  = np.array([140, 255, 255], dtype=np.uint8)


class EmergencyVehicleDetector:
    """
    Stateless-like detector that buffers optical signals to identify emergency
    vehicles based on beacon flash frequencies.
    """

    def __init__(self, fps: float = 30.0, history_len: int = _HISTORY_LEN) -> None:
        self.fps = fps
        self.history_len = history_len
        
        # Track active pixel counts over time: track_id -> (red_deque, blue_deque)
        self._signals: Dict[int, Tuple[Deque[int], Deque[int]]] = {}
        
        # Permanently store IDs of confirmed emergency vehicles
        self.confirmed_emergencies: Set[int] = set()
        
        # CLAHE instance for enhancing the Value (V) channel
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _extract_beacon_roi(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        """Extract the top 25% of the bounding box."""
        h_frame, w_frame = frame.shape[:2]
        
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w_frame, int(bbox.x2))
        y2 = min(h_frame, int(bbox.y2))
        
        box_h = y2 - y1
        if box_h <= 4 or (x2 - x1) <= 4:
            return None
            
        # Top 25%
        roi_y2 = y1 + max(1, int(box_h * 0.25))
        return frame[y1:roi_y2, x1:x2]

    def _count_active_pixels(self, roi: np.ndarray) -> Tuple[int, int]:
        """Convert to HSV, apply CLAHE, and threshold for Red/Blue pixels."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Apply CLAHE on the V channel
        h, s, v = cv2.split(hsv)
        v_eq = self._clahe.apply(v)
        hsv_eq = cv2.merge([h, s, v_eq])
        
        # Red masks
        mask_red1 = cv2.inRange(hsv_eq, _RED_LOWER1, _RED_UPPER1)
        mask_red2 = cv2.inRange(hsv_eq, _RED_LOWER2, _RED_UPPER2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Blue mask
        mask_blue = cv2.inRange(hsv_eq, _BLUE_LOWER, _BLUE_UPPER)
        
        red_count = cv2.countNonZero(mask_red)
        blue_count = cv2.countNonZero(mask_blue)
        return red_count, blue_count

    def _detect_frequency(self, signal: Deque[int]) -> float:
        """
        Peak Detection algorithm for a square wave signal.
        Finds local maxima crossing the mean to calculate flash frequency (Hz).
        """
        if len(signal) < self.fps: # Require at least ~1 sec of data
            return 0.0
            
        arr = np.array(signal)
        if arr.max() < _NOISE_FLOOR:
            return 0.0
            
        mean_val = np.mean(arr)
        peaks = 0
        
        # Detect upward zero-crossings around the mean
        for i in range(1, len(arr)):
            if arr[i-1] <= mean_val and arr[i] > mean_val:
                # Validate it's a prominent peak
                if arr[i] > _NOISE_FLOOR:
                    peaks += 1
                    
        window_seconds = len(arr) / self.fps
        return peaks / window_seconds

    def process(self, frame: np.ndarray, events: List[DetectionEvent]) -> Set[int]:
        """
        Analyze current frame events and update the optical signal buffers.
        Returns the set of active track_ids confirmed as emergency vehicles.
        """
        active_ids = set()
        
        for ev in events:
            if ev.track_id is None:
                continue
                
            tid = ev.track_id
            active_ids.add(tid)
            
            # Skip heavy processing if already confirmed
            if tid in self.confirmed_emergencies:
                continue
                
            roi = self._extract_beacon_roi(frame, ev.bbox)
            if roi is None:
                continue
                
            red_px, blue_px = self._count_active_pixels(roi)
            
            if tid not in self._signals:
                self._signals[tid] = (
                    deque(maxlen=self.history_len),
                    deque(maxlen=self.history_len)
                )
                
            red_deq, blue_deq = self._signals[tid]
            red_deq.append(red_px)
            blue_deq.append(blue_px)
            
            # Analyze frequencies
            red_hz = self._detect_frequency(red_deq)
            blue_hz = self._detect_frequency(blue_deq)
            
            # If either color flashes at the standard emergency frequencies
            if (_MIN_HZ <= red_hz <= _MAX_HZ) or (_MIN_HZ <= blue_hz <= _MAX_HZ):
                self.confirmed_emergencies.add(tid)
                logger.info(
                    f"Emergency vehicle detected! Track ID: {tid} "
                    f"(Red Hz: {red_hz:.1f}, Blue Hz: {blue_hz:.1f})"
                )
                # Cleanup memory as we don't need to analyze this ID anymore
                del self._signals[tid]

        # Cleanup stale signals for disappeared tracks
        stale_ids = [tid for tid in self._signals if tid not in active_ids]
        for tid in stale_ids:
            del self._signals[tid]
            
        return self.confirmed_emergencies
