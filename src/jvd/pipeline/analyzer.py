"""
analyzer.py — Spatial logic and junction violation reasoning.

Layer: pipeline/ (imports from core/ and inference/)

This module acts as the legal reasoning center of the system.
It evaluates vehicle spatial behavior and applies traffic rules
concerning yellow-box junctions.

Core Logic
~~~~~~~~~~
1. Point-in-Polygon (Ray Casting):
   Determines if a vehicle is inside the prohibited yellow box.
   Uses the bottom-center of the bounding box (ground touch point)
   to avoid perspective distortion errors. The polygon is defined
   in normalized coordinates (0.0 - 1.0) for resolution independence.

2. Kinematic Analysis:
   Checks if the vehicle's velocity has dropped near zero,
   indicating a stop on the junction.

3. Congestion Reasoning (Forward Collision Check):
   If stopped, projects a virtual search area "upwards" (smaller y)
   to find vehicles in front. If a vehicle ahead is also stopped
   and has significant horizontal overlap (Horizontal IoU), the current
   vehicle is granted a `BLOCKED_BY_TRAFFIC` exemption.

4. State Machine:
   Triggers a `VIOLATION` only if a vehicle is completely stopped
   in the ROI for > 3 seconds without any exemptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

from jvd.core.datamodels import BoundingBox, DetectionEvent, ViolationRecord
from jvd.inference.tracker import VehicleTrackerManager

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_STOP_VELOCITY_THRESH = 2.0      # pixels/frame
_VIOLATION_TIME_SEC   = 3.0      # seconds
_BLOCK_DIST_THRESH    = 50.0     # max vertical pixels to consider "blocked"
_BLOCK_HIOU_THRESH    = 0.3      # min horizontal IoU to consider "blocked"


class RegionOfInterest:
    """Resolution-independent polygon manager for the yellow box ROI."""

    def __init__(self, normalized_points: List[Tuple[float, float]]) -> None:
        if len(normalized_points) < 3:
            raise ValueError("ROI requires at least 3 points to form a polygon.")
        self._norm_pts = normalized_points
        self._scaled_pts: np.ndarray | None = None
        self._cached_dim: Tuple[int, int] | None = None

    def get_polygon(self, width: int, height: int) -> np.ndarray:
        """Interpolate normalized coordinates to frame pixels."""
        if self._cached_dim == (width, height) and self._scaled_pts is not None:
            return self._scaled_pts

        pts = [[int(nx * width), int(ny * height)] for nx, ny in self._norm_pts]
        self._scaled_pts = np.array(pts, dtype=np.int32)
        self._cached_dim = (width, height)
        return self._scaled_pts

    def contains(self, pt: Tuple[float, float], width: int, height: int) -> bool:
        """Ray-casting point-in-polygon test using cv2."""
        poly = self.get_polygon(width, height)
        # measureDist=False returns +1 inside, 0 on edge, -1 outside
        return cv2.pointPolygonTest(poly, pt, measureDist=False) >= 0


@dataclass
class TrackState:
    """State machine context for a single vehicle."""
    first_stop_time: float | None = None
    is_blocked: bool = False
    violation_triggered: bool = False


class ViolationAnalyzer:
    """Evaluates spatial constraints and kinematic interactions."""

    def __init__(self, roi_normalized: List[Tuple[float, float]]) -> None:
        self.roi = RegionOfInterest(roi_normalized)
        self.states: Dict[int, TrackState] = {}

    def _is_stopped(self, velocity: Tuple[float, float] | None) -> bool:
        if velocity is None:
            return False
        vx, vy = velocity
        mag = (vx**2 + vy**2)**0.5
        return mag < _STOP_VELOCITY_THRESH

    def _horizontal_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Compute 1D intersection over union along the X axis."""
        inter_x1 = max(box1.x1, box2.x1)
        inter_x2 = min(box1.x2, box2.x2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        if inter_w == 0:
            return 0.0
        union_w = max(box1.x2, box2.x2) - min(box1.x1, box2.x1)
        return inter_w / union_w if union_w > 0 else 0.0

    def analyze(
        self,
        events: List[DetectionEvent],
        tracker: VehicleTrackerManager,
        frame_width: int,
        frame_height: int,
        emergency_ids: Set[int] | None = None,
    ) -> List[ViolationRecord]:
        """
        Process current frame events to find junction violations.

        Args:
            events: List of tracked DetectionEvent for the current frame.
            tracker: The VehicleTrackerManager containing velocity history.
            frame_width: Pixel width of the frame.
            frame_height: Pixel height of the frame.
            emergency_ids: Set of track_ids permanently exempted (emergency vehicles).

        Returns:
            List of confirmed ViolationRecords for the current frame.
        """
        violations: List[ViolationRecord] = []
        current_stopped_events: List[DetectionEvent] = []
        active_ids: Set[int] = set()
        emergency_ids = emergency_ids or set()

        # 1. Filter events & Kinematic Analysis
        for ev in events:
            if ev.track_id is None or ev.track_id in emergency_ids:
                continue
            
            active_ids.add(ev.track_id)

            # Ground touch point
            bottom_center = (ev.bbox.center[0], ev.bbox.y2)

            # Ignore vehicles outside the yellow box
            if not self.roi.contains(bottom_center, frame_width, frame_height):
                self.states.pop(ev.track_id, None)
                continue

            velocity = tracker.get_velocity(ev.track_id)
            if self._is_stopped(velocity):
                current_stopped_events.append(ev)
                if ev.track_id not in self.states:
                    self.states[ev.track_id] = TrackState()
                
                state = self.states[ev.track_id]
                if state.first_stop_time is None:
                    state.first_stop_time = ev.timestamp
            else:
                # Moving: reset stop state and exemptions
                if ev.track_id in self.states:
                    state = self.states[ev.track_id]
                    state.first_stop_time = None
                    state.is_blocked = False

        # 2. Congestion Reasoning (Forward Collision Check)
        for ev in current_stopped_events:
            if ev.track_id is None:
                continue
            
            tid = ev.track_id
            state = self.states[tid]
            is_blocked = False

            for other_ev in current_stopped_events:
                if other_ev.track_id == tid:
                    continue

                # Project search area "upwards" in the image (smaller y values)
                # Front vehicle is blocking if its bottom (y2) is just above our top (y1)
                if other_ev.bbox.y2 <= ev.bbox.y1:
                    dist = ev.bbox.y1 - other_ev.bbox.y2
                    if dist < _BLOCK_DIST_THRESH:
                        hiou = self._horizontal_iou(ev.bbox, other_ev.bbox)
                        if hiou > _BLOCK_HIOU_THRESH:
                            is_blocked = True
                            break
            
            state.is_blocked = is_blocked

            # 3. State Machine & Violation Generation
            if not is_blocked and state.first_stop_time is not None:
                dwell_time = ev.timestamp - state.first_stop_time
                if dwell_time >= _VIOLATION_TIME_SEC and not state.violation_triggered:
                    state.violation_triggered = True
                    violations.append(ViolationRecord(
                        event=ev,
                        dwell_time_seconds=dwell_time,
                        violation_type="yellow_box_stop"
                    ))
                    logger.info(
                        f"Violation detected! Track ID: {tid}, Dwell: {dwell_time:.1f}s"
                    )

        # 4. Clean up stale tracked vehicles from memory
        stale_keys = [tid for tid in self.states if tid not in active_ids]
        for tid in stale_keys:
            del self.states[tid]

        return violations
