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
_STOP_BUFFER_FRAMES   = 15       # frames to keep "stopped" status during tracking jitter


class RegionOfInterest:
    """Resolution-independent polygon manager for the yellow box ROI."""

    def __init__(self, normalized_points: List[Tuple[float, float]]) -> None:
        if len(normalized_points) < 3:
            raise ValueError("ROI requires at least 3 points to form a polygon.")
        self._norm_pts = normalized_points
        self._scaled_pts: np.ndarray | None = None
        self._cached_dim: Tuple[int, int] | None = None

    def update_points(self, normalized_points: List[Tuple[float, float]]) -> None:
        """Update ROI points and clear caches."""
        if len(normalized_points) < 3:
            return
        self._norm_pts = normalized_points
        self._cached_dim = None
        self._scaled_pts = None

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
    is_inside: bool = False  # NEW: Spatial membership flag
    stop_buffer: int = 0
    last_seen_frame: int = 0


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
        matrix: np.ndarray | None = None,
    ) -> List[ViolationRecord]:
        """
        Process current frame events to find junction violations.

        Args:
            events: List of tracked DetectionEvent for the current frame.
            tracker: The VehicleTrackerManager containing velocity history.
            frame_width: Pixel width of the frame.
            frame_height: Pixel height of the frame.
            emergency_ids: Set of track_ids permanently exempted.
            matrix: 2x3 transformation matrix (Current -> Anchor).
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
            px, py = ev.bbox.center[0], ev.bbox.y2
            
            # If camera moved, transform the point back to anchor coordinate system
            if matrix is not None:
                # Point transformation: p' = M * [x, y, 1]^T
                pt = np.array([px, py, 1.0], dtype=np.float32)
                transformed_pt = matrix @ pt
                px, py = transformed_pt[0], transformed_pt[1]

            if ev.track_id not in self.states:
                self.states[ev.track_id] = TrackState()
            
            state = self.states[ev.track_id]
            state.last_seen_frame = ev.frame_id

            # Ignore vehicles outside the yellow box (using anchor-aligned coords)
            # Use a margin to be more forgiving for vehicles on the edge or entering from side
            # If already in violation, use a larger margin (30px) to prevent flicker at edges
            margin = 30 if state.violation_triggered else 10
            is_inside = self.roi.contains((px, py), frame_width, frame_height)
            
            if not is_inside:
                # Check with margin (simple box approximation for speed)
                poly = self.roi.get_polygon(frame_width, frame_height)
                dist = cv2.pointPolygonTest(poly, (px, py), measureDist=True)
                if dist >= -margin: # Negative distance means outside
                    is_inside = True

            state.is_inside = is_inside
            
            if not is_inside:
                # If they truly leave, we can keep the state for a bit but reset timers
                state.first_stop_time = None
                state.is_blocked = False
                continue

            velocity = tracker.get_velocity(ev.track_id)
            if self._is_stopped(velocity):
                state.stop_buffer = _STOP_BUFFER_FRAMES
                current_stopped_events.append(ev)
                if state.first_stop_time is None:
                    state.first_stop_time = ev.timestamp
            else:
                # Use buffer to smooth out temporary movement or tracking jitter
                if state.stop_buffer > 0:
                    state.stop_buffer -= 1
                    current_stopped_events.append(ev)
                else:
                    state.first_stop_time = None
                    state.is_blocked = False

        # 2. Congestion Reasoning (Forward Collision Check)
        for ev in current_stopped_events:
            tid = ev.track_id
            state = self.states[tid]
            is_blocked = False

            for other_ev in current_stopped_events:
                if other_ev.track_id == tid:
                    continue

                if other_ev.bbox.y2 <= ev.bbox.y1:
                    dist = ev.bbox.y1 - other_ev.bbox.y2
                    if dist < _BLOCK_DIST_THRESH:
                        hiou = self._horizontal_iou(ev.bbox, other_ev.bbox)
                        if hiou > _BLOCK_HIOU_THRESH:
                            is_blocked = True
                            break
            
            state.is_blocked = is_blocked

            # 3. State Machine & Violation Generation
            if state.violation_triggered:
                pass
            elif not is_blocked and state.first_stop_time is not None:
                dwell_time = ev.timestamp - state.first_stop_time
                if dwell_time >= _VIOLATION_TIME_SEC:
                    state.violation_triggered = True
                    violations.append(ViolationRecord(
                        event=ev,
                        dwell_time_seconds=dwell_time,
                        violation_type="yellow_box_stop"
                    ))
                    logger.info(
                        f"Violation confirmed! Track ID: {tid}, Dwell: {dwell_time:.1f}s"
                    )

        # 4. Clean up stale tracked vehicles from memory (Defer deletion)
        # We wait _MAX_STALE_FRAMES before deleting to handle detection hiccups
        current_frame = events[0].frame_id if events else 0
        stale_keys = [
            tid for tid, s in self.states.items() 
            if (current_frame - s.last_seen_frame) > 120 # 4 seconds grace period
        ]
        for tid in stale_keys:
            del self.states[tid]

        return violations
