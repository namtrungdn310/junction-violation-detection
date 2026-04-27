"""
osd.py — On-Screen Display (OSD) Renderer.

Layer: pipeline/

Implements the MVC View layer to draw violation bounding boxes,
yellow-box polygons with Alpha Blending, and state labels
(EMERGENCY, BLOCKED_BY_TRAFFIC, VIOLATION).
"""

from __future__ import annotations

from typing import Dict, List, Set

import cv2
import numpy as np

from jvd.core.datamodels import DetectionEvent
from jvd.pipeline.analyzer import RegionOfInterest, TrackState


class OSDRenderer:
    """Renders visual overlays onto the video frame."""

    def __init__(self) -> None:
        # BGR Colors
        self.COLOR_GREEN   = (0, 255, 0)
        self.COLOR_RED     = (0, 0, 255)
        self.COLOR_BLUE    = (255, 0, 0)
        self.COLOR_YELLOW  = (153, 255, 255)  # Light yellow

    def draw(
        self,
        frame: np.ndarray,
        events: List[DetectionEvent],
        roi: RegionOfInterest,
        states: Dict[int, TrackState],
        emergency_ids: Set[int],
        ocr_results: Dict[int, str],
    ) -> np.ndarray:
        """
        Draw all analytical states and ROI onto a copy of the frame.
        """
        output = frame.copy()
        h, w = output.shape[:2]

        # 1. Determine global ROI status
        roi_color = self.COLOR_GREEN
        for ev in events:
            if ev.track_id in states and states[ev.track_id].violation_triggered:
                roi_color = self.COLOR_RED
                break

        # 2. Draw ROI with Alpha Blending
        poly = roi.get_polygon(w, h)
        overlay = output.copy()
        cv2.fillPoly(overlay, [poly], roi_color)
        output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)
        cv2.polylines(output, [poly], True, roi_color, 2)

        # 3. Draw Bounding Boxes and Labels
        for ev in events:
            if ev.track_id is None:
                continue

            tid = ev.track_id
            x1, y1, x2, y2 = map(int, [ev.bbox.x1, ev.bbox.y1, ev.bbox.x2, ev.bbox.y2])
            
            # Default state
            color = self.COLOR_GREEN
            label = f"{ev.class_label.name} {tid}"
            thickness = 2

            # State overrides
            if tid in emergency_ids:
                color = self.COLOR_BLUE
                label = f"EMERGENCY {tid}"
            elif tid in states:
                state = states[tid]
                if state.violation_triggered:
                    color = self.COLOR_RED
                    thickness = 3
                    lp = ocr_results.get(tid, "PENDING")
                    label = f"<VIOLATION> {lp}"
                elif state.is_blocked:
                    color = self.COLOR_YELLOW
                    label = f"BLOCKED {tid}"

            # Render
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                output, label, (x1, max(10, y1 - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        return output
