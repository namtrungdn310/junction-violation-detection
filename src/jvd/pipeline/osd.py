"""
osd.py — On-Screen Display (OSD) Renderer.

Layer: pipeline/

Implements the MVC View layer to draw violation bounding boxes,
yellow-box polygons with Alpha Blending, and state labels
(EMERGENCY, BLOCKED_BY_TRAFFIC, VIOLATION).
"""

from __future__ import annotations

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
        events: list[DetectionEvent],
        roi: RegionOfInterest,
        states: dict[int, TrackState],
        emergency_ids: set[int],
        ocr_results: dict[int, str],
        matrix: np.ndarray | None = None,
        fps: float = 0.0,
        frame_id: int = 0,
        total_frames: int = 0
    ) -> np.ndarray:
        """
        Draw analytical states and ROI onto a copy of the frame with updated UI.
        """
        output = frame.copy()
        h, w = output.shape[:2]

        # 1. Global OSD (FPS & Frame Counter)
        info_text = f"FPS: {fps:.1f} | Frame: {frame_id}/{total_frames}"
        cv2.putText(
            output, info_text, (20, 40),
            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2
        )

        # 2. Draw ROI (Yellow Outline Only)
        poly = roi.get_polygon(w, h)
        if matrix is not None:
            try:
                M_forward = cv2.invertAffineTransform(matrix)
                poly_reshaped = poly.reshape(-1, 1, 2).astype(np.float32)
                poly_transformed = cv2.transform(poly_reshaped, M_forward)
                poly = poly_transformed.reshape(-1, 2).astype(np.int32)
            except cv2.error:
                pass

        # Pure yellow outline (no fill)
        cv2.polylines(output, [poly], True, (0, 255, 255), 3)

        # Vehicle type abbreviation map
        CLASS_MAP = {
            "motorcycle": "moto",
            "motorbike": "moto",
            "motobike": "moto",
            "car": "car",
            "truck": "truck",
            "bus": "bus",
            "bicycle": "bike"
        }

        # 3. Draw Bounding Boxes and Detailed Labels
        for ev in events:
            tid = ev.track_id
            if tid is None: continue

            x1, y1, x2, y2 = map(int, [ev.bbox.x1, ev.bbox.y1, ev.bbox.x2, ev.bbox.y2])
            vehicle_type = CLASS_MAP.get(ev.class_label.name.lower(), ev.class_label.name)

            # Default: Outside ROI (Royal Blue)
            color = (255, 50, 50) # BGR Royal Blue
            thickness = 2
            label = f"{tid} | {vehicle_type}"

            if tid in states:
                state = states[tid]
                if state.is_inside:
                    # Calculate real-time dwell time (starts when vehicle stops)
                    dwell = 0.0
                    if state.first_stop_time is not None:
                        dwell = ev.timestamp - state.first_stop_time

                    if state.violation_triggered:
                        # Inside ROI & Violating (Red)
                        color = (0, 0, 255) # BGR Red
                        thickness = 3
                        lp = ocr_results.get(tid, "SEARCHING...")
                        label = f"{tid} | {vehicle_type} | {dwell:.1f}s | {lp}"
                    else:
                        # Inside ROI & Normal (Green)
                        color = (0, 255, 0) # BGR Green
                        label = f"{tid} | {vehicle_type} | {dwell:.1f}s"
                else:
                    # Outside ROI
                    pass

            # Render Box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # Render Label with Background for readability (Make text bolder: thickness=2)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(
                output, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            )

        return output
