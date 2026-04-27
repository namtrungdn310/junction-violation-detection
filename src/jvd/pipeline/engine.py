"""
engine.py — The End-to-End Orchestrator for Junction Violation Detection.

Layer: pipeline/

Coordinates detection, tracking, spatial analysis, multiprocessing OCR,
and evidence reporting in a single runtime loop.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import List, Tuple

import cv2
import numpy as np

from jvd.core.device import DeviceManager
from jvd.inference.detector import ObjectDetector
from jvd.inference.ocr import LicensePlateRecognizer
from jvd.inference.tracker import ByteTrackSession, VehicleTrackerManager
from jvd.pipeline.analyzer import ViolationAnalyzer
from jvd.pipeline.emergency import EmergencyVehicleDetector
from jvd.pipeline.osd import OSDRenderer
from jvd.pipeline.reporter import ViolationReporter

logger = logging.getLogger(__name__)


class PipelineEngine:
    """Assembles all modules into a fully functional video processing pipeline."""

    def __init__(
        self,
        video_source: str,
        yolo_model: str,
        roi_points: List[Tuple[float, float]],
        export_dir: str = "data/exports",
        display: bool = True,
    ) -> None:
        self.video_source = video_source
        self.display = display
        self.export_dir = export_dir

        # 1. Device Setup (Enforce GPU VRAM limit)
        self.dm = DeviceManager()
        self.dm.initialize(use_gpu=True, max_vram_fraction=0.9)

        # 2. Inference & Tracking
        self.detector = ObjectDetector(yolo_model, device=self.dm.device)
        self.detector.load()
        self.tracker_session = ByteTrackSession(self.detector)
        self.tracker_mgr = VehicleTrackerManager()

        # 3. OCR (Multiprocessed)
        self.manager = mp.Manager()
        self.ocr_results = self.manager.dict()
        self.lpr = LicensePlateRecognizer(self.ocr_results)

        # 4. Pipeline Logic
        self.analyzer = ViolationAnalyzer(roi_points)
        self.emergency = EmergencyVehicleDetector()
        self.osd = OSDRenderer()
        self.reporter: ViolationReporter | None = None

    def run(self) -> None:
        """Start the video processing loop."""
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {self.video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.emergency.fps = fps
        self.reporter = ViolationReporter(export_dir=self.export_dir, fps=int(fps))
        self.lpr.start()

        frame_id = 0
        logger.info(f"Starting pipeline on {self.video_source} ({width}x{height} @ {fps}fps)")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id += 1
                timestamp = frame_id / fps

                # A. Core Tracking Loop
                events = self.tracker_session.track(frame, frame_id, timestamp)
                self.tracker_mgr.update(events, frame_id)

                # B. Emergency Override
                emerg_ids = self.emergency.process(frame, events)

                # C. Violation Reasoning
                violations = self.analyzer.analyze(
                    events, self.tracker_mgr, width, height, emergency_ids=emerg_ids
                )

                # D. OCR & Evidence Triggers
                for v in violations:
                    tid = v.event.track_id
                    if tid is None:
                        continue
                    
                    # Crop vehicle for OCR (acts as LP crop heuristic here)
                    bbox = v.event.bbox
                    x1, y1, x2, y2 = map(int, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
                    crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                    
                    self.lpr.enqueue(tid, crop)
                    
                    # The OCR text might take a few frames to arrive asynchronously
                    lp_text = self.ocr_results.get(tid, "PENDING")
                    
                    self.reporter.trigger_violation(
                        track_id=tid,
                        timestamp=timestamp,
                        lp_text=lp_text,
                        wide_shot=frame,
                        lp_crop=crop
                    )

                # E. Render OSD
                osd_frame = self.osd.draw(
                    frame, events, self.analyzer.roi, 
                    self.analyzer.states, emerg_ids, self.ocr_results
                )

                # F. Buffer Evidence
                self.reporter.add_frame(osd_frame)

                # G. Housekeeping
                if frame_id % 60 == 0:
                    self.tracker_mgr.cleanup_stale(frame_id)

                # H. User Interface
                if self.display:
                    # Resize for display if too large
                    disp = cv2.resize(osd_frame, (1280, 720)) if width > 1280 else osd_frame
                    cv2.imshow("JVD Surveillance", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quit signal received.")
                        break

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        except Exception as e:
            logger.exception(f"Pipeline crashed: {e}")
        finally:
            logger.info("Cleaning up resources...")
            self.lpr.stop()
            cap.release()
            if self.display:
                cv2.destroyAllWindows()
