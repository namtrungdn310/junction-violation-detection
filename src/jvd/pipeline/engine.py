"""
engine.py — The End-to-End Orchestrator for Junction Violation Detection.

Layer: pipeline/

Coordinates detection, tracking, spatial analysis, multiprocessing OCR,
and evidence reporting in a single runtime loop.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
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
from jvd.utils.stabilizer import VideoStabilizer

logger = logging.getLogger(__name__)


class PipelineEngine:
    """Assembles all modules into a fully functional video processing pipeline."""

    def __init__(
        self,
        video_source: str,
        yolo_model: str,
        roi_points: List[Tuple[float, float]],
        device_flag: str = "gpu",
        vram_limit_gb: float = 2.8,
        enable_emergency: bool = False,
        export_dir: str = "data/exports",
        display: bool = True,
    ) -> None:
        self.video_source = video_source
        self.display = display
        self.export_dir = export_dir
        self.enable_emergency = enable_emergency

        # 1. Device Setup (Enforce GPU VRAM limit)
        self.dm = DeviceManager()
        self.dm.initialize(device_flag=device_flag, vram_limit_gb=vram_limit_gb)

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
        self.emergency = EmergencyVehicleDetector() if self.enable_emergency else None
        self.osd = OSDRenderer()
        self.reporter: ViolationReporter | None = None
        
        # 5. Stabilization
        self.stabilizer = VideoStabilizer(smooth_window=30, crop_pct=0.02)

    def run(self) -> None:
        """Start the video processing loop."""
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {self.video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.enable_emergency and self.emergency is not None:
            self.emergency.fps = fps
        
        video_name = Path(self.video_source).stem
        self.reporter = ViolationReporter(
            export_dir=self.export_dir, 
            fps=int(fps),
            video_name=video_name
        )
        
        self.lpr.start()
        self.stabilizer.start()

        if self.display:
            gui_line = next(
                (line for line in cv2.getBuildInformation().splitlines() if "GUI:" in line),
                "GUI: UNKNOWN",
            )
            if "NONE" in gui_line.upper():
                logger.error(
                    "cv2.imshow is unavailable because OpenCV was built without HighGUI. "
                    "Reinstall GUI build with: uv pip install --python .venv\\Scripts\\python.exe --force-reinstall opencv-contrib-python==4.10.0.84"
                )
                return

        frame_id = 0
        import time
        t_start = time.time()
        current_fps = 0.0

        # Statistics for Summary Report
        unique_vehicles_in_roi: Set[int] = set()

        try:
            logger.info(f"Starting pipeline on {self.video_source} ({width}x{height} @ {fps}fps)")

            while True:
                ret, raw_frame = cap.read()
                if not ret:
                    break

                frame_id += 1
                timestamp = frame_id / fps

                # Calculate FPS every 10 frames
                if frame_id % 10 == 0:
                    t_now = time.time()
                    current_fps = 10 / (t_now - t_start)
                    t_start = t_now

                # 0. Frame Stabilization
                self.stabilizer.put_frame(raw_frame)
                _, matrix = self.stabilizer.get_latest(timeout=0.01)
                
                # Use RAW frame for all downstream AI tasks to avoid stabilization artifacts
                frame = raw_frame

                # A. Core Tracking Loop
                events = self.tracker_session.track(frame, frame_id, timestamp)
                
                # Apply smoothing and update history
                events = self.tracker_mgr.update(events, frame_id)

                # B. Emergency Override
                emerg_ids = self.emergency.get_emergency_ids(frame, events) if self.emergency else set()

                # C. Spatial Violation Reasoning
                violations = self.analyzer.analyze(
                    events=events,
                    tracker=self.tracker_mgr,
                    matrix=matrix,
                    frame_width=width,
                    frame_height=height,
                    emergency_ids=emerg_ids
                )

                # Track unique vehicles seen in ROI for report (Strict Spatial Check)
                for ev in events:
                    tid = ev.track_id
                    if tid is not None:
                        # Only count if the vehicle is spatially inside the ROI right now
                        px, py = ev.bbox.center[0], ev.bbox.y2
                        if matrix is not None:
                            pt = np.array([px, py, 1.0], dtype=np.float32)
                            t_pt = matrix @ pt
                            px, py = t_pt[0], t_pt[1]
                        
                        if self.analyzer.roi.contains((px, py), width, height):
                            # Filter out tracking noise: require at least 5 frames of history (more sensitive to quick entries)
                            if len(self.tracker_mgr.get_history(tid)) >= 5:
                                unique_vehicles_in_roi.add(tid)

                # D. OCR & Evidence Triggers
                for v in violations:
                    tid = v.event.track_id
                    if tid is None:
                        continue
                    
                    # Crop vehicle for OCR (acts as LP crop heuristic here)
                    bbox = v.event.bbox
                    x1, y1, x2, y2 = map(int, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
                    crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                    
                    self.lpr.enqueue(int(tid), crop)
                    
                    # The OCR text might take a few frames to arrive asynchronously
                    lp_text = self.ocr_results.get(int(tid), "PENDING")
                    
                    self.reporter.trigger_violation(
                        track_id=int(tid),
                        timestamp=timestamp,
                        lp_text=lp_text,
                        wide_shot=frame,
                        lp_crop=crop
                    )
                    
                # E. Rendering (Pass FPS and frame counter)
                osd_frame = self.osd.draw(
                    frame=frame, 
                    events=events, 
                    roi=self.analyzer.roi, 
                    states=self.analyzer.states, 
                    emergency_ids=emerg_ids, 
                    ocr_results=self.ocr_results,
                    matrix=matrix,
                    fps=current_fps,
                    frame_id=frame_id,
                    total_frames=total_frames
                )

                # F. Buffer Evidence
                self.reporter.add_frame(osd_frame, self.ocr_results)

                # G. Housekeeping
                if frame_id % 60 == 0:
                    self.tracker_mgr.cleanup_stale(frame_id)

                # H. User Interface
                if self.display:
                    # Smart resize: maintain aspect ratio, target height 800
                    h_orig, w_orig = osd_frame.shape[:2]
                    display_h = 800
                    scale = h_orig / display_h
                    display_w = int(w_orig / scale)
                    
                    disp = cv2.resize(osd_frame, (display_w, display_h))
                    
                    cv2.namedWindow("JVD Surveillance", cv2.WINDOW_NORMAL)
                    cv2.imshow("JVD Surveillance", disp)
                    
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit signal received.")
                        break
                    elif key == ord('r'):
                        logger.info("Re-selecting ROI...")
                        from jvd.utils.roi_helper import select_roi_points
                        new_pts = select_roi_points(self.video_source, current_frame=raw_frame)
                        if new_pts:
                            if matrix is not None:
                                h_img, w_img = raw_frame.shape[:2]
                                transformed_pts = []
                                for nx, ny in new_pts:
                                    # Normalized -> Pixels
                                    px, py = nx * w_img, ny * h_img
                                    # Transform: p' = M * [x, y, 1]^T
                                    pt = np.array([px, py, 1.0], dtype=np.float32)
                                    t_pt = matrix @ pt
                                    # Pixels -> Normalized
                                    transformed_pts.append((t_pt[0] / w_img, t_pt[1] / h_img))
                                new_pts = transformed_pts
                            
                            self.analyzer.roi.update_points(new_pts)
                            logger.info("ROI updated and saved successfully.")

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        except Exception as e:
            logger.exception(f"Pipeline crashed: {e}")
        finally:
            logger.info("Cleaning up resources...")
            cap.release()
            cv2.destroyAllWindows()
            self.lpr.stop()
            self.stabilizer.stop()
            
            # --- Final Summary Report ---
            print("\n" + "="*50)
            print("         TRAFFIC VIOLATION SUMMARY REPORT")
            print("="*50)
            print(f" Video Source: {self.video_source}")
            print(f" Total Unique Vehicles in ROI: {len(unique_vehicles_in_roi)}")
            
            # Identify vehicles that triggered violation at any point
            violating_ids = [
                tid for tid, s in self.analyzer.states.items() if s.violation_triggered
            ]
            
            print(f" Total Violations Detected:    {len(violating_ids)}")
            print("-" * 50)
            if violating_ids:
                print(f" {'ID':<10} | {'License Plate':<20}")
                print("-" * 50)
                for tid in violating_ids:
                    plate = self.ocr_results.get(tid, "NOT_DETECTED")
                    print(f" {tid:<10} | {plate:<20}")
            else:
                print(" No violations detected in this session.")
            print("="*50 + "\n")
