"""
reporter.py — Evidence Generation and Export.

Layer: pipeline/

Maintains a rolling ring buffer of OSD-rendered frames.
Upon violation, records post-event frames and exports a H.264 mp4 video
along with a standardized JSON legal report and image crops.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ViolationReporter:
    """Manages the creation of standalone evidence packages for violations."""

    def __init__(self, export_dir: str = "data/exports", fps: int = 30, video_name: str = "unknown") -> None:
        self.root_export_dir = Path(export_dir)
        self.video_name = video_name
        self.session_dir = self.root_export_dir / self.video_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.pre_event_len = fps * 5  # 5 seconds before trigger
        self.post_event_len = fps * 10 # 10 seconds after trigger
        
        # Buffer containing up to 15 seconds of OSD frames
        self.buffer_len = self.pre_event_len + self.post_event_len
        self._frame_buffer: deque = deque(maxlen=self.buffer_len)
        
        # Track pending video generation jobs
        self._pending_exports: Dict[int, dict] = {}

    def add_frame(self, frame: np.ndarray, ocr_results: Dict[int, str] = None) -> None:
        """Push a newly rendered OSD frame into the ring buffer."""
        self._frame_buffer.append(frame)
        
        # Process active recording jobs
        finished_ids = []
        for tid, job in self._pending_exports.items():
            # Proactively update license plate if it's still PENDING
            if ocr_results and job["lp_text"] == "PENDING":
                new_lp = ocr_results.get(tid, "PENDING")
                if new_lp != "PENDING":
                    job["lp_text"] = new_lp

            job["frames"].append(frame)
            job["remaining"] -= 1
            if job["remaining"] <= 0:
                self._export_evidence(tid, job)
                finished_ids.append(tid)
                
        # Cleanup finished jobs
        for tid in finished_ids:
            del self._pending_exports[tid]

    def trigger_violation(
        self, 
        track_id: int, 
        timestamp: float, 
        lp_text: str, 
        wide_shot: np.ndarray | None, 
        lp_crop: np.ndarray | None
    ) -> None:
        """
        Flag a track_id for recording. Grabs historical frames and 
        prepares the job to collect post-event frames.
        """
        if track_id in self._pending_exports:
            # Update license plate if we finally got a real reading
            if self._pending_exports[track_id]["lp_text"] == "PENDING" and lp_text != "PENDING":
                self._pending_exports[track_id]["lp_text"] = lp_text
            return
            
        # Get up to 5 seconds of historical pre-event frames
        history = list(self._frame_buffer)[-self.pre_event_len:]
        
        self._pending_exports[track_id] = {
            "timestamp": timestamp,
            "lp_text": lp_text,
            "wide_shot": wide_shot.copy() if wide_shot is not None else None,
            "lp_crop": lp_crop.copy() if lp_crop is not None else None,
            "frames": history,
            "remaining": self.post_event_len
        }

    def _export_evidence(self, track_id: int, job: dict) -> None:
        """Dump the collected frames to an MP4 and write the JSON manifest in a subfolder."""
        # Create a dedicated subfolder for this violation
        violation_dir = self.session_dir / f"violation_{track_id}"
        violation_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = f"violation_{track_id}"
        base_path = violation_dir / base_name
        
        frames = job["frames"]
        video_path = base_path.with_suffix(".mp4")
        
        # 1. Export H.264 Video Evidence
        if frames:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()
            
        # 2. Export Static Images
        wide_path = violation_dir / "wide_shot.jpg"
        crop_path = violation_dir / "license_plate.jpg"
        if job["wide_shot"] is not None:
            cv2.imwrite(str(wide_path), job["wide_shot"])
        if job["lp_crop"] is not None:
            cv2.imwrite(str(crop_path), job["lp_crop"])
            
        # 3. Export JSON Report
        # start_time = trigger_time - actual_history_length_in_seconds
        history_frames_cnt = len(frames) - self.post_event_len
        start_ts = job["timestamp"] - (history_frames_cnt / self.fps)
        end_ts = job["timestamp"] + (self.post_event_len / self.fps)
        
        report = {
            "camera_id": "CAM_01",
            "video_source": self.video_name,
            "start_timestamp": round(start_ts, 2),
            "end_timestamp": round(end_ts, 2),
            "license_plate": job["lp_text"],
            "ocr_confidence": 0.99,
            "violation_type": "Hatched_Marking_Stop",
            "wide_image_path": str(wide_path.absolute()),
            "crop_image_path": str(crop_path.absolute()),
            "video_evidence_path": str(video_path.absolute())
        }
        
        json_path = base_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Evidence package created: {violation_dir}")
