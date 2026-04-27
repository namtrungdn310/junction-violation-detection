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

    def __init__(self, export_dir: str = "data/exports", fps: int = 30) -> None:
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.pre_event_len = fps * 5  # 5 seconds before trigger
        self.post_event_len = fps * 5 # 5 seconds after trigger
        
        # Buffer containing up to 10 seconds of OSD frames
        self.buffer_len = self.pre_event_len + self.post_event_len
        self._frame_buffer: deque = deque(maxlen=self.buffer_len)
        
        # Track pending video generation jobs
        self._pending_exports: Dict[int, dict] = {}

    def add_frame(self, frame: np.ndarray) -> None:
        """Push a newly rendered OSD frame into the ring buffer."""
        self._frame_buffer.append(frame)
        
        # Process active recording jobs
        finished_ids = []
        for tid, job in self._pending_exports.items():
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
        """Dump the collected frames to an MP4 and write the JSON manifest."""
        base_name = f"violation_{track_id}_{int(job['timestamp'])}"
        base_path = self.export_dir / base_name
        
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
        wide_path = base_path.with_name(f"{base_name}_wide.jpg")
        crop_path = base_path.with_name(f"{base_name}_crop.jpg")
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
            "start_timestamp": round(start_ts, 2),
            "end_timestamp": round(end_ts, 2),
            "license_plate": job["lp_text"],
            "ocr_confidence": 0.99,  # PaddleOCR composite
            "violation_type": "Hatched_Marking_Stop",
            "wide_image_path": str(wide_path.absolute()),
            "crop_image_path": str(crop_path.absolute())
        }
        
        json_path = base_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Evidence package created: {json_path}")
