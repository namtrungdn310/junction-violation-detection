"""
reporter.py — Tạo và xuất bằng chứng vi phạm.

Layer: pipeline/

Duy trì buffer xoay vòng các frame OSD. Khi có vi phạm, lưu frame trước/sau 
sự kiện, xuất video mp4, ảnh cắt và báo cáo JSON chuẩn.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ViolationReporter:
    """Quản lý việc tạo gói bằng chứng độc lập cho các vi phạm."""

    def __init__(self, export_dir: str = "data/exports", fps: int = 30, video_name: str = "unknown") -> None:
        self.root_export_dir = Path(export_dir)
        self.video_name = video_name
        self.session_dir = self.root_export_dir / self.video_name

        # ── Tự động dọn dẹp bằng chứng cũ của video này ─────────────────────────
        if self.session_dir.exists():
            import shutil
            logger.info(f"Đang xóa bằng chứng cũ cho {self.video_name}...")
            shutil.rmtree(self.session_dir)

        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.pre_event_len = fps * 5  # 5 giây trước sự kiện
        self.post_event_len = fps * 10 # 10 giây sau sự kiện

        # Buffer chứa tối đa 15 giây frame OSD
        self.buffer_len = self.pre_event_len + self.post_event_len
        self._frame_buffer: deque = deque(maxlen=self.buffer_len)

        # Theo dõi các tiến trình xuất video đang chờ
        self._pending_exports: dict[int, dict] = {}

    def add_frame(self, frame: np.ndarray, ocr_results: dict[int, str] = None) -> None:
        """Đẩy frame OSD mới render vào buffer xoay vòng."""
        self._frame_buffer.append(frame)

        # Xử lý các tác vụ ghi hình đang chạy
        finished_ids = []
        for tid, job in self._pending_exports.items():
            # Chủ động cập nhật biển số nếu vẫn đang PENDING
            if ocr_results and job["lp_text"] == "PENDING":
                new_lp = ocr_results.get(tid, "PENDING")
                if new_lp != "PENDING":
                    job["lp_text"] = new_lp

            job["frames"].append(frame)
            job["remaining"] -= 1
            if job["remaining"] <= 0:
                self._export_evidence(tid, job)
                finished_ids.append(tid)

        # Dọn dẹp các tác vụ đã xong
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
        Đánh dấu track_id để ghi hình. Lấy lịch sử frame và 
        chuẩn bị thu thập frame sau sự kiện.
        """
        if track_id in self._pending_exports:
            # Cập nhật biển số nếu đã có kết quả thực sự
            if self._pending_exports[track_id]["lp_text"] == "PENDING" and lp_text != "PENDING":
                self._pending_exports[track_id]["lp_text"] = lp_text
            return

        # Lấy tối đa 5 giây frame lịch sử trước sự kiện
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
        """Lưu frame vào file MP4 và viết báo cáo JSON vào thư mục con."""
        # Tạo thư mục con riêng cho vi phạm này
        violation_dir = self.session_dir / f"violation_{track_id}"
        violation_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"violation_{track_id}"
        base_path = violation_dir / base_name

        frames = job["frames"]
        video_path = base_path.with_suffix(".mp4")

        # 1. Xuất video bằng chứng H.264
        if frames:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()

        # 2. Xuất ảnh tĩnh
        wide_path = violation_dir / "wide_shot.jpg"
        crop_path = violation_dir / "license_plate.jpg"
        if job["wide_shot"] is not None:
            cv2.imwrite(str(wide_path), job["wide_shot"])
        if job["lp_crop"] is not None:
            cv2.imwrite(str(crop_path), job["lp_crop"])

        # 3. Xuất báo cáo JSON
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

        logger.info(f"Đã tạo gói bằng chứng: {violation_dir}")
