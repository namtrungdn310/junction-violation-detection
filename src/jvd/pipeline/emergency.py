"""
emergency.py — Phát hiện xe ưu tiên bằng quang học.

Layer: pipeline/

Dùng CV trên CPU (HSV, CLAHE, tìm đỉnh) để phân tích 25% trên cùng 
của bounding box, tìm đèn nháy Xanh/Đỏ tần số chuẩn (1Hz - 4Hz).
"""

from __future__ import annotations

import logging
from collections import deque

import cv2
import numpy as np

from jvd.core.datamodels import DetectionEvent

logger = logging.getLogger(__name__)

# ── Hằng số ───────────────────────────────────────────────────────────────────
_HISTORY_LEN = 60
_MIN_HZ      = 1.0
_MAX_HZ      = 4.0
_NOISE_FLOOR = 10     # Pixel tối thiểu để tính là 1 lần nháy sáng

# Dải màu HSV
_RED_LOWER1  = np.array([0, 120, 70], dtype=np.uint8)
_RED_UPPER1  = np.array([10, 255, 255], dtype=np.uint8)
_RED_LOWER2  = np.array([170, 120, 70], dtype=np.uint8)
_RED_UPPER2  = np.array([180, 255, 255], dtype=np.uint8)

_BLUE_LOWER  = np.array([100, 150, 0], dtype=np.uint8)
_BLUE_UPPER  = np.array([140, 255, 255], dtype=np.uint8)


class EmergencyVehicleDetector:
    """
    Lưu buffer tín hiệu quang học để xác định xe ưu tiên 
    dựa trên tần số nháy đèn.
    """

    def __init__(self, fps: float = 30.0, history_len: int = _HISTORY_LEN) -> None:
        self.fps = fps
        self.history_len = history_len

        # track_id -> (red_deque, blue_deque)
        self._signals: dict[int, tuple[deque[int], deque[int]]] = {}

        # Lưu vĩnh viễn ID các xe ưu tiên đã xác nhận
        self.confirmed_emergencies: set[int] = set()

        # CLAHE để tăng cường độ sáng (kênh V)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _extract_beacon_roi(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        """Cắt lấy 25% trên cùng của hộp giới hạn."""
        h_frame, w_frame = frame.shape[:2]

        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w_frame, int(bbox.x2))
        y2 = min(h_frame, int(bbox.y2))

        box_h = y2 - y1
        if box_h <= 4 or (x2 - x1) <= 4:
            return None

        # 25% trên cùng
        roi_y2 = y1 + max(1, int(box_h * 0.25))
        return frame[y1:roi_y2, x1:x2]

    def _count_active_pixels(self, roi: np.ndarray) -> tuple[int, int]:
        """Chuyển HSV, dùng CLAHE, đếm pixel Đỏ/Xanh."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Áp dụng CLAHE lên kênh V
        h, s, v = cv2.split(hsv)
        v_eq = self._clahe.apply(v)
        hsv_eq = cv2.merge([h, s, v_eq])

        # Mask Đỏ
        mask_red1 = cv2.inRange(hsv_eq, _RED_LOWER1, _RED_UPPER1)
        mask_red2 = cv2.inRange(hsv_eq, _RED_LOWER2, _RED_UPPER2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Mask Xanh
        mask_blue = cv2.inRange(hsv_eq, _BLUE_LOWER, _BLUE_UPPER)

        red_count = cv2.countNonZero(mask_red)
        blue_count = cv2.countNonZero(mask_blue)
        return red_count, blue_count

    def _detect_frequency(self, signal: deque[int]) -> float:
        """
        Thuật toán tìm đỉnh sóng vuông.
        Đếm số lần vượt qua giá trị trung bình để tính tần số (Hz).
        """
        if len(signal) < self.fps: # Cần ít nhất 1 giây dữ liệu
            return 0.0

        arr = np.array(signal)
        if arr.max() < _NOISE_FLOOR:
            return 0.0

        mean_val = np.mean(arr)
        peaks = 0

        # Tìm các điểm cắt lên (vượt qua mean)
        for i in range(1, len(arr)):
            if arr[i-1] <= mean_val and arr[i] > mean_val:
                if arr[i] > _NOISE_FLOOR:
                    peaks += 1

        window_seconds = len(arr) / self.fps
        return peaks / window_seconds

    def process(self, frame: np.ndarray, events: list[DetectionEvent]) -> set[int]:
        """
        Phân tích frame hiện tại và cập nhật tín hiệu.
        Trả về set các track_id là xe ưu tiên.
        """
        active_ids = set()

        for ev in events:
            if ev.track_id is None:
                continue

            tid = ev.track_id
            active_ids.add(tid)

            # Bỏ qua nếu đã xác nhận là xe ưu tiên
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

            # Phân tích tần số
            red_hz = self._detect_frequency(red_deq)
            blue_hz = self._detect_frequency(blue_deq)

            # Nếu Đỏ hoặc Xanh nháy đúng dải tần số
            if (_MIN_HZ <= red_hz <= _MAX_HZ) or (_MIN_HZ <= blue_hz <= _MAX_HZ):
                self.confirmed_emergencies.add(tid)
                logger.info(
                    f"Phát hiện xe ưu tiên! Track ID: {tid} "
                    f"(Đỏ: {red_hz:.1f}Hz, Xanh: {blue_hz:.1f}Hz)"
                )
                # Đã xác nhận xong, xóa dữ liệu để nhẹ RAM
                del self._signals[tid]

        # Xóa dữ liệu các xe đã biến mất
        stale_ids = [tid for tid in self._signals if tid not in active_ids]
        for tid in stale_ids:
            del self._signals[tid]

        return self.confirmed_emergencies
