"""
analyzer.py — Xử lý logic không gian và phát hiện vi phạm.

Layer: pipeline/

Trung tâm xử lý luật giao thông. Đánh giá vị trí xe và áp dụng luật vạch mắt võng.

Logic cốt lõi:
1. Point-in-Polygon: Kiểm tra xe trong vùng vàng. Dùng điểm giữa-dưới
   của bounding box để tránh lỗi phối cảnh.
2. Phân tích động học: Kiểm tra vận tốc giảm gần 0 -> xe đang dừng.
3. Xử lý ùn tắc: Nếu dừng, kiểm tra có xe dừng phía trước chặn đường không
   (thông qua độ giao thoa ngang - Horizontal IoU). Nếu có -> không vi phạm.
4. State Machine: Chỉ báo vi phạm nếu xe dừng hẳn trong vùng vàng > 3s
   và không bị cản phía trước.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from jvd.core.datamodels import BoundingBox, DetectionEvent, ViolationRecord
from jvd.inference.tracker import VehicleTrackerManager

logger = logging.getLogger(__name__)

# ── Hằng số ───────────────────────────────────────────────────────────────────
_STOP_VELOCITY_THRESH = 2.0      # pixel/frame (ngưỡng dừng)
_VIOLATION_TIME_SEC   = 3.0      # số giây dừng để tính vi phạm
_BLOCK_DIST_THRESH    = 50.0     # pixel dọc tối đa để tính là "bị cản"
_BLOCK_HIOU_THRESH    = 0.3      # IoU ngang tối thiểu để tính là "bị cản"
_STOP_BUFFER_FRAMES   = 15       # số frame giữ trạng thái "đang dừng" khi mất dấu


class RegionOfInterest:
    """Quản lý polygon độc lập độ phân giải cho vùng mắt võng."""

    def __init__(self, normalized_points: list[tuple[float, float]]) -> None:
        if len(normalized_points) < 3:
            raise ValueError("ROI cần ít nhất 3 điểm.")
        self._norm_pts = normalized_points
        self._scaled_pts: np.ndarray | None = None
        self._cached_dim: tuple[int, int] | None = None

    def update_points(self, normalized_points: list[tuple[float, float]]) -> None:
        """Cập nhật điểm ROI và xóa cache."""
        if len(normalized_points) < 3:
            return
        self._norm_pts = normalized_points
        self._cached_dim = None
        self._scaled_pts = None

    def get_polygon(self, width: int, height: int) -> np.ndarray:
        """Nội suy tọa độ chuẩn hóa sang pixel của frame."""
        if self._cached_dim == (width, height) and self._scaled_pts is not None:
            return self._scaled_pts

        pts = [[int(nx * width), int(ny * height)] for nx, ny in self._norm_pts]
        self._scaled_pts = np.array(pts, dtype=np.int32)
        self._cached_dim = (width, height)
        return self._scaled_pts

    def contains(self, pt: tuple[float, float], width: int, height: int) -> bool:
        """Kiểm tra điểm nằm trong đa giác."""
        poly = self.get_polygon(width, height)
        return cv2.pointPolygonTest(poly, pt, measureDist=False) >= 0


@dataclass
class TrackState:
    """Trạng thái của một xe."""
    first_stop_time: float | None = None
    is_blocked: bool = False
    violation_triggered: bool = False
    is_inside: bool = False  # Nằm trong ROI
    stop_buffer: int = 0
    last_seen_frame: int = 0


class ViolationAnalyzer:
    """Đánh giá không gian và va chạm động học."""

    def __init__(self, roi_normalized: list[tuple[float, float]]) -> None:
        self.roi = RegionOfInterest(roi_normalized)
        self.states: dict[int, TrackState] = {}

    def _is_stopped(self, velocity: tuple[float, float] | None) -> bool:
        if velocity is None:
            return False
        vx, vy = velocity
        mag = (vx**2 + vy**2)**0.5
        return mag < _STOP_VELOCITY_THRESH

    def _horizontal_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Tính Intersection over Union (IoU) theo trục X."""
        inter_x1 = max(box1.x1, box2.x1)
        inter_x2 = min(box1.x2, box2.x2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        if inter_w == 0:
            return 0.0
        union_w = max(box1.x2, box2.x2) - min(box1.x1, box2.x1)
        return inter_w / union_w if union_w > 0 else 0.0

    def analyze(
        self,
        events: list[DetectionEvent],
        tracker: VehicleTrackerManager,
        frame_width: int,
        frame_height: int,
        emergency_ids: set[int] | None = None,
        matrix: np.ndarray | None = None,
    ) -> list[ViolationRecord]:
        """Xử lý sự kiện frame hiện tại để tìm vi phạm."""
        violations: list[ViolationRecord] = []
        current_stopped_events: list[DetectionEvent] = []
        active_ids: set[int] = set()
        emergency_ids = emergency_ids or set()

        # 1. Lọc sự kiện & Động học
        for ev in events:
            if ev.track_id is None or ev.track_id in emergency_ids:
                continue

            active_ids.add(ev.track_id)

            # Điểm chạm đất
            px, py = ev.bbox.center[0], ev.bbox.y2

            # Dịch tọa độ nếu camera rung lắc
            if matrix is not None:
                pt = np.array([px, py, 1.0], dtype=np.float32)
                transformed_pt = matrix @ pt
                px, py = transformed_pt[0], transformed_pt[1]

            if ev.track_id not in self.states:
                self.states[ev.track_id] = TrackState()

            state = self.states[ev.track_id]
            state.last_seen_frame = ev.frame_id

            # Biên mỏng (2px) khi mới vào, biên dày (30px) khi đã vi phạm để chống nhiễu
            margin = 30 if state.violation_triggered else 2
            is_inside = self.roi.contains((px, py), frame_width, frame_height)

            if not is_inside:
                # Kiểm tra với biên
                poly = self.roi.get_polygon(frame_width, frame_height)
                dist = cv2.pointPolygonTest(poly, (px, py), measureDist=True)
                if dist >= -margin: 
                    is_inside = True

            state.is_inside = is_inside

            if not is_inside:
                # Vừa rời đi, reset thời gian
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
                # Dùng buffer làm mượt khi xe nhích nhẹ
                if state.stop_buffer > 0:
                    state.stop_buffer -= 1
                    current_stopped_events.append(ev)
                else:
                    state.first_stop_time = None
                    state.is_blocked = False

        # 2. Xử lý ùn tắc (Có xe chắn phía trước)
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

            # 3. Kích hoạt vi phạm
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
                        f"Phát hiện vi phạm! Track ID: {tid}, Dừng: {dwell_time:.1f}s"
                    )

        # 4. Xóa xe cũ khỏi bộ nhớ (giữ lại 1 thời gian để chống nhiễu)
        current_frame = events[0].frame_id if events else 0
        stale_keys = [
            tid for tid, s in self.states.items()
            if (current_frame - s.last_seen_frame) > 120 # 4 giây
        ]
        for tid in stale_keys:
            del self.states[tid]

        return violations
