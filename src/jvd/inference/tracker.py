"""
tracker.py — Tích hợp ByteTrack và quản lý lịch sử quỹ đạo.

Layer: inference/

Cung cấp ByteTrackSession (YOLO26 + ByteTrack) và
VehicleTrackerManager (lưu quỹ đạo, vận tốc, xóa ID cũ).
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

from jvd.core.datamodels import BoundingBox, DetectionEvent, VehicleClass

logger = logging.getLogger(__name__)

# ── Hằng số ───────────────────────────────────────────────────────────────────
_COCO_VEHICLE_IDS  = [2, 3, 5, 7]   # xe hơi, xe máy, xe buýt, xe tải
_DEFAULT_CONF      = 0.4
_DEFAULT_IMGSZ     = 640
_HISTORY_LEN       = 30             # số điểm tâm lưu mỗi xe
_MAX_STALE_FRAMES  = 120            # số frame trước khi xóa ID
_SMOOTHING_ALPHA   = 0.3            # Trọng số EWA cho mượt quỹ đạo

# Đường dẫn config ByteTrack
_BYTETRACK_CFG = Path(__file__).parent.parent.parent.parent / "configs" / "custom_bytetrack.yaml"


# ── ByteTrack session ─────────────────────────────────────────────────────────

class ByteTrackSession:
    """Bọc ObjectDetector để chạy YOLO26 + ByteTrack theo từng frame.

    persist=True giữ trạng thái Kalman filter tính vận tốc.
    """

    def __init__(
        self,
        detector,                        # ObjectDetector (tránh circular import)
        tracker_cfg: Path = _BYTETRACK_CFG,
    ) -> None:
        """
        Args:
            detector:    Instance ObjectDetector đã load.
            tracker_cfg: File config YAML của ByteTrack.
        """
        from jvd.core.exceptions import PipelineConfigError
        if not detector.is_ready:
            raise PipelineConfigError(
                "ByteTrackSession cần detector đã load. "
                "Gọi detector.load() trước khi tạo ByteTrackSession."
            )
        self._detector   = detector
        self._cfg        = tracker_cfg
        self._frame_count: int = 0
        logger.info(f"ByteTrackSession ready (cfg={tracker_cfg.name})")

    def track(
        self,
        frame,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> list[DetectionEvent]:
        """Chạy YOLO26 + ByteTrack trên 1 frame BGR.

        Returns:
            Danh sách DetectionEvent kèm track_id.
        """
        model = self._detector.raw_model
        results = model.track(
            source=frame,
            persist=True,                        # giữ trạng thái Kalman
            tracker=str(self._cfg),              # config ByteTrack
            conf=self._detector.conf,
            classes=_COCO_VEHICLE_IDS,
            imgsz=self._detector.imgsz,
            verbose=False,
            device=self._detector.device,
        )
        self._frame_count += 1
        return self._parse(results, frame_id, timestamp)

    # ── Parser nội bộ ────────────────────────────────────────────────────────

    def _parse(
        self,
        results: list,
        frame_id: int,
        timestamp: float,
    ) -> list[DetectionEvent]:
        """Trích xuất DetectionEvent từ kết quả YOLO.

        ByteTrack thêm boxes.id. Nếu None là chưa map được track.
        """
        events: list[DetectionEvent] = []
        if not results or results[0].boxes is None:
            return events

        boxes = results[0].boxes
        ids   = boxes.id   # None cho đến khi ByteTrack map được

        for i in range(len(boxes)):
            try:
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf    = float(boxes.conf[i])
                coco_id = int(boxes.cls[i])
                tid     = int(ids[i]) if ids is not None else None
                events.append(DetectionEvent(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    track_id=tid,
                    class_label=VehicleClass.from_coco_id(coco_id),
                    confidence=conf,
                ))
            except (IndexError, ValueError) as exc:
                logger.warning(f"Bỏ qua box lỗi [{i}]: {exc}")

        return events


# ── Quản lý quỹ đạo ──────────────────────────────────────────────────────────

class VehicleTrackerManager:
    """Quản lý lịch sử tọa độ tâm (cx, cy) để phân tích chuyển động.

    Xóa ID xe nếu không cập nhật quá max_stale_frames.
    """

    def __init__(
        self,
        history_len:      int = _HISTORY_LEN,
        max_stale_frames: int = _MAX_STALE_FRAMES,
    ) -> None:
        self._history_len      = history_len
        self._max_stale_frames = max_stale_frames
        # track_id → deque tâm (cx, cy)
        self._histories: dict[int, deque[tuple[float, float]]] = {}
        # track_id → frame_id cuối cùng nhìn thấy
        self._last_seen: dict[int, int] = {}
        # track_id → BoundingBox đã làm mượt
        self._last_boxes: dict[int, BoundingBox] = {}

    # ── Cập nhật ─────────────────────────────────────────────────────────────

    def update(self, events: list[DetectionEvent], frame_id: int) -> list[DetectionEvent]:
        """Nhận sự kiện, làm mượt Bounding Box và trả về sự kiện mới."""
        from dataclasses import replace
        smoothed_events: list[DetectionEvent] = []

        for ev in events:
            if ev.track_id is None:
                smoothed_events.append(ev)
                continue

            tid = ev.track_id
            current_bbox = ev.bbox

            # 1. Làm mượt Box (EWA)
            if tid in self._last_boxes:
                prev = self._last_boxes[tid]
                alpha = _SMOOTHING_ALPHA

                # Làm mượt từng tọa độ
                smoothed_bbox = BoundingBox(
                    x1 = alpha * current_bbox.x1 + (1 - alpha) * prev.x1,
                    y1 = alpha * current_bbox.y1 + (1 - alpha) * prev.y1,
                    x2 = alpha * current_bbox.x2 + (1 - alpha) * prev.x2,
                    y2 = alpha * current_bbox.y2 + (1 - alpha) * prev.y2
                )
                ev = replace(ev, bbox=smoothed_bbox)

            self._last_boxes[tid] = ev.bbox
            smoothed_events.append(ev)

            # 2. Lưu lịch sử
            if tid not in self._histories:
                self._histories[tid] = deque(maxlen=self._history_len)
            cx, cy = ev.bbox.center
            self._histories[tid].append((cx, cy))
            self._last_seen[tid] = frame_id

        return smoothed_events

    # ── Truy vấn ─────────────────────────────────────────────────────────────

    def get_history(self, track_id: int) -> list[tuple[float, float]]:
        """Lấy danh sách tâm (cx, cy) của 1 xe. Cũ -> mới."""
        return list(self._histories.get(track_id, []))

    def get_velocity(self, track_id: int) -> tuple[float, float] | None:
        """Ước tính vận tốc (vx, vy) pixel/frame."""
        hist = self._histories.get(track_id)
        if hist is None or len(hist) < 2:
            return None
        pts = list(hist)
        vx = pts[-1][0] - pts[-2][0]
        vy = pts[-1][1] - pts[-2][1]
        return (vx, vy)

    # ── Bảo trì ──────────────────────────────────────────────────────────────

    def cleanup_stale(self, current_frame_id: int) -> int:
        """Xóa ID xe không xuất hiện quá max_stale_frames."""
        stale = [
            tid for tid, last in self._last_seen.items()
            if (current_frame_id - last) > self._max_stale_frames
        ]
        for tid in stale:
            self._histories.pop(tid, None)
            self._last_seen.pop(tid, None)
            self._last_boxes.pop(tid, None)
        if stale:
            logger.debug(f"Đã xóa {len(stale)} track ID cũ: {stale}")
        return len(stale)

    @property
    def active_track_ids(self) -> set[int]:
        """Danh sách ID xe đang theo dõi."""
        return set(self._histories.keys())

    @property
    def track_count(self) -> int:
        """Số lượng xe đang lưu quỹ đạo."""
        return len(self._histories)
