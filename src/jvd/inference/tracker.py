"""
tracker.py — ByteTrack integration and trajectory history management.

Layer: inference/  (imports from core/ only)

Provides ``ByteTrackSession`` (stateful YOLO26 + ByteTrack tracking) and
``VehicleTrackerManager`` (trajectory history, velocity, and stale ID eviction).
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Dict, Deque, List, Optional, Set, Tuple

from jvd.core.datamodels import BoundingBox, DetectionEvent, VehicleClass

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_COCO_VEHICLE_IDS  = [2, 3, 5, 7]   # car, motorcycle, bus, truck
_DEFAULT_CONF      = 0.4
_DEFAULT_IMGSZ     = 640
_HISTORY_LEN       = 30             # centre points retained per track
_MAX_STALE_FRAMES  = 60             # frames before ID eviction (= track_buffer)

# Resolve config path relative to this file's package root
_BYTETRACK_CFG = Path(__file__).parent.parent.parent.parent / "configs" / "custom_bytetrack.yaml"


# ── ByteTrack session ─────────────────────────────────────────────────────────

class ByteTrackSession:
    """Wraps an ``ObjectDetector`` to run YOLO26 + ByteTrack frame-by-frame.
    
    ``persist=True`` instructs ultralytics to keep a stateful ``BYTETracker``
    instance across calls so the Kalman filter accumulates velocity estimates.
    """

    def __init__(
        self,
        detector,                        # ObjectDetector (avoids circular import)
        tracker_cfg: Path = _BYTETRACK_CFG,
    ) -> None:
        """
        Args:
            detector:    A loaded ``ObjectDetector`` instance.
            tracker_cfg: Path to the ByteTrack YAML override config.
        """
        from jvd.core.exceptions import PipelineConfigError
        if not detector.is_ready:
            raise PipelineConfigError(
                "ByteTrackSession requires a loaded detector. "
                "Call detector.load() before constructing ByteTrackSession."
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
    ) -> List[DetectionEvent]:
        """Run YOLO26 + ByteTrack on one BGR frame with stateful persistence.

        Args:
            frame: Stabilized BGR numpy frame (H, W, 3).
            frame_id: Monotonically increasing frame index.
            timestamp: Wall-clock seconds since stream start.

        Returns:
            List of ``DetectionEvent`` objects with ``track_id`` populated.
        """
        model = self._detector.raw_model
        results = model.track(
            source=frame,
            persist=True,                        # stateful Kalman across frames
            tracker=str(self._cfg),              # custom ByteTrack parameters
            conf=self._detector.conf,
            classes=_COCO_VEHICLE_IDS,
            imgsz=self._detector.imgsz,
            verbose=False,
            device=self._detector.device,
        )
        self._frame_count += 1
        return self._parse(results, frame_id, timestamp)

    # ── Internal parser ───────────────────────────────────────────────────────

    def _parse(
        self,
        results: list,
        frame_id: int,
        timestamp: float,
    ) -> List[DetectionEvent]:
        """Extract ``DetectionEvent`` list from raw YOLO tracker results.

        ByteTrack adds a ``boxes.id`` tensor to results when ``persist=True``.
        If ``boxes.id`` is None (first frame or no match), ``track_id`` is set
        to None and the caller may discard or buffer the event.
        """
        events: List[DetectionEvent] = []
        if not results or results[0].boxes is None:
            return events

        boxes = results[0].boxes
        ids   = boxes.id   # None until ByteTrack associates a track

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
                logger.warning(f"Skipping malformed tracker box [{i}]: {exc}")

        return events


# ── Trajectory manager ────────────────────────────────────────────────────────

class VehicleTrackerManager:
    """Manages per-vehicle spatial trajectory histories for motion analysis.

    Maintains a rolling deque of ``(cx, cy)`` centre coordinates over the last
    ``history_len`` frames per track_id. Stale IDs are automatically evicted
    when not updated for ``max_stale_frames`` consecutive frames.
    """

    def __init__(
        self,
        history_len:      int = _HISTORY_LEN,
        max_stale_frames: int = _MAX_STALE_FRAMES,
    ) -> None:
        self._history_len      = history_len
        self._max_stale_frames = max_stale_frames
        # track_id → deque of (cx, cy) centre coordinates
        self._histories: Dict[int, Deque[Tuple[float, float]]] = {}
        # track_id → last frame_id it was seen on
        self._last_seen: Dict[int, int] = {}

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, events: List[DetectionEvent], frame_id: int) -> None:
        """Ingest a list of tracking events and append centres to histories.

        Args:
            events:   ``DetectionEvent`` list from ``ByteTrackSession.track()``.
            frame_id: Current frame index (used for stale detection).
        """
        for ev in events:
            if ev.track_id is None:
                continue
            tid = ev.track_id
            if tid not in self._histories:
                self._histories[tid] = deque(maxlen=self._history_len)
            cx, cy = ev.bbox.center
            self._histories[tid].append((cx, cy))
            self._last_seen[tid] = frame_id

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_history(self, track_id: int) -> List[Tuple[float, float]]:
        """Return the ordered list of (cx, cy) centres for a track.

        The list is ordered oldest-to-newest and has at most
        ``history_len`` elements.  Returns an empty list for unknown IDs.
        """
        return list(self._histories.get(track_id, []))

    def get_velocity(self, track_id: int) -> Optional[Tuple[float, float]]:
        """Estimate instantaneous velocity (vx, vy) in pixels/frame.

        Returns:
            (vx, vy) tuple, or None if fewer than 2 history points exist.
        """
        hist = self._histories.get(track_id)
        if hist is None or len(hist) < 2:
            return None
        pts = list(hist)
        vx = pts[-1][0] - pts[-2][0]
        vy = pts[-1][1] - pts[-2][1]
        return (vx, vy)

    # ── Maintenance ───────────────────────────────────────────────────────────

    def cleanup_stale(self, current_frame_id: int) -> int:
        """Remove track IDs not updated within ``max_stale_frames``.

        Args:
            current_frame_id: The current frame index.

        Returns:
            Number of track IDs evicted.
        """
        stale = [
            tid for tid, last in self._last_seen.items()
            if (current_frame_id - last) > self._max_stale_frames
        ]
        for tid in stale:
            self._histories.pop(tid, None)
            self._last_seen.pop(tid, None)
        if stale:
            logger.debug(f"Evicted {len(stale)} stale track IDs: {stale}")
        return len(stale)

    @property
    def active_track_ids(self) -> Set[int]:
        """Set of currently tracked vehicle IDs."""
        return set(self._histories.keys())

    @property
    def track_count(self) -> int:
        """Number of active track histories in memory."""
        return len(self._histories)
