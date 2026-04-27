"""
stabilizer.py — Multi-threaded CPU-only video stabilization worker.

Layer: utils/  (imports from core/ and utils/transform)

Architecture
~~~~~~~~~~~~
                   ┌─────────────────────────────────────────┐
  Main thread      │  VideoStabilizer                        │
  ──────────────►  │  raw_queue (Queue, cap=60)              │
  push raw frames  │      ↓  worker thread (CPU)             │
                   │  _stabilize_loop() → _FrameProcessor    │
                   │  ├─ extract_keypoints (ORB/SIFT+mask)   │
                   │  ├─ estimate_transform (LK + Affine)    │
                   │  ├─ TrajectoryBuffer.push() (MA-30)     │
                   │  └─ warp_and_crop (BORDER_REPLICATE)    │
                   │      ↓                                  │
                   │  stable_queue (Queue, cap=60)           │
  Main thread ◄──  │  get_stable_frame()                     │
                   └─────────────────────────────────────────┘

Queue pressure control
~~~~~~~~~~~~~~~~~~~~~~
If the *input* queue is full (60 frames), the **oldest** frame is
discarded before the new one is enqueued, preserving real-time throughput.

GPU budget
~~~~~~~~~~
Zero CUDA calls are made in this module.  All cv2 operations run on CPU,
preserving the ≤ 2.8 GB VRAM budget for YOLO and PaddleOCR.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from jvd.utils.transform import (
    Frame,
    TrajectoryBuffer,
    estimate_transform,
    extract_keypoints,
    warp_and_crop,
)

logger = logging.getLogger(__name__)

_QUEUE_MAXSIZE   = 60
_REINIT_INTERVAL = 30     # Re-detect keypoints every N frames
_MIN_STABLE_RATIO = 0.15  # Fallback to raw if < 15 % keypoints tracked


# ── Per-frame stateful processor ─────────────────────────────────────────────

@dataclass
class _FrameProcessor:
    """Stateful per-frame stabilization processor (CPU-only).

    Separates CV2 bookkeeping from thread management so ``VideoStabilizer``
    remains a pure threading concern.
    """
    detector:   str            = "ORB"
    crop_pct:   float          = 0.05
    trajectory: TrajectoryBuffer = field(default_factory=lambda: TrajectoryBuffer(30))
    _prev_gray: Optional[Frame]   = field(default=None, init=False, repr=False)
    _prev_pts:  Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _counter:   int               = field(default=0,    init=False, repr=False)

    def process(
        self,
        frame: Frame,
        boxes: List[Tuple[int, int, int, int]],
    ) -> Frame:
        """One CPU stabilization step: greyscale→keypoints→LK flow→Affine→MA smooth→warp+crop.

        Returns stabilized frame, or raw frame if keypoints collapse below threshold.
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = curr_gray
            self._prev_pts  = extract_keypoints(curr_gray, self.detector, boxes)
            return frame

        needs_reinit = (
            self._counter % _REINIT_INTERVAL == 0
            or not (self._prev_pts is not None and len(self._prev_pts) > 0)
        )
        if needs_reinit:
            self._prev_pts = extract_keypoints(self._prev_gray, self.detector, boxes)

        M_diff = estimate_transform(self._prev_gray, curr_gray, self._prev_pts)

        # Keypoint quality guard
        n_init = len(self._prev_pts) if self._prev_pts is not None else 0
        if n_init > 0:
            _, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, curr_gray, self._prev_pts, None
            )
            n_tracked = int(np.sum(status)) if status is not None else 0
            if n_tracked / n_init < _MIN_STABLE_RATIO:
                logger.debug(f"Keypoint collapse ({n_tracked}/{n_init}), passthrough.")
                self._prev_gray = curr_gray
                self._prev_pts  = extract_keypoints(curr_gray, self.detector, boxes)
                return frame

        M_comp    = self.trajectory.push(M_diff)
        stabilised = warp_and_crop(frame, M_comp, self.crop_pct)

        self._prev_gray = curr_gray
        self._prev_pts  = extract_keypoints(curr_gray, self.detector, boxes)
        self._counter  += 1
        return stabilised


# ── Public stabilizer ─────────────────────────────────────────────────────────

class VideoStabilizer:
    """
    Multi-threaded CPU video stabilizer with queue pressure control.

    Usage::

        vs = VideoStabilizer(detector="ORB", smooth_window=30)
        vs.start()
        vs.put_frame(raw_frame, vehicle_boxes=boxes)   # producer
        stable = vs.get_stable_frame(timeout=0.1)      # consumer
        vs.stop()
    """

    def __init__(
        self,
        detector:      str   = "ORB",
        smooth_window: int   = 30,
        crop_pct:      float = 0.05,
        queue_maxsize: int   = _QUEUE_MAXSIZE,
    ) -> None:
        self._raw_q: queue.Queue[
            Tuple[Frame, List[Tuple[int, int, int, int]]]
        ] = queue.Queue(maxsize=queue_maxsize)
        self._stable_q: queue.Queue[Frame] = queue.Queue(maxsize=queue_maxsize)

        self._proc   = _FrameProcessor(
            detector=detector,
            crop_pct=crop_pct,
            trajectory=TrajectoryBuffer(window=smooth_window),
        )
        self._thread: Optional[threading.Thread] = None
        self._stop   = threading.Event()
        logger.info(
            f"VideoStabilizer ready: detector={detector}, "
            f"smooth_window={smooth_window}, queue_cap={queue_maxsize}"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the daemon stabilization thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("VideoStabilizer already running.")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="stabilizer-worker", daemon=True
        )
        self._thread.start()
        logger.info("VideoStabilizer thread started.")

    def stop(self, timeout: float = 3.0) -> None:
        """Signal stop and join the worker thread."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            logger.info("VideoStabilizer thread stopped.")

    # ── Producer / Consumer ───────────────────────────────────────────────────

    def put_frame(
        self,
        frame: Frame,
        vehicle_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> None:
        """Enqueue raw frame; drop oldest if queue is full."""
        if self._raw_q.full():
            try:
                self._raw_q.get_nowait()
                logger.debug("Raw queue full — oldest frame dropped.")
            except queue.Empty:
                pass
        self._raw_q.put_nowait((frame.copy(), vehicle_boxes or []))

    def get_stable_frame(self, timeout: float = 0.05) -> Optional[Frame]:
        """Pop next stabilized frame, or None on timeout."""
        try:
            return self._stable_q.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def raw_queue_size(self) -> int:
        return self._raw_q.qsize()

    @property
    def stable_queue_size(self) -> int:
        return self._stable_q.qsize()

    # ── Worker ────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        logger.info("Stabilization loop entered.")
        while not self._stop.is_set():
            try:
                frame, boxes = self._raw_q.get(timeout=0.1)
            except queue.Empty:
                continue
            stable = self._proc.process(frame, boxes)
            self._enqueue_stable(stable)
        logger.info("Stabilization loop exited.")

    def _enqueue_stable(self, frame: Frame) -> None:
        if self._stable_q.full():
            try:
                self._stable_q.get_nowait()
            except queue.Empty:
                pass
        self._stable_q.put_nowait(frame)
