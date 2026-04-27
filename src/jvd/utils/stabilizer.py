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
    Keypoints,
    get_anchor_compensation,
    extract_keypoints,
    warp_and_crop,
)

logger = logging.getLogger(__name__)

_QUEUE_MAXSIZE   = 60
_REINIT_INTERVAL = 60     # Every 60 frames, refresh points from anchor
_MIN_ANCHOR_KPS  = 20     # Minimum keypoints to consider anchor tracking valid


@dataclass
class _FrameProcessor:
    """Stateful per-frame stabilization processor using Anchor Reference.
    """
    detector:   str            = "ORB" # Switch back to ORB for speed
    crop_pct:   float          = 0.05
    _anchor_gray: Optional[Frame] = field(default=None, init=False, repr=False)
    _anchor_pts:  Optional[Keypoints] = field(default=None, init=False, repr=False)
    _counter:   int               = field(default=0,    init=False, repr=False)
    _last_M:    Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def process(
        self,
        frame: Frame,
        boxes: List[Tuple[int, int, int, int]],
    ) -> Transform:
        """Estimate transform from Current back to Anchor (Frame 0).
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize Anchor on the very first frame
        if self._anchor_gray is None:
            self._anchor_gray = curr_gray
            self._anchor_pts  = extract_keypoints(curr_gray, self.detector, boxes)
            self._last_M      = np.eye(2, 3, dtype=np.float64)
            return self._last_M

        if self._counter % _REINIT_INTERVAL == 0:
             self._anchor_pts = extract_keypoints(self._anchor_gray, self.detector, boxes)

        M_comp = get_anchor_compensation(self._anchor_gray, curr_gray, self._anchor_pts)
        
        # Smoothing
        if self._last_M is not None:
            alpha = 0.5  # More smoothing for ROI motion
            M_comp = alpha * M_comp + (1 - alpha) * self._last_M
            self._last_M = M_comp

        self._counter += 1
        return M_comp


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
        self._matrix_q: queue.Queue[Tuple[Frame, np.ndarray]] = queue.Queue(maxsize=queue_maxsize)

        self._proc   = _FrameProcessor(
            detector=detector,
            crop_pct=crop_pct,
        )
        self._thread: Optional[threading.Thread] = None
        self._stop   = threading.Event()
        logger.info(
            f"MotionEstimator ready: detector={detector}, queue_cap={queue_maxsize}"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the daemon motion estimation thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="motion-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    # ── Producer / Consumer ───────────────────────────────────────────────────

    def put_frame(
        self,
        frame: Frame,
        vehicle_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> None:
        if self._raw_q.full():
            try:
                self._raw_q.get_nowait()
            except queue.Empty:
                pass
        self._raw_q.put_nowait((frame.copy(), vehicle_boxes or []))

    def get_latest(self, timeout: float = 0.05) -> Tuple[Optional[Frame], Optional[np.ndarray]]:
        """Pop next (raw_frame, matrix), or (None, None) on timeout."""
        try:
            return self._matrix_q.get(timeout=timeout)
        except queue.Empty:
            return None, None

    # ── Worker ────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame, boxes = self._raw_q.get(timeout=0.1)
            except queue.Empty:
                continue
            matrix = self._proc.process(frame, boxes)
            self._enqueue_matrix(frame, matrix)

    def _enqueue_matrix(self, frame: Frame, matrix: np.ndarray) -> None:
        if self._matrix_q.full():
            try:
                self._matrix_q.get_nowait()
            except queue.Empty:
                pass
        self._matrix_q.put_nowait((frame, matrix))
