"""
transform.py — Core mathematical engine for video stabilization.

Layer: utils/  (imports from core/ only)

Mathematical pipeline (executed per-frame on CPU):
───────────────────────────────────────────────────
1. Feature Extraction  : ORB or SIFT keypoints on static background pixels
2. Optical Flow        : Lucas–Kanade pyramid tracking between consecutive frames
3. Affine Estimation   : cv2.estimateAffinePartial2D → 2×3 matrix [R|t]
4. Trajectory          : Cumulative sum of differential transforms
5. Smoothing           : Moving-average window (radius W) to suppress vibration
6. Compensation        : Δtransform = smooth_trajectory − raw_trajectory
7. Warp + Crop         : cv2.warpAffine (BORDER_REPLICATE) + 5 % static crop
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────────
Frame = np.ndarray          # H×W×C uint8, BGR
Keypoints = np.ndarray      # shape (N, 1, 2) float32
Transform = np.ndarray      # shape (2, 3) float64 Affine matrix

_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
_ORB_N_FEATURES = 2000


# ── Trajectory record ─────────────────────────────────────────────────────────

@dataclass
class TrajectoryPoint:
    """Cumulative pose at frame k: (dx, dy, dθ)."""
    dx: float = 0.0
    dy: float = 0.0
    dtheta: float = 0.0


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_keypoints(
    gray: Frame,
    detector: str = "ORB",
    vehicle_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Keypoints:
    """
    Detect static-background keypoints, masking out moving vehicles.

    Mask strategy
    ~~~~~~~~~~~~~
    A binary mask M is initialised to all-255 (white = valid).
    For each bounding box (x1, y1, x2, y2) in ``vehicle_boxes``,
    the rectangular region M[y1:y2, x1:x2] is set to 0 (black = ignore).
    This prevents the tracker from latching onto moving objects that would
    otherwise corrupt the homography estimate.

    Args:
        gray:          Single-channel (H, W) uint8 greyscale frame.
        detector:      ``"ORB"`` (fast, no licence) or ``"SIFT"`` (robust).
        vehicle_boxes: List of (x1, y1, x2, y2) pixel boxes to mask out.

    Returns:
        Keypoints array of shape (N, 1, 2) float32 suitable for
        ``cv2.calcOpticalFlowPyrLK``.  Returns empty array on failure.
    """
    mask = np.full(gray.shape[:2], 255, dtype=np.uint8)
    if vehicle_boxes:
        for x1, y1, x2, y2 in vehicle_boxes:
            mask[max(0, y1):y2, max(0, x1):x2] = 0

    if detector == "SIFT":
        det = cv2.SIFT_create()
    else:
        det = cv2.ORB_create(nfeatures=_ORB_N_FEATURES)

    kps = det.detect(gray, mask=mask)
    if not kps:
        return np.empty((0, 1, 2), dtype=np.float32)
    pts = np.array([[kp.pt] for kp in kps], dtype=np.float32)
    return pts


# ── Optical flow + Affine estimation ─────────────────────────────────────────

def estimate_transform(
    prev_gray: Frame,
    curr_gray: Frame,
    prev_pts: Keypoints,
) -> Transform:
    """
    Estimate the 2-D Affine transform between two consecutive frames.
    """
    identity = np.eye(2, 3, dtype=np.float64)

    if prev_pts is None or len(prev_pts) < 4:
        return identity

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **_LK_PARAMS
    )
    if status is None or np.sum(status) < 4:
        return identity

    good_prev = prev_pts[status.ravel() == 1]
    good_curr = curr_pts[status.ravel() == 1]

    M, _ = cv2.estimateAffinePartial2D(
        good_prev, good_curr,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0, # Tighter threshold for accuracy
    )
    return M if M is not None else identity


def get_anchor_compensation(
    anchor_gray: Frame,
    curr_gray: Frame,
    anchor_pts: Keypoints
) -> Transform:
    """
    Directly calculate the transform from current frame back to anchor frame.
    This eliminates drift entirely.
    """
    identity = np.eye(2, 3, dtype=np.float64)
    if anchor_pts is None or len(anchor_pts) < 4:
        return identity

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        anchor_gray, curr_gray, anchor_pts, None, **_LK_PARAMS
    )
    
    if status is None or np.sum(status) < 10: # Higher threshold for anchor
        return identity

    good_anchor = anchor_pts[status.ravel() == 1]
    good_curr = curr_pts[status.ravel() == 1]

    # Find transform from CURRENT to ANCHOR (inverse mapping)
    M, _ = cv2.estimateAffinePartial2D(
        good_curr, good_anchor,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
    )
    return M if M is not None else identity


# ── Trajectory & smoothing ────────────────────────────────────────────────────

class TrajectoryBuffer:
    """
    Accumulates differential Affine transforms and applies moving-average
    smoothing to derive a vibration-free compensation matrix.

    Smoothing formula (window radius W = 30 frames)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Raw cumulative trajectory at frame k:
        T_raw[k] = Σ_{i=0}^{k} (dx_i, dy_i, dθ_i)

    Smoothed trajectory (causal moving average):
        T_smooth[k] = (1 / |W_k|) · Σ_{j=k-W}^{k} T_raw[j]
        where W_k = min(W, k)  — truncated at stream start.

    Compensation transform decoded from:
        Δ = T_smooth[k] − T_raw[k]

    The compensation 2×3 matrix is then:
        M_comp = [cos Δθ   −sin Δθ   Δtx]
                 [sin Δθ    cos Δθ   Δty]
    """

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._raw: deque[TrajectoryPoint] = deque(maxlen=window * 2 + 1)
        self._cum = TrajectoryPoint()

    def push(self, M: Transform) -> Transform:
        """
        Ingest one differential Affine matrix and return the compensation
        matrix for the current frame.

        Args:
            M: 2×3 Affine matrix from ``estimate_transform``.

        Returns:
            2×3 compensation Affine matrix.
        """
        dx = float(M[0, 2])
        dy = float(M[1, 2])
        dtheta = float(np.arctan2(M[1, 0], M[0, 0]))

        self._cum.dx += dx
        self._cum.dy += dy
        self._cum.dtheta += dtheta
        self._raw.append(TrajectoryPoint(self._cum.dx, self._cum.dy, self._cum.dtheta))

        # Moving average over last `window` points
        n = min(len(self._raw), self._window)
        recent: Sequence[TrajectoryPoint] = list(self._raw)[-n:]
        smooth_dx = sum(p.dx for p in recent) / n
        smooth_dy = sum(p.dy for p in recent) / n
        smooth_dt = sum(p.dtheta for p in recent) / n

        delta_dx = smooth_dx - self._cum.dx
        delta_dy = smooth_dy - self._cum.dy
        delta_dt = smooth_dt - self._cum.dtheta

        cos_t, sin_t = np.cos(delta_dt), np.sin(delta_dt)
        return np.array(
            [[cos_t, -sin_t, delta_dx],
             [sin_t,  cos_t, delta_dy]],
            dtype=np.float64,
        )


# ── Warp & crop ───────────────────────────────────────────────────────────────

def warp_and_crop(frame: Frame, M_comp: Transform, crop_pct: float = 0.05) -> Frame:
    """
    Apply compensation warp and remove black border artefacts.

    Border artefacts arise because the affine warp shifts pixel content
    away from the canvas edges, leaving unfilled regions.  Two mitigations:
    1. ``cv2.BORDER_REPLICATE`` — fills border pixels by replicating the
       nearest edge pixel rather than inserting black zeros.
    2. Static crop — removes ``crop_pct`` (default 5 %) of rows/columns
       from all four edges and bilinear-resizes back to the original
       resolution so downstream models receive consistent frame dimensions.

    Args:
        frame:    Input BGR frame (H, W, 3) uint8.
        M_comp:   2×3 compensation Affine matrix.
        crop_pct: Fraction of each edge to crop (default 0.05 → 5 %).

    Returns:
        Stabilised BGR frame at the original (H, W) resolution.
    """
    h, w = frame.shape[:2]
    stabilised = cv2.warpAffine(
        frame, M_comp, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    dx = int(w * crop_pct)
    dy = int(h * crop_pct)
    cropped = stabilised[dy: h - dy, dx: w - dx]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
