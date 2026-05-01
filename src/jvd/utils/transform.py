"""
transform.py — Engine toán học cốt lõi cho chống rung video.

Layer: utils/

Pipeline toán học (chạy từng frame trên CPU):
───────────────────────────────────────────────────
1. Trích xuất đặc trưng : ORB/SIFT trên vùng nền tĩnh
2. Optical Flow         : Theo dõi Lucas-Kanade giữa 2 frame liên tiếp
3. Ước tính Affine      : cv2.estimateAffinePartial2D → ma trận 2x3 [R|t]
4. Quỹ đạo              : Tổng tích lũy các ma trận biến đổi
5. Làm mượt             : Trung bình động (bán kính W) để giảm rung
6. Bù trừ               : Δtransform = smooth_trajectory − raw_trajectory
7. Warp + Crop          : cv2.warpAffine + cắt viền 5%
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

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


# ── Lưu quỹ đạo ─────────────────────────────────────────────────────────────

@dataclass
class TrajectoryPoint:
    """Pose tích lũy ở frame k: (dx, dy, dθ)."""
    dx: float = 0.0
    dy: float = 0.0
    dtheta: float = 0.0


# ── Trích xuất đặc trưng ─────────────────────────────────────────────────────

def extract_keypoints(
    gray: Frame,
    detector: str = "ORB",
    vehicle_boxes: list[tuple[int, int, int, int]] | None = None,
) -> Keypoints:
    """
    Tìm điểm đặc trưng nền tĩnh, loại bỏ vùng xe đang chạy.

    Chiến lược Mask:
    Tạo mask toàn trắng. Tô đen vùng bounding box của các xe
    để thuật toán tracking không bám vào xe đang di chuyển.
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


# ── Optical flow + Ước tính Affine ──────────────────────────────────────────

def estimate_transform(
    prev_gray: Frame,
    curr_gray: Frame,
    prev_pts: Keypoints,
) -> Transform:
    """Ước tính ma trận Affine 2D giữa 2 frame liên tiếp."""
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
        ransacReprojThreshold=2.0, # Ngưỡng chặt để tăng độ chính xác
    )
    return M if M is not None else identity


def get_anchor_compensation(
    anchor_gray: Frame,
    curr_gray: Frame,
    anchor_pts: Keypoints
) -> Transform:
    """
    Tính trực tiếp ma trận biến đổi từ frame HIỆN TẠI về frame NEO.
    Cách này loại bỏ hoàn toàn hiện tượng trôi dạt (drift).
    """
    identity = np.eye(2, 3, dtype=np.float64)
    if anchor_pts is None or len(anchor_pts) < 4:
        return identity

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        anchor_gray, curr_gray, anchor_pts, None, **_LK_PARAMS
    )

    if status is None or np.sum(status) < 10: # Ngưỡng cao hơn cho NEO
        return identity

    good_anchor = anchor_pts[status.ravel() == 1]
    good_curr = curr_pts[status.ravel() == 1]

    # Tìm biến đổi từ HIỆN TẠI về NEO (mapping ngược)
    M, _ = cv2.estimateAffinePartial2D(
        good_curr, good_anchor,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
    )
    return M if M is not None else identity


# ── Quỹ đạo & Làm mượt ──────────────────────────────────────────────────────

class TrajectoryBuffer:
    """
    Tích lũy ma trận Affine vi phân và làm mượt bằng trung bình động 
    để tạo ma trận bù trừ không rung lắc.
    """

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._raw: deque[TrajectoryPoint] = deque(maxlen=window * 2 + 1)
        self._cum = TrajectoryPoint()

    def push(self, M: Transform) -> Transform:
        """
        Nhận một ma trận Affine vi phân và trả về ma trận bù trừ 
        cho frame hiện tại.
        """
        dx = float(M[0, 2])
        dy = float(M[1, 2])
        dtheta = float(np.arctan2(M[1, 0], M[0, 0]))

        self._cum.dx += dx
        self._cum.dy += dy
        self._cum.dtheta += dtheta
        self._raw.append(TrajectoryPoint(self._cum.dx, self._cum.dy, self._cum.dtheta))

        # Trung bình động trên n điểm gần nhất
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


# ── Warp & Crop ─────────────────────────────────────────────────────────────

def warp_and_crop(frame: Frame, M_comp: Transform, crop_pct: float = 0.05) -> Frame:
    """
    Áp dụng warp bù trừ và xóa lỗi viền đen.

    Lỗi viền xảy ra do warp đẩy pixel rời khỏi cạnh. Khắc phục:
    1. BORDER_REPLICATE: nhân bản pixel cạnh thay vì tô đen.
    2. Cắt viền tĩnh: cắt 5% rìa sau đó resize lại kích thước gốc.
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
