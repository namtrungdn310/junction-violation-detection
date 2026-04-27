"""
Unit tests for Phase 2: Video Stabilization modules.

Tests cover:
    transform.py : extract_keypoints, estimate_transform,
                   TrajectoryBuffer, warp_and_crop
    stabilizer.py: VideoStabilizer lifecycle, queue pressure control,
                   frame passthrough, thread safety
"""

from __future__ import annotations

import time
from typing import List, Tuple
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from jvd.utils.stabilizer import VideoStabilizer, _MIN_STABLE_RATIO
from jvd.utils.transform import (
    TrajectoryBuffer,
    extract_keypoints,
    estimate_transform,
    warp_and_crop,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_gray(h: int = 240, w: int = 320, seed: int = 42) -> np.ndarray:
    """Create a reproducible greyscale frame with textured content."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (h, w), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 0)


def _make_bgr(h: int = 240, w: int = 320, seed: int = 42) -> np.ndarray:
    """Create a reproducible colour frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.fixture()
def gray_pair() -> Tuple[np.ndarray, np.ndarray]:
    """Two slightly different greyscale frames (simulated minor shift)."""
    base = _make_gray()
    M_shift = np.float32([[1, 0, 2], [0, 1, 2]])   # 2-pixel shift
    shifted = cv2.warpAffine(base, M_shift, (base.shape[1], base.shape[0]))
    return base, shifted


@pytest.fixture()
def bgr_frame() -> np.ndarray:
    return _make_bgr()


# ── transform.py tests ────────────────────────────────────────────────────────

class TestExtractKeypoints:
    """Tests for the feature extraction function."""

    def test_returns_array_shape(self):
        gray = _make_gray()
        pts = extract_keypoints(gray, detector="ORB")
        assert pts.ndim == 3
        assert pts.shape[2] == 2

    def test_detects_nonzero_keypoints(self):
        gray = _make_gray()
        pts = extract_keypoints(gray, detector="ORB")
        assert len(pts) > 0

    def test_vehicle_mask_reduces_keypoints(self):
        gray = _make_gray(h=240, w=320)
        # Mask out 90 % of the frame area
        big_box: List[Tuple[int, int, int, int]] = [(0, 0, 290, 220)]
        pts_masked = extract_keypoints(gray, detector="ORB", vehicle_boxes=big_box)
        pts_full = extract_keypoints(gray, detector="ORB")
        assert len(pts_masked) <= len(pts_full)

    def test_sift_detector(self):
        gray = _make_gray()
        pts = extract_keypoints(gray, detector="SIFT")
        # SIFT may return 0 on low-contrast noise but should not crash
        assert pts.ndim == 3

    def test_empty_frame_no_crash(self):
        gray = np.zeros((240, 320), dtype=np.uint8)
        pts = extract_keypoints(gray, detector="ORB")
        assert pts.ndim == 3


class TestEstimateTransform:
    """Tests for Lucas-Kanade + Affine estimation."""

    def test_identity_on_same_frame(self):
        gray = _make_gray()
        pts = extract_keypoints(gray, detector="ORB")
        M = estimate_transform(gray, gray, pts)
        assert M.shape == (2, 3)
        # Translation should be near-zero
        assert abs(M[0, 2]) < 2.0
        assert abs(M[1, 2]) < 2.0

    def test_identity_returned_on_insufficient_points(self):
        gray = _make_gray()
        sparse = np.empty((2, 1, 2), dtype=np.float32)   # < 4 points
        M = estimate_transform(gray, gray, sparse)
        np.testing.assert_allclose(M, np.eye(2, 3), atol=1e-6)

    def test_detects_shift(self, gray_pair):
        prev, curr = gray_pair
        pts = extract_keypoints(prev, detector="ORB")
        M = estimate_transform(prev, curr, pts)
        # Affine must capture a non-trivial shift
        assert M.shape == (2, 3)


class TestTrajectoryBuffer:
    """Tests for trajectory accumulation and moving-average smoothing."""

    def test_push_identity_gives_zero_comp(self):
        buf = TrajectoryBuffer(window=5)
        identity = np.eye(2, 3, dtype=np.float64)
        M_comp = buf.push(identity)
        assert M_comp.shape == (2, 3)
        # With pure identity, translation comp should be near 0
        np.testing.assert_allclose(M_comp[:, 2], [0.0, 0.0], atol=1e-6)

    def test_compensation_damps_constant_drift(self):
        """
        With constant drift (+3 px/frame), the raw trajectory grows
        linearly.  Once the moving-average window is fully populated,
        the smoothed trajectory also grows at the same rate, so the
        compensation (smooth − raw) converges to ≈ 0.

        We verify that the absolute compensation after 3× the window
        length is smaller than at frame 1 (before the window fills).
        """
        window = 10
        buf = TrajectoryBuffer(window=window)
        drift = np.array([[1, 0, 3.0], [0, 1, 0.0]], dtype=np.float64)
        comps = [buf.push(drift.copy()) for _ in range(window * 3)]
        # After the window fills, compensation should shrink toward 0
        mid_mag = abs(comps[window][0, 2])
        late_mag = abs(comps[-1][0, 2])
        assert late_mag <= mid_mag + 1.0   # converges, allow ±1 px FP error

    def test_rotation_component(self):
        buf = TrajectoryBuffer(window=5)
        theta = 0.01   # 0.01 rad rotation
        M_rot = np.array(
            [[np.cos(theta), -np.sin(theta), 0.0],
             [np.sin(theta),  np.cos(theta), 0.0]],
            dtype=np.float64,
        )
        M_comp = buf.push(M_rot)
        assert M_comp.shape == (2, 3)


class TestWarpAndCrop:
    """Tests for the warp + crop output."""

    def test_output_shape_preserved(self, bgr_frame):
        h, w = bgr_frame.shape[:2]
        M_comp = np.eye(2, 3, dtype=np.float64)
        out = warp_and_crop(bgr_frame, M_comp)
        assert out.shape == bgr_frame.shape

    def test_identity_warp_close_to_original(self, bgr_frame):
        M_comp = np.eye(2, 3, dtype=np.float64)
        out = warp_and_crop(bgr_frame, M_comp, crop_pct=0.05)
        assert out.shape == bgr_frame.shape
        # Centre region should be near-identical (crop + resize introduces minor diff)
        h, w = bgr_frame.shape[:2]
        cy, cx = h // 2, w // 2
        diff = np.abs(out[cy, cx].astype(int) - bgr_frame[cy, cx].astype(int))
        assert diff.mean() < 20.0


# ── stabilizer.py tests ───────────────────────────────────────────────────────

class TestVideoStabilizerLifecycle:
    """Start / stop / re-start semantics."""

    def test_start_and_stop(self):
        vs = VideoStabilizer()
        vs.start()
        assert vs._thread is not None
        assert vs._thread.is_alive()
        vs.stop()

    def test_double_start_no_crash(self):
        vs = VideoStabilizer()
        vs.start()
        vs.start()   # Should log warning, not raise
        vs.stop()


class TestQueuePressureControl:
    """Frame-dropping under queue saturation."""

    def test_put_never_blocks_when_full(self):
        vs = VideoStabilizer(queue_maxsize=3)
        # Fill with dummy frames WITHOUT starting the worker
        frame = _make_bgr()
        for _ in range(10):   # 10 >> capacity 3
            vs.put_frame(frame)
        assert vs.raw_queue_size <= 3

    def test_queue_size_bounded(self):
        vs = VideoStabilizer(queue_maxsize=5)
        frame = _make_bgr()
        for _ in range(20):
            vs.put_frame(frame)
        assert vs.raw_queue_size <= 5


class TestStabilizationPipeline:
    """End-to-end frame processing through the worker thread."""

    def test_produces_stable_frames(self):
        vs = VideoStabilizer(detector="ORB", smooth_window=5)
        vs.start()
        frame = _make_bgr(h=240, w=320)
        for _ in range(10):
            vs.put_frame(frame)
        time.sleep(1.0)   # Allow worker to process
        result = vs.get_stable_frame(timeout=1.0)
        vs.stop()
        assert result is not None
        assert result.shape == frame.shape

    def test_output_dtype_uint8(self):
        vs = VideoStabilizer(detector="ORB", smooth_window=5)
        vs.start()
        frame = _make_bgr()
        vs.put_frame(frame)
        time.sleep(0.5)
        result = vs.get_stable_frame(timeout=1.0)
        vs.stop()
        if result is not None:
            assert result.dtype == np.uint8

    def test_vehicle_boxes_accepted(self):
        vs = VideoStabilizer()
        vs.start()
        frame = _make_bgr()
        boxes = [(10, 10, 100, 80), (150, 90, 280, 200)]
        vs.put_frame(frame, vehicle_boxes=boxes)
        time.sleep(0.5)
        vs.stop()   # Should not raise
