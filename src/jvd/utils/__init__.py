"""
Utility layer: Logging, video I/O, stabilization helpers.

Modules in this layer may import from core/ but never from
inference/ or pipeline/.

Public API
~~~~~~~~~~
- VideoStabilizer : Multi-threaded CPU-only video stabilizer
- transform       : Feature extraction, optical flow, affine math
"""

from jvd.utils.stabilizer import VideoStabilizer
from jvd.utils.transform import (
    TrajectoryBuffer,
    extract_keypoints,
    estimate_transform,
    warp_and_crop,
)

__all__ = [
    "VideoStabilizer",
    "TrajectoryBuffer",
    "extract_keypoints",
    "estimate_transform",
    "warp_and_crop",
]
