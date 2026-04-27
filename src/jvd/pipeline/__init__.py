"""
Pipeline layer: End-to-end processing orchestration.

This is the top-level layer that composes all lower layers.
It may import from core/, inference/, and utils/.
No other layer should import from pipeline/.

Public API
~~~~~~~~~~
- RegionOfInterest         : Handles normalized yellow-box polygons
- ViolationAnalyzer        : Processes vehicle kinematics and spatial constraints
- EmergencyVehicleDetector : Optical heuristic detection for emergency vehicles
- OSDRenderer              : Visual overlays and state rendering
- ViolationReporter        : Evidence generation (MP4, JSON, Images)
"""

from jvd.pipeline.analyzer import RegionOfInterest, ViolationAnalyzer
from jvd.pipeline.emergency import EmergencyVehicleDetector
from jvd.pipeline.osd import OSDRenderer
from jvd.pipeline.reporter import ViolationReporter

__all__ = [
    "RegionOfInterest",
    "ViolationAnalyzer",
    "EmergencyVehicleDetector",
    "OSDRenderer",
    "ViolationReporter",
]
