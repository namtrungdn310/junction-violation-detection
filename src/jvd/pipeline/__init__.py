"""
Pipeline layer: End-to-end processing orchestration.

This is the top-level layer that composes all lower layers.
It may import from core/, inference/, and utils/.
No other layer should import from pipeline/.

Public API
~~~~~~~~~~
- RegionOfInterest  : Handles normalized yellow-box polygons
- ViolationAnalyzer : Processes vehicle kinematics and spatial constraints
"""

from jvd.pipeline.analyzer import RegionOfInterest, ViolationAnalyzer

__all__ = [
    "RegionOfInterest",
    "ViolationAnalyzer",
]
