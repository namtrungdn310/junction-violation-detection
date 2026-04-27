"""
Data models — Immutable intermediary data structures.

All cross-module data flows through these standardized dataclasses.
Using frozen=True ensures thread-safety and prevents accidental
mutation during multi-stage pipeline processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class VehicleClass(Enum):
    """Enumeration of detectable vehicle types on Vietnamese roads."""

    CAR = auto()
    MOTORBIKE = auto()
    TRUCK = auto()
    BUS = auto()
    VAN = auto()
    UNKNOWN = auto()

    @classmethod
    def from_label(cls, label: str) -> VehicleClass:
        """
        Map a YOLO class label string to a VehicleClass enum.

        Args:
            label: Raw string label from the detector (case-insensitive).

        Returns:
            Matching VehicleClass or UNKNOWN if unrecognized.
        """
        mapping = {
            "car": cls.CAR,
            "motorbike": cls.MOTORBIKE,
            "motorcycle": cls.MOTORBIKE,
            "truck": cls.TRUCK,
            "bus": cls.BUS,
            "van": cls.VAN,
        }
        return mapping.get(label.strip().lower(), cls.UNKNOWN)

    @classmethod
    def from_coco_id(cls, coco_id: int) -> VehicleClass:
        """
        Map a COCO dataset integer class index to a VehicleClass enum.

        COCO indices used by this system
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        | ID |  COCO Label  | VehicleClass |
        |----|--------------|--------------|
        |  2 | car          | CAR          |
        |  3 | motorcycle   | MOTORBIKE    |
        |  5 | bus          | BUS          |
        |  7 | truck        | TRUCK        |

        Args:
            coco_id: Integer class index from YOLO / COCO dataset.

        Returns:
            Matching VehicleClass or UNKNOWN for unmapped IDs.
        """
        _COCO_MAP = {2: cls.CAR, 3: cls.MOTORBIKE, 5: cls.BUS, 7: cls.TRUCK}
        return _COCO_MAP.get(coco_id, cls.UNKNOWN)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """
    Axis-aligned bounding box in pixel coordinates.

    Attributes:
        x1: Left edge.
        y1: Top edge.
        x2: Right edge.
        y2: Bottom edge.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        """Center point (cx, cy) of the bounding box."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        """Area of the bounding box in square pixels."""
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(frozen=True, slots=True)
class DetectionEvent:
    """
    Standardized intermediary data structure for a single detection.

    This is the canonical unit of data that flows between pipeline
    stages (detector → tracker → violation analyzer → evidence recorder).

    Attributes:
        frame_id:    Sequential frame index from the video source.
        timestamp:   Wall-clock time in seconds since stream start.
        bbox:        Bounding box of the detected object.
        track_id:    Unique ID assigned by the object tracker.
                     None if tracking has not yet been applied.
        class_label: Detected vehicle class.
        confidence:  Detection confidence score in [0.0, 1.0].
        velocity:    Velocity vector (vx, vy) in pixels/frame.
                     None if optical-flow has not yet been computed.
    """

    frame_id: int
    timestamp: float
    bbox: BoundingBox
    track_id: int | None = None
    class_label: VehicleClass = VehicleClass.UNKNOWN
    confidence: float = 0.0
    velocity: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Validate field constraints on construction."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if self.frame_id < 0:
            raise ValueError(
                f"frame_id must be non-negative, got {self.frame_id}"
            )


@dataclass(frozen=True, slots=True)
class ViolationRecord:
    """
    A confirmed violation event ready for evidence archival.

    Created when a vehicle is determined to have stopped on
    the yellow-box junction markings beyond the threshold duration.
    """

    event: DetectionEvent
    license_plate: str | None = None
    dwell_time_seconds: float = 0.0
    evidence_frame_path: str | None = None
    violation_type: str = "yellow_box_stop"
