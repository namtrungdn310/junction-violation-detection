"""
Các kiểu dữ liệu trung gian bất biến (immutable).

Toàn bộ dữ liệu trao đổi giữa các module đều dùng các dataclass này.
frozen=True đảm bảo an toàn luồng và tránh thay đổi ngoài ý muốn.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class VehicleClass(Enum):
    """Các loại phương tiện được nhận diện trên đường Việt Nam."""

    CAR = auto()
    MOTORBIKE = auto()
    TRUCK = auto()
    BUS = auto()
    VAN = auto()
    UNKNOWN = auto()

    @classmethod
    def from_label(cls, label: str) -> VehicleClass:
        """Chuyển nhãn chuỗi YOLO sang VehicleClass (không phân biệt hoa/thường)."""
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
        Chuyển chỉ số lớp COCO sang VehicleClass.

        Ánh xạ COCO được dùng:
        | ID |  Nhãn COCO  | VehicleClass |
        |----|-------------|--------------|
        |  2 | car         | CAR          |
        |  3 | motorcycle  | MOTORBIKE    |
        |  5 | bus         | BUS          |
        |  7 | truck       | TRUCK        |
        """
        _COCO_MAP = {2: cls.CAR, 3: cls.MOTORBIKE, 5: cls.BUS, 7: cls.TRUCK}
        return _COCO_MAP.get(coco_id, cls.UNKNOWN)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """
    Hộp giới hạn căn chỉnh theo trục (pixel).

    Thuộc tính:
        x1: Cạnh trái.
        y1: Cạnh trên.
        x2: Cạnh phải.
        y2: Cạnh dưới.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Chiều rộng của hộp."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Chiều cao của hộp."""
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        """Tâm điểm (cx, cy) của hộp."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        """Diện tích hộp (pixel²)."""
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(frozen=True, slots=True)
class DetectionEvent:
    """
    Đơn vị dữ liệu chuẩn cho một lần phát hiện phương tiện.

    Là đầu ra chung của các giai đoạn: detector → tracker → analyzer → reporter.

    Thuộc tính:
        frame_id:    Số thứ tự frame trong video.
        timestamp:   Thời điểm (giây) kể từ đầu video.
        bbox:        Hộp giới hạn của phương tiện.
        track_id:    ID theo dõi do ByteTrack gán (None nếu chưa có).
        class_label: Loại phương tiện.
        confidence:  Độ tin cậy nhận diện trong khoảng [0.0, 1.0].
        velocity:    Vector vận tốc (vx, vy) pixel/frame (None nếu chưa tính).
    """

    frame_id: int
    timestamp: float
    bbox: BoundingBox
    track_id: int | None = None
    class_label: VehicleClass = VehicleClass.UNKNOWN
    confidence: float = 0.0
    velocity: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Kiểm tra ràng buộc khi khởi tạo."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(
                f"confidence phải trong [0.0, 1.0], nhận được {self.confidence}"
            )
        if self.frame_id < 0:
            raise ValueError(
                f"frame_id phải >= 0, nhận được {self.frame_id}"
            )


@dataclass(frozen=True, slots=True)
class ViolationRecord:
    """
    Một vi phạm đã được xác nhận, sẵn sàng để lưu bằng chứng.

    Được tạo khi phương tiện dừng trên vạch mắt võng quá thời gian quy định.
    """

    event: DetectionEvent
    license_plate: str | None = None
    dwell_time_seconds: float = 0.0
    evidence_frame_path: str | None = None
    violation_type: str = "yellow_box_stop"
