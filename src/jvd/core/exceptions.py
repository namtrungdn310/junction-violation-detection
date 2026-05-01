"""
Hệ thống ngoại lệ tùy chỉnh của JVD.

Tất cả exception kế thừa từ JVDBaseError để dễ bắt lỗi
mà không ảnh hưởng đến các lỗi hệ thống không liên quan.
"""

from __future__ import annotations


class JVDBaseError(Exception):
    """Lớp cơ sở cho mọi exception của hệ thống JVD."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class HardwareConstraintError(JVDBaseError):
    """
    Lỗi khi không đáp ứng được yêu cầu phần cứng.

    Exception này là lỗi nghiêm trọng — hệ thống KHÔNG tự fallback sang CPU
    khi người dùng đã yêu cầu GPU, để tránh kết quả benchmark sai lệch.
    """

    def __init__(self, message: str) -> None:
        super().__init__(f"[HARDWARE] {message}")


class VRAMExceededError(JVDBaseError):
    """Lỗi khi yêu cầu VRAM vượt giới hạn đã cấu hình."""

    def __init__(self, requested_gb: float, limit_gb: float) -> None:
        super().__init__(
            f"[VRAM] Yêu cầu {requested_gb:.2f} GB vượt giới hạn {limit_gb:.2f} GB"
        )


class PipelineConfigError(JVDBaseError):
    """Lỗi khi cấu hình pipeline không hợp lệ hoặc thiếu thông tin."""

    def __init__(self, message: str) -> None:
        super().__init__(f"[CONFIG] {message}")
