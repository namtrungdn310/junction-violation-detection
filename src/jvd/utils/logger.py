"""
Cấu hình logging tập trung cho hệ thống JVD.

Cung cấp định dạng log chuẩn trên tất cả module
cho cả terminal (console) và file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = (
    "%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_DIR = Path("outputs/logs")
_configured: bool = False


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """
    Tạo hoặc lấy logger với định dạng chuẩn.

    Ở lần gọi đầu tiên, hàm này cấu hình root logger với 
    console handler và (tùy chọn) file handler.

    Args:
        name: Tên logger (thường là __name__).
        level: Mức log tối thiểu. Mặc định: INFO.
        log_file: Tên file log (tùy chọn) lưu trong outputs/logs/.

    Returns:
        Instance logging.Logger đã được cấu hình.
    """
    global _configured

    if not _configured:
        _configure_root(level, log_file)
        _configured = True

    return logging.getLogger(name)


def _configure_root(level: int, log_file: str | None) -> None:
    """Thiết lập root logger với console + file handler (tùy chọn)."""
    root = logging.getLogger()
    root.setLevel(level)

    # ── Console handler ─────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(console)

    # ── File handler (tùy chọn) ─────────────────────────────────
    if log_file:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            _LOG_DIR / log_file, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)
        )
        root.addHandler(file_handler)
