"""
Centralized logging configuration for the JVD system.

Provides a consistent log format across all modules with
both console and file output handlers.
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
    Create or retrieve a named logger with standardized formatting.

    On the first call, this function also configures the root logger
    with a console handler and (optionally) a file handler.

    Args:
        name: Logger name (typically ``__name__`` or a module path).
        level: Minimum log level. Default: INFO.
        log_file: Optional filename inside ``outputs/logs/``.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    global _configured

    if not _configured:
        _configure_root(level, log_file)
        _configured = True

    return logging.getLogger(name)


def _configure_root(level: int, log_file: str | None) -> None:
    """Set up the root logger with console + optional file handlers."""
    root = logging.getLogger()
    root.setLevel(level)

    # ── Console handler ─────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(console)

    # ── File handler (optional) ─────────────────────────────────
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
