"""
Custom exception hierarchy for the JVD system.

All exceptions inherit from JVDBaseError to enable granular
catch blocks without masking unrelated system errors.
"""

from __future__ import annotations


class JVDBaseError(Exception):
    """Base exception for all JVD-specific errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class HardwareConstraintError(JVDBaseError):
    """
    Raised when a hardware requirement cannot be satisfied.

    This exception is deliberately fatal — the system MUST NOT
    silently fall back to CPU when GPU is explicitly requested.
    Doing so would produce misleading latency benchmarks and
    mask deployment misconfigurations on edge devices.
    """

    def __init__(self, message: str) -> None:
        super().__init__(f"[HARDWARE] {message}")


class VRAMExceededError(JVDBaseError):
    """Raised when VRAM allocation exceeds the configured hard limit."""

    def __init__(self, requested_gb: float, limit_gb: float) -> None:
        super().__init__(
            f"[VRAM] Requested {requested_gb:.2f} GB exceeds "
            f"hard limit of {limit_gb:.2f} GB"
        )


class PipelineConfigError(JVDBaseError):
    """Raised when pipeline configuration is invalid or incomplete."""

    def __init__(self, message: str) -> None:
        super().__init__(f"[CONFIG] {message}")
