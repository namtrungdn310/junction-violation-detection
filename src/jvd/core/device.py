"""
DeviceManager — Singleton controller for hardware resource allocation.

This module enforces strict GPU/VRAM discipline for edge deployment
scenarios where VRAM is critically limited (e.g., RTX A500 with < 3 GB).

Design decisions:
    1. Singleton pattern ensures ONE device context across all modules.
    2. GPU request with no CUDA → raises HardwareConstraintError (no silent fallback).
    3. VRAM is hard-capped via torch.cuda.set_per_process_memory_fraction
       to prevent OOM crashes that would kill the entire pipeline.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import torch

from jvd.core.exceptions import HardwareConstraintError

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Thread-safe Singleton that owns the torch.device lifecycle.

    Usage::

        dm = DeviceManager.get_instance()
        dm.initialize(device_flag="gpu", vram_limit_gb=2.8)
        tensor = torch.zeros(10, device=dm.device)
    """

    _instance: Optional[DeviceManager] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    # ── Singleton access ────────────────────────────────────────────

    def __new__(cls) -> DeviceManager:
        """Guarantee a single instance across the entire process."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking for thread safety
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> DeviceManager:
        """Return the singleton instance (create if needed)."""
        return cls()

    # ── Initialization ──────────────────────────────────────────────

    def initialize(
        self,
        device_flag: str = "gpu",
        vram_limit_gb: float = 2.8,
    ) -> None:
        """
        Configure the device context exactly once.

        Args:
            device_flag: ``"gpu"`` or ``"cpu"`` from the ``--device`` CLI flag.
            vram_limit_gb: Maximum VRAM allocation in gigabytes.
                           Default 2.8 GB leaves ~200 MB headroom for OS
                           background tasks on a 3 GB card.

        Raises:
            HardwareConstraintError: If GPU is requested but CUDA is unavailable,
                                     or if the requested VRAM exceeds physical capacity.
        """
        if self._initialized:
            logger.warning("DeviceManager already initialized — skipping.")
            return

        device_flag = device_flag.strip().lower()

        if device_flag == "gpu":
            self._ensure_gpu_available()
            self._device = torch.device("cuda:0")
            self._apply_vram_constraint(vram_limit_gb)
            self._log_gpu_info()
        elif device_flag == "cpu":
            self._device = torch.device("cpu")
            logger.info("Device configured: CPU (no VRAM constraint applied)")
        else:
            raise HardwareConstraintError(
                f"Unrecognized device flag '{device_flag}'. "
                f"Expected 'gpu' or 'cpu'."
            )

        self._initialized = True
        logger.info(f"DeviceManager initialized → {self._device}")

    # ── Public properties ───────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        """Return the active torch device."""
        if not self._initialized:
            raise HardwareConstraintError(
                "DeviceManager has not been initialized. "
                "Call initialize() before accessing .device"
            )
        return self._device

    @property
    def is_gpu(self) -> bool:
        """Return True if the active device is a CUDA GPU."""
        return self._initialized and self._device.type == "cuda"

    # ── Private helpers ─────────────────────────────────────────────

    @staticmethod
    def _ensure_gpu_available() -> None:
        """
        Verify CUDA hardware is physically present.

        Raises:
            HardwareConstraintError: Immediately halts the process
                if torch.cuda.is_available() returns False.
                We deliberately do NOT fall back to CPU.
        """
        if not torch.cuda.is_available():
            raise HardwareConstraintError(
                "GPU was explicitly requested via --device gpu, "
                "but torch.cuda.is_available() returned False. "
                "Possible causes:\n"
                "  1. No NVIDIA GPU detected on this machine.\n"
                "  2. CUDA drivers are not installed or incompatible.\n"
                "  3. PyTorch was installed without CUDA support.\n"
                "System will NOT silently fall back to CPU. "
                "Fix the hardware configuration or use --device cpu."
            )

    def _apply_vram_constraint(self, limit_gb: float) -> None:
        """
        Hard-cap the per-process VRAM allocation.

        The fraction is computed as ``limit_gb / total_vram_gb``,
        clamped to [0.1, 0.95] for safety.

        Args:
            limit_gb: Desired maximum VRAM in gigabytes.

        Raises:
            HardwareConstraintError: If the requested limit exceeds
                the total physical VRAM of the GPU.
        """
        total_vram_bytes = torch.cuda.get_device_properties(0).total_mem
        total_vram_gb = total_vram_bytes / (1024 ** 3)

        if limit_gb > total_vram_gb:
            raise HardwareConstraintError(
                f"Requested VRAM limit ({limit_gb:.2f} GB) exceeds "
                f"total GPU memory ({total_vram_gb:.2f} GB)."
            )

        fraction = self._compute_fraction(limit_gb, total_vram_gb)
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)

        logger.info(
            f"VRAM constraint applied: {limit_gb:.2f} GB / "
            f"{total_vram_gb:.2f} GB (fraction={fraction:.4f})"
        )

    @staticmethod
    def _compute_fraction(limit_gb: float, total_gb: float) -> float:
        """
        Compute the memory fraction, clamped to safe bounds.

        Args:
            limit_gb: Desired VRAM cap.
            total_gb: Total physical VRAM.

        Returns:
            A float in [0.10, 0.95].
        """
        raw = limit_gb / total_gb
        return max(0.10, min(0.95, raw))

    def _log_gpu_info(self) -> None:
        """Log diagnostic GPU information."""
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_mem / (1024 ** 3)
        logger.info(
            f"GPU: {props.name} | "
            f"Total VRAM: {total_gb:.2f} GB | "
            f"CUDA Capability: {props.major}.{props.minor}"
        )

    # ── Teardown (for testing) ──────────────────────────────────────

    @classmethod
    def _reset(cls) -> None:
        """Reset the singleton state. For unit tests ONLY."""
        with cls._lock:
            cls._instance = None
            cls._initialized = False
