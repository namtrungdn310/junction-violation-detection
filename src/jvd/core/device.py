"""
DeviceManager — Singleton quản lý tài nguyên phần cứng GPU/CPU.

Kiểm soát nghiêm ngặt VRAM cho môi trường triển khai thiếu tài nguyên
(ví dụ RTX A500 với < 3 GB khả dụng).

Thiết kế:
    1. Singleton đảm bảo toàn bộ hệ thống dùng chung một context thiết bị.
    2. Yêu cầu GPU mà không có CUDA → raise HardwareConstraintError (không fallback thầm lặng).
    3. VRAM bị giới hạn cứng qua torch.cuda.set_per_process_memory_fraction
       để tránh OOM crash làm chết toàn bộ pipeline.
"""

from __future__ import annotations

import logging
import threading

import torch

from jvd.core.exceptions import HardwareConstraintError

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Singleton thread-safe quản lý vòng đời torch.device.

    Ví dụ sử dụng::

        dm = DeviceManager.get_instance()
        dm.initialize(device_flag="gpu", vram_limit_gb=2.8)
        tensor = torch.zeros(10, device=dm.device)
    """

    _instance: DeviceManager | None = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    # ── Truy cập Singleton ──────────────────────────────────────────

    def __new__(cls) -> DeviceManager:
        """Đảm bảo chỉ tạo một instance duy nhất trong toàn process."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking để an toàn luồng
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> DeviceManager:
        """Trả về instance singleton (tạo mới nếu chưa có)."""
        return cls()

    # ── Khởi tạo ────────────────────────────────────────────────────

    def initialize(
        self,
        device_flag: str = "gpu",
        vram_limit_gb: float = 2.8,
    ) -> None:
        """
        Cấu hình thiết bị một lần duy nhất.

        Args:
            device_flag:   ``"gpu"`` hoặc ``"cpu"`` từ tham số ``--device``.
            vram_limit_gb: Giới hạn VRAM tối đa (GB). Mặc định 2.8 GB để
                           dành ~200 MB cho hệ thống trên card 3 GB.

        Raises:
            HardwareConstraintError: Nếu yêu cầu GPU nhưng CUDA không khả dụng,
                                     hoặc VRAM yêu cầu vượt dung lượng thực.
        """
        if self._initialized:
            logger.warning("DeviceManager đã được khởi tạo — bỏ qua.")
            return

        device_flag = device_flag.strip().lower()

        if device_flag == "gpu":
            self._ensure_gpu_available()
            self._device = torch.device("cuda:0")
            self._apply_vram_constraint(vram_limit_gb)
            self._log_gpu_info()
        elif device_flag == "cpu":
            self._device = torch.device("cpu")
            logger.info("Thiết bị: CPU (không áp dụng giới hạn VRAM)")
        else:
            raise HardwareConstraintError(
                f"Tham số device không hợp lệ '{device_flag}'. "
                f"Chỉ chấp nhận 'gpu' hoặc 'cpu'."
            )

        self._initialized = True
        logger.info(f"DeviceManager initialized → {self._device}")

    # ── Thuộc tính công khai ─────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        """Trả về thiết bị torch đang hoạt động."""
        if not self._initialized:
            raise HardwareConstraintError(
                "DeviceManager chưa được khởi tạo. "
                "Gọi initialize() trước khi dùng .device"
            )
        return self._device

    @property
    def is_gpu(self) -> bool:
        """Trả về True nếu đang chạy trên GPU CUDA."""
        return self._initialized and self._device.type == "cuda"

    # ── Hàm nội bộ ──────────────────────────────────────────────────

    @staticmethod
    def _ensure_gpu_available() -> None:
        """
        Kiểm tra GPU CUDA có tồn tại không.

        Raises:
            HardwareConstraintError: Nếu torch.cuda.is_available() = False.
                Hệ thống KHÔNG tự động chuyển sang CPU.
        """
        if not torch.cuda.is_available():
            raise HardwareConstraintError(
                "Yêu cầu GPU (--device gpu) nhưng torch.cuda.is_available() = False.\n"
                "Nguyên nhân có thể:\n"
                "  1. Không có GPU NVIDIA trên máy.\n"
                "  2. Driver CUDA chưa cài hoặc không tương thích.\n"
                "  3. PyTorch được cài không hỗ trợ CUDA.\n"
                "Hãy sửa lỗi phần cứng hoặc dùng --device cpu."
            )

    def _apply_vram_constraint(self, limit_gb: float) -> None:
        """
        Giới hạn cứng VRAM được cấp phát cho process này.

        Tỉ lệ = limit_gb / total_vram_gb, được kẹp trong [0.1, 0.95].

        Args:
            limit_gb: Giới hạn VRAM mong muốn (GB).

        Raises:
            HardwareConstraintError: Nếu limit_gb vượt dung lượng GPU thực.
        """
        total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
        total_vram_gb = total_vram_bytes / (1024 ** 3)

        if limit_gb > total_vram_gb:
            raise HardwareConstraintError(
                f"Giới hạn VRAM yêu cầu ({limit_gb:.2f} GB) vượt "
                f"dung lượng GPU thực ({total_vram_gb:.2f} GB)."
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
        Tính tỉ lệ bộ nhớ, được kẹp trong [0.10, 0.95].

        Args:
            limit_gb: Giới hạn mong muốn.
            total_gb: Tổng VRAM thực.

        Returns:
            Số thực trong [0.10, 0.95].
        """
        raw = limit_gb / total_gb
        return max(0.10, min(0.95, raw))

    def _log_gpu_info(self) -> None:
        """Ghi thông tin chẩn đoán GPU vào log."""
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        torch.cuda.memory_reserved(0) / (1024 ** 3)
        torch.cuda.memory_allocated(0) / (1024 ** 3)
        logger.info(
            f"GPU: {props.name} | "
            f"Total VRAM: {total_gb:.2f} GB | "
            f"CUDA Capability: {props.major}.{props.minor}"
        )

    # ── Reset (chỉ dùng khi test) ────────────────────────────────────

    @classmethod
    def _reset(cls) -> None:
        """Reset trạng thái singleton. CHỈ dùng trong unit test."""
        with cls._lock:
            cls._instance = None
            cls._initialized = False
