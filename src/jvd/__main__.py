"""
CLI entry point for the Junction Violation Detection system.

Usage:
    uv run jvd --device gpu --vram-limit 2.8
    uv run jvd --device cpu
"""

from __future__ import annotations

import argparse

from jvd.core.device import DeviceManager
from jvd.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the JVD system."""
    parser = argparse.ArgumentParser(
        prog="jvd",
        description="Junction Violation Detection - Phát hiện phương tiện đè vạch mắt võng",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Thiết bị xử lý: 'gpu' (yêu cầu CUDA) hoặc 'cpu'. Mặc định: gpu",
    )
    parser.add_argument(
        "--vram-limit",
        type=float,
        default=2.8,
        help="Giới hạn VRAM tối đa (GB). Mặc định: 2.8 GB",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the JVD system."""
    args = parse_args()
    logger = setup_logger("jvd")

    logger.info("=" * 60)
    logger.info("Junction Violation Detection System v0.1.0")
    logger.info("Phát hiện phương tiện đè vạch mắt võng")
    logger.info("=" * 60)

    # ── Phase 1: Device Initialization ──────────────────────────────
    device_manager = DeviceManager.get_instance()
    device_manager.initialize(
        device_flag=args.device,
        vram_limit_gb=args.vram_limit,
    )

    logger.info(f"Device active: {device_manager.device}")
    logger.info("Phase 1 initialization complete. System ready.")


if __name__ == "__main__":
    main()
