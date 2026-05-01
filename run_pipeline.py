#!/usr/bin/env python3
"""
run_pipeline.py — Command-line interface to execute the Junction Violation Detection system.
"""



import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import logging
import sys
from pathlib import Path

from jvd.utils.roi_helper import load_roi_config, select_roi_points

# Ensure log directory exists
log_dir = Path("outputs/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "system.log", mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    # Import lazily so Windows multiprocessing spawn does not load heavy CV stack
    # in child bootstrap processes.
    from jvd.pipeline.engine import PipelineEngine

    parser = argparse.ArgumentParser(description="JVD - Traffic Surveillance System")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, default="models/yolo26n.pt", help="Path to YOLO model (.pt)")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Inference device: 'gpu' (requires CUDA) or 'cpu'",
    )
    parser.add_argument(
        "--vram-limit",
        type=float,
        default=2.8,
        help="Maximum GPU VRAM to reserve in GB (used only when --device gpu)",
    )
    parser.add_argument(
        "--enable-emergency",
        action="store_true",
        help="Enable emergency vehicle exemption logic (disabled by default)",
    )
    parser.add_argument("--export-dir", type=str, default="data/exports", help="Directory to save evidence")
    parser.add_argument("--no-display", action="store_true", help="Disable OSD display window")
    parser.add_argument("--reset-roi", action="store_true", help="Redraw ROI region (discard saved config for this video)")

    args = parser.parse_args()

    # 1. Try to load ROI from config (skip if --reset-roi)
    roi_points = None
    if not args.no_display and not args.reset_roi:
        roi_points = load_roi_config(args.video)
        if roi_points:
            print(f"Loaded existing ROI config for {args.video}")

    # 2. If no config or headless, handle selection
    if roi_points is None:
        if args.no_display:
            roi_points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        else:
            roi_points = select_roi_points(args.video)
            if roi_points is None:
                print("ROI selection cancelled. Exiting.")
                return

    engine = PipelineEngine(
        video_source=args.video,
        yolo_model=args.model,
        roi_points=roi_points,
        device_flag=args.device,
        vram_limit_gb=args.vram_limit,
        enable_emergency=args.enable_emergency,
        export_dir=args.export_dir,
        display=not args.no_display
    )
    engine.run()

if __name__ == "__main__":
    main()
