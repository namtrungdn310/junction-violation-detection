#!/usr/bin/env python3
"""
run_pipeline.py — Command-line interface to execute the Junction Violation Detection system.
"""

import argparse
import sys
from jvd.pipeline.engine import PipelineEngine

def main():
    parser = argparse.ArgumentParser(description="JVD - Traffic Surveillance System")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model (.pt)")
    parser.add_argument("--export-dir", type=str, default="data/exports", help="Directory to save evidence")
    parser.add_argument("--no-display", action="store_true", help="Disable OSD display window")
    
    args = parser.parse_args()
    
    # Default ROI: normalized polygon for the yellow box (adjust as needed per camera)
    # Example: A trapezoid roughly covering the lower center of the screen
    default_roi = [
        (0.3, 0.4),
        (0.7, 0.4),
        (0.9, 0.9),
        (0.1, 0.9)
    ]
    
    engine = PipelineEngine(
        video_source=args.video,
        yolo_model=args.model,
        roi_points=default_roi,
        export_dir=args.export_dir,
        display=not args.no_display
    )
    
    engine.run()

if __name__ == "__main__":
    main()
