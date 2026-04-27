import json
from pathlib import Path

import cv2
import numpy as np


def get_config_path():
    path = Path("configs/roi_configs.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def load_roi_config(video_path: str):
    """Load ROI points for a specific video from config file."""
    config_path = get_config_path()
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            data = json.load(f)
            return data.get(str(Path(video_path).absolute()))
    except:
        return None

def save_roi_config(video_path: str, points: list):
    """Save ROI points for a specific video to config file."""
    config_path = get_config_path()
    data = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = json.load(f)
        except:
            pass

    data[str(Path(video_path).absolute())] = points
    with open(config_path, "w") as f:
        json.dump(data, f, indent=4)

def select_roi_points(video_path: str, current_frame: np.ndarray = None):
    """Interactively select ROI polygon points."""
    if current_frame is None:
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok: return None
    else:
        frame = current_frame

    h, w = frame.shape[:2]
    points = []
    window_name = f"ROI Selector - {Path(video_path).name}"

    display_h = 800
    scale = h / display_h
    display_w = int(w / scale)

    def _mouse_cb(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x * scale), int(y * scale)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, _mouse_cb)

    while True:
        canvas = frame.copy()
        for px, py in points:
            cv2.circle(canvas, (px, py), 4, (0, 255, 255), -1)
        if len(points) >= 2:
            pts = np.array(points, dtype="int32")
            cv2.polylines(canvas, [pts], False, (0, 255, 255), 2)

        status = f"Points: {len(points)} | 'c': confirm | 'r': reset | 'q': cancel"
        cv2.putText(canvas, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        disp = cv2.resize(canvas, (display_w, display_h))
        cv2.imshow(window_name, disp)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key == ord("c"):
            if len(points) >= 3:
                break
        elif key == ord("q"):
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    res = [(x / float(w), y / float(h)) for (x, y) in points]
    save_roi_config(video_path, res)
    return res
