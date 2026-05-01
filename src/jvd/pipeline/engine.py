"""
engine.py — Điều phối toàn bộ pipeline phát hiện vi phạm.

Layer: pipeline/

Điều phối nhận diện, theo dõi, phân tích không gian, OCR đa tiến trình 
và xuất báo cáo bằng chứng trong một vòng lặp duy nhất.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np

from jvd.core.device import DeviceManager
from jvd.inference.detector import ObjectDetector
from jvd.inference.ocr import LicensePlateRecognizer
from jvd.inference.tracker import ByteTrackSession, VehicleTrackerManager
from jvd.pipeline.analyzer import ViolationAnalyzer
from jvd.pipeline.emergency import EmergencyVehicleDetector
from jvd.pipeline.osd import OSDRenderer
from jvd.pipeline.reporter import ViolationReporter
from jvd.utils.stabilizer import VideoStabilizer

logger = logging.getLogger(__name__)


class PipelineEngine:
    """Kết nối tất cả module thành một pipeline xử lý video hoàn chỉnh."""

    def __init__(
        self,
        video_source: str,
        yolo_model: str,
        roi_points: list[tuple[float, float]],
        device_flag: str = "gpu",
        vram_limit_gb: float = 2.8,
        enable_emergency: bool = False,
        export_dir: str = "data/exports",
        display: bool = True,
    ) -> None:
        self.video_source = video_source
        self.display = display
        self.export_dir = export_dir
        self.enable_emergency = enable_emergency

        # 1. Thiết lập thiết bị (Giới hạn VRAM GPU)
        self.dm = DeviceManager()
        self.dm.initialize(device_flag=device_flag, vram_limit_gb=vram_limit_gb)

        # 2. Suy luận & Theo dõi
        self.detector = ObjectDetector(yolo_model, device=self.dm.device)
        self.detector.load()
        self.tracker_session = ByteTrackSession(self.detector)
        self.tracker_mgr = VehicleTrackerManager()

        # 3. OCR (Đa tiến trình)
        self.manager = mp.Manager()
        self.ocr_results = self.manager.dict()
        self.lpr = LicensePlateRecognizer(self.ocr_results)

        # 4. Logic Pipeline
        self.analyzer = ViolationAnalyzer(roi_points)
        self.emergency = EmergencyVehicleDetector() if self.enable_emergency else None
        self.osd = OSDRenderer()
        self.reporter: ViolationReporter | None = None

        # 5. Chống rung
        self.stabilizer = VideoStabilizer(smooth_window=30, crop_pct=0.02)

    def run(self) -> None:
        """Bắt đầu vòng lặp xử lý video."""
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logger.error(f"Không thể mở video: {self.video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.enable_emergency and self.emergency is not None:
            self.emergency.fps = fps

        video_name = Path(self.video_source).stem
        self.reporter = ViolationReporter(
            export_dir=self.export_dir,
            fps=int(fps),
            video_name=video_name
        )

        self.lpr.start()
        self.stabilizer.start()

        if self.display:
            gui_line = next(
                (line for line in cv2.getBuildInformation().splitlines() if "GUI:" in line),
                "GUI: UNKNOWN",
            )
            if "NONE" in gui_line.upper():
                logger.error(
                    "cv2.imshow không khả dụng do OpenCV build thiếu HighGUI. "
                    "Cài lại bằng: uv pip install --python .venv\\Scripts\\python.exe --force-reinstall opencv-contrib-python==4.10.0.84"
                )
                return

        frame_id = 0
        import time
        t_start = time.time()
        current_fps = 0.0

        # Thống kê cho Báo cáo
        unique_vehicles_in_roi: set[int] = set()

        try:
            logger.info(f"Bắt đầu xử lý {self.video_source} ({width}x{height} @ {fps}fps)")

            while True:
                ret, raw_frame = cap.read()
                if not ret:
                    break

                frame_id += 1
                timestamp = frame_id / fps

                # Tính FPS mỗi 10 frame
                if frame_id % 10 == 0:
                    t_now = time.time()
                    current_fps = 10 / (t_now - t_start)
                    t_start = t_now

                # 0. Chống rung Frame
                self.stabilizer.put_frame(raw_frame)
                _, matrix = self.stabilizer.get_latest(timeout=0.01)

                # Dùng frame GỐC cho AI để tránh nhiễu chống rung
                frame = raw_frame

                # A. Core Tracking Loop
                events = self.tracker_session.track(frame, frame_id, timestamp)

                # Làm mượt và cập nhật lịch sử
                events = self.tracker_mgr.update(events, frame_id)

                # B. Bỏ qua xe ưu tiên
                emerg_ids = self.emergency.get_emergency_ids(frame, events) if self.emergency else set()

                # C. Phân tích vi phạm không gian
                violations = self.analyzer.analyze(
                    events=events,
                    tracker=self.tracker_mgr,
                    matrix=matrix,
                    frame_width=width,
                    frame_height=height,
                    emergency_ids=emerg_ids
                )

                # Thống kê số xe đi qua ROI
                for ev in events:
                    tid = ev.track_id
                    if tid is not None:
                        # Chỉ đếm nếu xe đang trong ROI lúc này
                        px, py = ev.bbox.center[0], ev.bbox.y2
                        if matrix is not None:
                            pt = np.array([px, py, 1.0], dtype=np.float32)
                            t_pt = matrix @ pt
                            px, py = t_pt[0], t_pt[1]

                        if self.analyzer.roi.contains((px, py), width, height):
                            # Lọc nhiễu: cần xuất hiện đủ 5 frame
                            if len(self.tracker_mgr.get_history(tid)) >= 5:
                                unique_vehicles_in_roi.add(tid)

                # D. OCR & Trích xuất bằng chứng
                for v in violations:
                    tid = v.event.track_id
                    if tid is None:
                        continue

                    # Cắt ảnh xe cho OCR
                    bbox = v.event.bbox
                    x1, y1, x2, y2 = map(int, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
                    crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]

                    self.lpr.enqueue(int(tid), crop)

                    # Kết quả OCR có thể delay vài frame
                    lp_text = self.ocr_results.get(int(tid), "PENDING")

                    self.reporter.trigger_violation(
                        track_id=int(tid),
                        timestamp=timestamp,
                        lp_text=lp_text,
                        wide_shot=frame,
                        lp_crop=crop
                    )

                # E. Render OSD
                osd_frame = self.osd.draw(
                    frame=frame,
                    events=events,
                    roi=self.analyzer.roi,
                    states=self.analyzer.states,
                    emergency_ids=emerg_ids,
                    ocr_results=self.ocr_results,
                    matrix=matrix,
                    fps=current_fps,
                    frame_id=frame_id,
                    total_frames=total_frames
                )

                # F. Lưu frame vào bộ nhớ tạm bằng chứng
                self.reporter.add_frame(osd_frame, self.ocr_results)

                # G. Dọn dẹp RAM
                if frame_id % 60 == 0:
                    self.tracker_mgr.cleanup_stale(frame_id)

                # H. Giao diện xem trực tiếp
                if self.display:
                    # Resize giữ tỉ lệ, cao 800px
                    h_orig, w_orig = osd_frame.shape[:2]
                    display_h = 800
                    scale = h_orig / display_h
                    display_w = int(w_orig / scale)

                    disp = cv2.resize(osd_frame, (display_w, display_h))

                    cv2.namedWindow("JVD Surveillance", cv2.WINDOW_NORMAL)
                    cv2.imshow("JVD Surveillance", disp)

                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        logger.info("Nhận tín hiệu thoát.")
                        break
                    elif key == ord('r'):
                        logger.info("Chọn lại vùng ROI...")
                        from jvd.utils.roi_helper import select_roi_points
                        new_pts = select_roi_points(self.video_source, current_frame=raw_frame)
                        if new_pts:
                            if matrix is not None:
                                h_img, w_img = raw_frame.shape[:2]
                                transformed_pts = []
                                for nx, ny in new_pts:
                                    # Normalize -> Pixel
                                    px, py = nx * w_img, ny * h_img
                                    # Biến đổi: p' = M * [x, y, 1]^T
                                    pt = np.array([px, py, 1.0], dtype=np.float32)
                                    t_pt = matrix @ pt
                                    # Pixel -> Normalize
                                    transformed_pts.append((t_pt[0] / w_img, t_pt[1] / h_img))
                                new_pts = transformed_pts

                            self.analyzer.roi.update_points(new_pts)
                            logger.info("Cập nhật ROI thành công.")

        except KeyboardInterrupt:
            logger.info("Người dùng ngắt.")
        except Exception as e:
            logger.exception(f"Lỗi pipeline: {e}")
        finally:
            logger.info("Giải phóng tài nguyên...")
            cap.release()
            cv2.destroyAllWindows()
            self.lpr.stop()
            self.stabilizer.stop()

            # --- Báo cáo Tổng kết ---
            print("\n" + "="*50)
            print("         BÁO CÁO TỔNG KẾT VI PHẠM")
            print("="*50)
            print(f" Video: {self.video_source}")
            print(f" Tổng xe đi qua ROI: {len(unique_vehicles_in_roi)}")

            # Lấy danh sách ID vi phạm
            violating_ids = [
                tid for tid, s in self.analyzer.states.items() if s.violation_triggered
            ]

            print(f" Tổng xe vi phạm:    {len(violating_ids)}")
            print("-" * 50)
            if violating_ids:
                print(f" {'ID':<10} | {'Biển số':<20}")
                print("-" * 50)
                for tid in violating_ids:
                    plate = self.ocr_results.get(tid, "NOT_DETECTED")
                    print(f" {tid:<10} | {plate:<20}")
            else:
                print(" Không có vi phạm nào.")
            print("="*50 + "\n")
