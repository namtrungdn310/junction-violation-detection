# Junction Violation Detection — Phát Hiện Phương Tiện Đè Vạch Mắt Võng

> Hệ thống AI giám sát không gian hình học đa luồng (Multi-process Spatial AI), phát hiện phương tiện vi phạm đè vạch mắt võng (Yellow Box Junction) tại các nút giao thông Việt Nam theo chuẩn phạt nguội.

## Các Chức Năng Nổi Bật (Phases 1-8)

Toàn bộ hệ thống được lập trình theo thiết kế Hướng Đối Tượng (OOP), phân tách module (MVC) đảm bảo tối đa hóa hiệu suất phần cứng (RAM < 3GB) và tính bảo trì cao (LOC < 200/file).

- **`VideoStabilizer` (Đa Luồng CPU)**: Chống rung video quang học dùng SIFT & ORB.
- **`ObjectDetector` (YOLO26 & TensorRT)**: Tối ưu VRAM bằng cơ chế cascade fallback (Engine -> ONNX).
- **`VehicleTracker` (ByteTrack)**: Giải quyết sự cố che khuất (Occlusion) bằng Kalman Filter.
- **`ViolationAnalyzer`**: Logic không gian bằng Ray Casting (Point-in-Polygon), miễn trừ xe bị ùn tắc dựa trên Horizontal IoU và Forward Collision logic.
- **`EmergencyDetector`**: Phân tích quang học sóng nhấp nháy đèn khẩn cấp 1-4Hz trên không gian màu HSV để nhận diện xe ưu tiên.
- **`LicensePlateRecognizer` (Đa Luồng Độc Lập)**: Chạy OCR Paddle tách biệt hoàn toàn GPU, cắt ảnh biển số tự động theo tỷ lệ Aspect Ratio, lọc bằng Biểu Thức Chính Quy (Regex) chuẩn Việt Nam.
- **`OSDRenderer`**: Alpha Blending để tô nền giao diện trực quan trực tiếp.
- **`ViolationReporter`**: Ghi hình vòng lặp tự động đóng gói hồ sơ pháp lý (MP4 + Hình Cắt + JSON).

## Yêu Cầu Hệ Thống

- **Python** ≥ 3.10
- **NVIDIA GPU** với cấu trúc CUDA (vẫn có thể chạy trên CPU thuần nhưng chậm hơn).
- **uv**: Trình quản lý thư viện thay thế pip.

## Cài Đặt

Mở Terminal tại thư mục này và chạy các lệnh sau để khởi tạo:

```bash
# Thiết lập môi trường ảo và tự động cài đặt mọi dependencies
uv sync

# Kích hoạt môi trường (tuỳ chọn nếu dùng trực tiếp `uv run`)
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
```

## Hướng Dẫn Kiểm Thử Bằng Video Thực Tế

Bạn có thể đưa các video thử nghiệm của mình vào cùng thư mục dự án (ví dụ `test1.mp4`).

**Cú pháp khởi chạy Pipeline:**

```bash
uv run run_pipeline.py --video "ten_video_cua_ban.mp4"
```

**Các tham số mở rộng (Optional):**

- `--model`: Trỏ đường dẫn đến tệp trọng số YOLO. Mặc định `yolov8n.pt` (Sẽ tự tải nếu không có).
- `--export-dir`: Thư mục lưu trữ bằng chứng. Mặc định là `data/exports`.
- `--no-display`: Thêm cờ này nếu chạy trên server không có màn hình (Headless), tắt luồng `cv2.imshow`.

**Tương tác Runtime:**
- Nhấn phím **`q`** để dừng tiến trình ghi hình và kết xuất an toàn hệ thống OCR.

## Quản Lý Dữ Liệu Bằng Chứng (Evidence Package)

Bất kì khi nào một phương tiện dừng trong vạch quá 3 giây (Và không bị kẹt xe/không là xe ưu tiên), module `ViolationReporter` sẽ xuất bằng chứng tại thư mục `data/exports/`:

```
data/exports/
├── violation_1_1684345.mp4    # Video 10 giây (5 giây trước, 5 giây sau)
├── violation_1_1684345_wide.jpg # Ảnh toàn cảnh khi bắt đầu đè vạch
├── violation_1_1684345_crop.jpg # Ảnh cắt cận cảnh vào phương tiện/biển số
└── violation_1_1684345.json   # JSON hồ sơ vi phạm (Camera ID, Timestamp, Biển Số...)
```

## Kiến Trúc Core

```text
src/jvd/
├── core/        # Quản lý thiết bị GPU cứng và cấu trúc Data Models
├── inference/   # Biên dịch TensorRT, Tracker (ByteTrack), và LPR (PaddleOCR)
├── utils/       # Ổn định hình ảnh, Camera Logging
└── pipeline/    # Giao điểm thuật toán: Analyzer, Emergency, OSD, Reporter & Engine
```

## Chạy Bộ Kiểm Thử (Unit Tests)

Dự án này sở hữu độ phủ test khổng lồ (gần 70 bài test tự động) xác nhận các ranh giới toán học và quản lý RAM của từng Phase:

```bash
uv run pytest tests/ -v
```

## License
MIT
