# Junction Violation Detection — Phát Hiện Phương Tiện Đè Vạch Mắt Võng

> Hệ thống AI giám sát không gian hình học đa luồng (Multi-process Spatial AI), phát hiện phương tiện vi phạm đè vạch mắt võng (Yellow Box Junction) tại các nút giao thông Việt Nam theo chuẩn phạt nguội.

## Các Chức Năng Nổi Bật (Phases 1-8)

Toàn bộ hệ thống được lập trình theo thiết kế Hướng Đối Tượng (OOP), phân tách module (MVC) đảm bảo tối đa hóa hiệu suất phần cứng và tính bảo trì cao (LOC < 200/file).

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

Hệ thống hỗ trợ tự động lưu và nạp cấu hình ROI cho từng video. Các video thử nghiệm nên được đặt trong thư mục `data/videos/`.

**Cú pháp khởi chạy Pipeline:**

```bash
uv run run_pipeline.py --video "data/videos/video_test1.mp4"
# Nếu máy không có CUDA/NVIDIA, dùng CPU:
# uv run run_pipeline.py --video "data/videos/video_test1.mp4" --device cpu
```

**Các tham số mở rộng (Optional):**

- `--model`: Trỏ đường dẫn đến tệp trọng số YOLO. Mặc định `models/yolo26n.pt`.
- `--device`: Thiết bị suy luận `gpu` hoặc `cpu`. Mặc định `gpu`.
- `--vram-limit`: Giới hạn VRAM (GB) khi dùng GPU. Mặc định `2.8`.
- `--enable-emergency`: Bật logic miễn trừ xe ưu tiên (mặc định **tắt**).
- `--export-dir`: Thư mục lưu trữ bằng chứng. Mặc định là `data/exports`.
- `--no-display`: Chế độ Headless, tắt cửa sổ hiển thị.

**Tương tác Runtime:**
- Nhấn phím **`q`** để dừng tiến trình ghi hình và kết xuất an toàn hệ thống OCR.

## Quản Lý Dữ Liệu Bằng Chứng (Evidence Package)

Bất kì khi nào một phương tiện dừng trong vạch quá 3 giây (và không bị kẹt xe), module `ViolationReporter` sẽ xuất bằng chứng tại thư mục `data/exports/`:

```
data/exports/
└── video_test1/               # Phân loại theo tên video nguồn
    └── violation_1/           # Thư mục riêng cho mỗi vụ vi phạm
        ├── violation_1.mp4    # Video bằng chứng (15 giây)
        ├── violation_1.json   # Hồ sơ pháp lý (Biển số, tọa độ, timestamp)
        ├── wide_shot.jpg      # Ảnh toàn cảnh lúc bắt đầu đè vạch
        └── license_plate.jpg  # Ảnh cận cảnh biển số đã nhận diện
```

## Kiến Trúc Core

```text
src/jvd/
├── core/        # Quản lý thiết bị GPU và cấu trúc Data Models
├── inference/   # Detector (YOLO), Tracker (ByteTrack), và LPR (PaddleOCR)
├── utils/       # Ổn định hình ảnh (Stabilizer), ROI Helper
└── pipeline/    # Logic chính: Analyzer, Emergency, OSD, Reporter & Engine

Cấu trúc tệp tin dự án:
├── configs/     # Cấu hình ByteTrack và ROI (roi_configs.json)
├── data/
│   ├── exports/ # Kết quả vi phạm (Phân cấp theo Video/Violation_ID)
│   └── videos/  # Thư mục chứa các video đầu vào
├── models/      # Chứa các tệp trọng số AI (.pt, .onnx)
├── outputs/
│   └── logs/    # Lưu trữ Log hệ thống (system.log) để hậu kiểm
├── run_pipeline.py # Script khởi chạy chính
└── pyproject.toml  # Quản lý phụ thuộc bằng UV
```

---
*Dự án được phát triển phục vụ mục đích Nghiên cứu Khoa học (NCKH) về lĩnh vực Giao thông thông minh (ITS).*

## Chạy Bộ Kiểm Thử (Unit Tests)

Dự án này sở hữu độ phủ test khổng lồ (gần 70 bài test tự động) xác nhận các ranh giới toán học và quản lý RAM của từng Phase:

```bash
uv run pytest tests/ -v
```

## License
MIT
