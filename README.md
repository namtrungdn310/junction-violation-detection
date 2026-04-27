# Junction Violation Detection (JVD)

Hệ thống AI giám sát không gian hình học đa luồng, chuyên phát hiện phương tiện đè vạch mắt võng (Yellow Box Junction) phục vụ hệ thống phạt nguội tại Việt Nam.

## Tính Năng & Kiến Trúc Cốt Lõi

Hệ thống vận hành theo kiến trúc **bất đồng bộ (Asynchronous)**, phân tách tác vụ nhằm tối đa hóa hiệu suất phần cứng (không rớt FPS):

- **YOLO26 & TensorRT (GPU):** Nhận diện phương tiện thời gian thực ở tốc độ cao. Hỗ trợ đầy đủ kiến trúc GPU hiện đại (RTX 50-series Blackwell, Ampere).
- **ByteTrack (CPU):** Bám sát đối tượng (Tracking) qua bộ lọc Kalman.
- **Phân tích Không gian (CPU):** Ray Casting & Horizontal IoU để xác định lỗi đè vạch >3s và loại trừ trường hợp kẹt xe.
- **PaddleOCR (CPU - Luồng Độc Lập):** Trích xuất biển số xe chạy ngầm. Không gây độ trễ (delay) cho camera.
- **Đóng Gói Bằng Chứng:** Tự động cắt video (15s), trích xuất ảnh góc rộng, ảnh cận biển số và hồ sơ pháp lý JSON.

## Cài Đặt Nhanh

Yêu cầu: `Python >= 3.10` và thư viện quản lý gói `uv`.

```bash
# Cài đặt tự động toàn bộ môi trường (PyTorch cu128, PaddleOCR, TensorRT, v.v.)
uv sync
```

## Hướng Dẫn Sử Dụng

Khởi chạy hệ thống phân tích video:

```bash
uv run run_pipeline.py --video "data/videos/video_test1.mp4"
```

**Các tham số chính:**
- `--device cpu`: Ép chạy bằng CPU (nếu không có GPU NVIDIA).
- `--export-dir`: Thư mục lưu bằng chứng (Mặc định: `data/exports`).

*Lưu ý: Bấm phím **`q`** trên cửa sổ video để dừng an toàn và kết xuất hồ sơ.*

## Hồ Sơ Vi Phạm (Evidence Package)

Sau khi chạy xong, dữ liệu phạt nguội tự động lưu tại `data/exports/`:

```text
data/exports/video_test1/violation_1/
├── violation_1.mp4    # Video cắt 15 giây
├── violation_1.json   # Hồ sơ pháp lý (Biển số: 92CA-13144, Tọa độ, Timestamp)
├── wide_shot.jpg      # Ảnh toàn cảnh
└── license_plate.jpg  # Ảnh cận cảnh biển số
```

## Kiểm Thử (Unit Tests)

```bash
uv run pytest tests/ -v
```

---
*Dự án NCKH về lĩnh vực Giao thông Thông minh (ITS).*
*License: MIT*
