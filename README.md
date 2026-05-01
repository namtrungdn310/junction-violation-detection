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

## Build TensorRT Engine (Bắt Buộc)

> ⚠️ File `.engine` **không được đưa lên Git** vì TensorRT engine chỉ tương thích với GPU đã build nó. Mỗi máy cần tự build engine riêng.

Sau khi `uv sync` xong, chạy lệnh sau để export model YOLO sang TensorRT engine phù hợp với GPU của bạn:

```bash
uv run yolo export model=models/yolo26n.pt format=engine half=True imgsz=1024
```

- Quá trình build mất khoảng **3–5 phút** tuỳ GPU.
- File `models/yolo26n.engine` sẽ được tạo tự động.
- Nếu không có GPU NVIDIA, bỏ qua bước này và chạy pipeline với `--device cpu`.

## Hướng Dẫn Sử Dụng

Khởi chạy hệ thống phân tích video:

```bash
uv run run_pipeline.py --video "data/videos/video_test1.mp4"
```

**Lần chạy đầu tiên**, hệ thống sẽ mở cửa sổ để bạn **vẽ vùng ROI** (vùng mắt võng) trên khung hình đầu tiên:
- **Click trái**: Đặt điểm polygon
- **Click phải**: Xoá điểm cuối
- **`r`**: Reset toàn bộ điểm
- **`c`**: Xác nhận (≥ 3 điểm)
- **`q`**: Huỷ bỏ

Toạ độ ROI sẽ được lưu vào `configs/roi_configs.json`. Các lần chạy tiếp theo với cùng video sẽ **tự động tải lại** ROI đã lưu mà không cần vẽ lại.

**Các tham số chính:**
- `--device cpu`: Ép chạy bằng CPU (nếu không có GPU NVIDIA).
- `--reset-roi`: Vẽ lại vùng ROI (xoá toạ độ cũ, thay bằng toạ độ mới).
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
