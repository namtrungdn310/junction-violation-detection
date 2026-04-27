# Junction Violation Detection — Phát Hiện Phương Tiện Đè Vạch Mắt Võng

> Hệ thống AI phát hiện phương tiện vi phạm vùng vạch mắt võng (Yellow Box Junction) tại các nút giao thông Việt Nam.

## Yêu Cầu Hệ Thống

- Python ≥ 3.10
- NVIDIA GPU with CUDA support (recommended)
- [uv](https://github.com/astral-sh/uv) package manager

## Cài Đặt

```bash
# Clone repository
git clone <repo-url>
cd junction-violation-detection

# Tạo môi trường ảo & cài đặt dependencies
uv sync

# Chạy hệ thống
uv run jvd --device gpu --vram-limit 2.8
uv run jvd --device cpu
```

## Chạy Tests

```bash
uv run pytest tests/ -v
```

## Kiến Trúc

```
src/jvd/
├── core/        # Tầng lõi: Device, Exceptions, Data Models
├── inference/   # Tầng suy luận: YOLO26, PaddleOCR
├── utils/       # Tầng tiện ích: Logging, Video I/O
└── pipeline/    # Tầng đường ống: Orchestration
```

## License

MIT
