"""
ocr.py — Nhận diện biển số đa tiến trình bất đồng bộ.

Layer: inference/

Chạy PaddleOCR trên tiến trình riêng để tránh nghẽn CPU và VRAM với YOLO26.
Dùng regex lọc chuẩn định dạng biển số Việt Nam.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import re

import numpy as np

logger = logging.getLogger(__name__)

# Chuẩn biển số VN regex (VD: 43A-12345, 29H1-12345)
# Cho phép có/không dấu gạch ngang giữa vùng, seri và số.
_LP_REGEX = re.compile(r"^[0-9]{2}\-?[A-Z]{1,2}[0-9]?\-?[0-9]{4,5}$")


class LicensePlateRecognizer:
    """
    Wrapper đa tiến trình cho PaddleOCR.
    Dùng Queue nhận ảnh và Manager.dict lưu kết quả.
    """

    def __init__(self, shared_dict: dict[int, str]) -> None:
        self._shared_dict = shared_dict
        self._queue: mp.Queue = mp.Queue(maxsize=100)
        self._stop_event = mp.Event()
        self._process = mp.Process(
            target=self._run,
            args=(self._queue, self._shared_dict, self._stop_event),
            daemon=True,
        )

    def start(self) -> None:
        """Bắt đầu tiến trình OCR ngầm."""
        self._process.start()
        logger.info("LicensePlateRecognizer process started.")

    def stop(self) -> None:
        """Dừng tiến trình ngầm an toàn."""
        self._stop_event.set()
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()
        logger.info("LicensePlateRecognizer process stopped.")

    def enqueue(self, track_id: int, crop: np.ndarray) -> None:
        """
        Đẩy ảnh crop vào queue OCR.
        Bỏ qua nếu queue đầy để tránh nghẽn pipeline chính.
        """
        try:
            self._queue.put_nowait((track_id, crop))
        except queue.Full:
            pass  # Bỏ qua để duy trì FPS thời gian thực

    @staticmethod
    def _run(q: mp.Queue, results: dict[int, str], stop_event: mp.Event) -> None:
        """Hàm chạy trong tiến trình con."""
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # ── Ngăn lỗi DLL Hell trên Windows ────────────────────────────────────
        # Import torch TRƯỚC paddle để dùng chung OpenMP DLL mới nhất.
        try:
            import torch  # noqa: F401
        except ImportError:
            pass

        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # pragma: no cover
            logger.error(f"PaddleOCR không khả dụng: {exc}")
            return

        # Ép dùng CPU để dành VRAM cho YOLO26.
        try:
            ocr = PaddleOCR(use_angle_cls=False, lang="en")
        except Exception as exc:  # pragma: no cover
            logger.error(f"Khởi tạo PaddleOCR lỗi: {exc}")
            return

        while not stop_event.is_set():
            try:
                track_id, crop = q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                text = LicensePlateRecognizer._process_crop(ocr, crop, track_id)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Lỗi OCR track {track_id}: {exc}")
                continue
            if text is not None:
                results[track_id] = text

    @staticmethod
    def _process_crop(ocr, crop: np.ndarray, track_id: int) -> str | None:
        """Tiền xử lý ảnh và chạy OCR."""
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return None

        w / h

        # Chạy OCR trên crop màu gốc
        text = LicensePlateRecognizer._read_text(ocr, crop)

        # Xóa khoảng trắng/dấu chấm do OCR nhận diện nhầm
        cleaned = text.replace(" ", "").replace(".", "").replace("-", "").upper()

        # Chuẩn VN: 2 số + (1 chữ+1 số HOẶC 1-2 chữ) + 4-5 số
        # VD: 43F161888 -> 43F1-61888
        match = re.match(r"^([0-9]{2}(?:[A-Z][0-9]|[A-Z]{1,2}))([0-9]{4,5})$", cleaned)
        if match:
            formatted = f"{match.group(1)}-{match.group(2)}"
            return formatted

        return None

    @staticmethod
    def _read_text(ocr, img: np.ndarray) -> str:
        """Chạy PaddleOCR và nối các khối text."""
        res = ocr.ocr(img, cls=False)
        if not res or not res[0]:
            return ""

        texts = []
        for line in res[0]:
            if line and len(line) == 2 and isinstance(line[1], tuple):
                texts.append(line[1][0])

        return "".join(texts)
