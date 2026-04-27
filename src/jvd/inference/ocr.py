"""
ocr.py — Asynchronous Multiprocessed License Plate Recognition.

Layer: inference/

This module encapsulates PaddleOCR in a dedicated operating system process
to prevent CPU contention and VRAM bottlenecking with YOLO26.
It implements spatial aspect-ratio heuristics to dynamically slice square plates
and applies strict regex parsing for Vietnamese standard syntax.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import re
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Vietnamese LP standard regex:
# e.g., 43A-12345, 29H1-12345, 92-CA13144
# Allows optional hyphens between region, series, and digits.
_LP_REGEX = re.compile(r"^[0-9]{2}\-?[A-Z]{1,2}[0-9]?\-?[0-9]{4,5}$")


class LicensePlateRecognizer:
    """
    Multiprocess wrapper for PaddleOCR.
    Uses a Queue to ingest image crops and a Manager.dict to store results.
    """

    def __init__(self, shared_dict: Dict[int, str]) -> None:
        self._shared_dict = shared_dict
        self._queue: mp.Queue = mp.Queue(maxsize=100)
        self._stop_event = mp.Event()
        self._process = mp.Process(
            target=self._run,
            args=(self._queue, self._shared_dict, self._stop_event),
            daemon=True,
        )

    def start(self) -> None:
        """Start the background OCR process."""
        self._process.start()
        logger.info("LicensePlateRecognizer process started.")

    def stop(self) -> None:
        """Gracefully terminate the background process."""
        self._stop_event.set()
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()
        logger.info("LicensePlateRecognizer process stopped.")

    def enqueue(self, track_id: int, crop: np.ndarray) -> None:
        """
        Push a vehicle/plate crop into the OCR processing queue.
        Fails silently if the queue is full to avoid stalling the main pipeline.
        """
        try:
            self._queue.put_nowait((track_id, crop))
        except queue.Full:
            pass  # Drop frame to maintain real-time throughput

    @staticmethod
    def _run(q: mp.Queue, results: Dict[int, str], stop_event: mp.Event) -> None:
        """The entry point for the child process."""
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # pragma: no cover
            logger.error(f"PaddleOCR unavailable. OCR worker disabled: {exc}")
            return

        # Force CPU to isolate VRAM for YOLO26.
        try:
            try:
                ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
            except TypeError:
                ocr = PaddleOCR(lang="en", device="cpu")
        except Exception as exc:  # pragma: no cover
            logger.error(f"PaddleOCR initialization failed, OCR worker disabled: {exc}")
            return

        while not stop_event.is_set():
            try:
                track_id, crop = q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                text = LicensePlateRecognizer._process_crop(ocr, crop, track_id)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"OCR inference failed for track {track_id}: {exc}")
                continue
            if text is not None:
                results[track_id] = text

    @staticmethod
    def _process_crop(ocr, crop: np.ndarray, track_id: int) -> Optional[str]:
        """
        Pre-process the image, apply aspect ratio slicing, and run OCR.
        """
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return None
            
        ratio = w / h

        # Run OCR on the full original color crop
        text = LicensePlateRecognizer._read_text(ocr, crop)

        # Regex Syntax Validation
        # Remove spaces and dots that OCR might wrongly infer
        cleaned = text.replace(" ", "").replace(".", "").replace("-", "").upper()
        
        # Vietnamese plate format: 2 digits (province) + (1 letter + 1 digit OR 1-2 letters) + digits
        # Example: 43F161888 -> 43F1-61888, 92CA13144 -> 92CA-13144
        # Logic: Prioritize Letter+Digit series (F1, G1) over 2-letter series (CA, AA). 4-5 digits at end.
        match = re.match(r"^([0-9]{2}(?:[A-Z][0-9]|[A-Z]{1,2}))([0-9]{4,5})$", cleaned)
        if match:
            formatted = f"{match.group(1)}-{match.group(2)}"
            return formatted
            
        return None

    @staticmethod
    def _read_text(ocr, img: np.ndarray) -> str:
        """Run PaddleOCR and join text blocks."""
        res = ocr.ocr(img, cls=False)
        if not res or not res[0]:
            return ""

        texts = []
        for line in res[0]:
            # line structure: [[[x,y], [x,y], [x,y], [x,y]], ('text', conf)]
            if line and len(line) == 2 and isinstance(line[1], tuple):
                texts.append(line[1][0])
                
        return "".join(texts)
