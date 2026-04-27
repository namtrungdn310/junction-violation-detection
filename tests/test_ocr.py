"""
Unit tests for Phase 7: LicensePlateRecognizer (Multiprocessed OCR & Regex).
"""

from __future__ import annotations

import multiprocessing as mp
import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jvd.inference.ocr import _LP_REGEX, LicensePlateRecognizer

def test_vietnamese_lp_regex():
    # Valid 1-line plates
    assert _LP_REGEX.match("43A-12345")
    assert _LP_REGEX.match("51C-1234")
    assert _LP_REGEX.match("29A12345")  # Missing hyphen is allowed
    
    # Valid 2-line plates (joined by code)
    assert _LP_REGEX.match("43H1-12345")
    assert _LP_REGEX.match("29AA-1234")
    
    # Invalid plates
    assert not _LP_REGEX.match("ABC-1234")   # Starts with letters
    assert not _LP_REGEX.match("43A-123")    # Too few digits
    assert not _LP_REGEX.match("43A-123456") # Too many digits
    assert not _LP_REGEX.match("43-12345")   # Missing letters


class TestLicensePlateRecognizer:
    def setup_method(self):
        self.manager = mp.Manager()
        self.shared_dict = self.manager.dict()

    def teardown_method(self):
        # We don't want leaked processes
        pass

    def test_aspect_ratio_heuristics(self):
        mock_ocr = MagicMock()
        
        # Structure: [ [ [box, ('text', conf)] ] ]
        mock_ocr.ocr.return_value = [
            [ [ [[0,0]], ('43A12345', 0.99) ] ]
        ]
        
        # 1-line plate: ratio 4.0 (W=40, H=10)
        crop_1line = np.zeros((10, 40, 3), dtype=np.uint8)
        text = LicensePlateRecognizer._process_crop(mock_ocr, crop_1line)
        assert text == "43A12345"
        
        # 2-line plate: ratio 1.0 (W=20, H=20)
        crop_2line = np.zeros((20, 20, 3), dtype=np.uint8)
        mock_ocr.ocr.side_effect = [
            [ [ [ [[0,0]], ('29H1', 0.99) ] ] ],  # top
            [ [ [ [[0,0]], ('12345', 0.99) ] ] ]   # bottom
        ]
        text2 = LicensePlateRecognizer._process_crop(mock_ocr, crop_2line)
        assert text2 == "29H1-12345"

        # Regex invalidates bad OCR
        mock_ocr.ocr.side_effect = [
            [ [ [ [[0,0]], ('GARBAGE', 0.99) ] ] ]
        ]
        text3 = LicensePlateRecognizer._process_crop(mock_ocr, crop_1line)
        assert text3 is None

    @patch('jvd.inference.ocr.PaddleOCR')
    def test_process_lifecycle(self, mock_paddleocr_cls):
        # We test _run synchronously to allow mocks to work (multiprocessing spawn ignores mocks)
        mock_ocr_instance = MagicMock()
        mock_ocr_instance.ocr.return_value = [ [ [ [[0,0]], ('43A12345', 0.99) ] ] ]
        mock_paddleocr_cls.return_value = mock_ocr_instance
        
        q = mp.Queue()
        stop_event = mp.Event()
        
        # Enqueue job
        crop = np.zeros((10, 40, 3), dtype=np.uint8)
        q.put((99, crop))
        
        import threading
        # Start a thread to stop the _run loop after a short delay
        def stopper():
            import time
            time.sleep(0.5)
            stop_event.set()
            
        threading.Thread(target=stopper).start()
        
        LicensePlateRecognizer._run(q, self.shared_dict, stop_event)
        
        assert 99 in self.shared_dict
        assert self.shared_dict[99] == "43A12345"

