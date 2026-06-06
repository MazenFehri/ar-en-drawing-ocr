from dataclasses import dataclass
from typing import Optional
import numpy as np

_ocr_instance = None


@dataclass
class OcrWord:
    text: str
    confidence: float
    bbox_px: dict        # {x, y, w, h}
    flagged: bool = False


def run_ocr(image: np.ndarray, confidence_threshold: float = 0.75) -> list[OcrWord]:
    """Run PaddleOCR on image, return words with bounding boxes and confidence."""
    ocr = _get_ocr()
    raw = ocr.ocr(image, cls=True)
    words = []
    if not raw or not raw[0]:
        return words
    for line in raw[0]:
        points, (text, conf) = line
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = {
            "x": int(min(xs)),
            "y": int(min(ys)),
            "w": int(max(xs) - min(xs)),
            "h": int(max(ys) - min(ys)),
        }
        words.append(OcrWord(
            text=text,
            confidence=conf,
            bbox_px=bbox,
            flagged=conf < confidence_threshold,
        ))
    return words


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        _ocr_instance = PaddleOCR(lang="arabic", use_angle_cls=True, show_log=False)
    return _ocr_instance
