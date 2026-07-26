from dataclasses import dataclass
import numpy as np

# One cached PaddleOCR instance per model language — each is ~100 MB of weights,
# so they are built on first use and kept.
_ocr_instances: dict[str, object] = {}

# The caller's language_hint mapped to a PaddleOCR model. The "arabic" model also
# recognises Latin letters and digits, which is what mixed drawings need.
_LANG_MODELS = {"ar+en": "arabic", "ar": "arabic", "en": "en"}
DEFAULT_LANGUAGE_HINT = "ar+en"


@dataclass
class OcrWord:
    text: str
    confidence: float
    bbox_px: dict        # {x, y, w, h}
    flagged: bool = False


def run_ocr(
    image: np.ndarray,
    confidence_threshold: float = 0.75,
    language_hint: str = DEFAULT_LANGUAGE_HINT,
) -> list[OcrWord]:
    """Run PaddleOCR on image, return words with bounding boxes and confidence."""
    ocr = _get_ocr(language_hint)
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


def _get_ocr(language_hint: str = DEFAULT_LANGUAGE_HINT):
    model_lang = _LANG_MODELS.get(language_hint, "arabic")
    if model_lang not in _ocr_instances:
        from pipeline import _paddle_patch  # noqa: F401  (patches paddle before predictor build)
        from paddleocr import PaddleOCR
        _ocr_instances[model_lang] = PaddleOCR(
            lang=model_lang, use_angle_cls=True, show_log=False
        )
    return _ocr_instances[model_lang]
