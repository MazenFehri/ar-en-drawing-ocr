import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pipeline.ocr import run_ocr, OcrWord, _LANG_MODELS

MOCK_PADDLE_RESULT = [[
    [[[10, 20], [110, 20], [110, 45], [10, 45]], ("غرفة النوم", 0.92)],
    [[[120, 20], [200, 20], [200, 45], [120, 45]], ("3.5m", 0.88)],
    [[[10, 60], [150, 60], [150, 85], [10, 85]], ("entrnce", 0.48)],
]]

@patch("pipeline.ocr._get_ocr")
def test_run_ocr_returns_ocrword_list(mock_get_ocr):
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = MOCK_PADDLE_RESULT
    mock_get_ocr.return_value = mock_ocr
    words = run_ocr(np.ones((200, 300, 3), dtype=np.uint8) * 255)
    assert len(words) == 3
    assert all(isinstance(w, OcrWord) for w in words)

@patch("pipeline.ocr._get_ocr")
def test_ocrword_fields(mock_get_ocr):
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = MOCK_PADDLE_RESULT
    mock_get_ocr.return_value = mock_ocr
    words = run_ocr(np.ones((200, 300, 3), dtype=np.uint8) * 255)
    assert words[0].text == "غرفة النوم"
    assert words[0].confidence == pytest.approx(0.92)
    assert words[0].bbox_px["x"] == 10
    assert words[0].bbox_px["w"] == 100

@patch("pipeline.ocr._get_ocr")
def test_low_confidence_flagged(mock_get_ocr):
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = MOCK_PADDLE_RESULT
    mock_get_ocr.return_value = mock_ocr
    words = run_ocr(np.ones((200, 300, 3), dtype=np.uint8) * 255, confidence_threshold=0.75)
    flagged = [w for w in words if w.flagged]
    assert len(flagged) == 1
    assert flagged[0].text == "entrnce"

@patch("pipeline.ocr._get_ocr")
def test_empty_result(mock_get_ocr):
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = [[]]
    mock_get_ocr.return_value = mock_ocr
    words = run_ocr(np.ones((200, 300, 3), dtype=np.uint8) * 255)
    assert words == []

def test_en_hint_maps_to_english_model_not_arabic():
    # Regression guard for the language_hint=en 500 (root cause was a missing
    # runtime download of a model that only "en" needs, not a mapping bug —
    # but the tempting quick "fix" is to silently route "en" through the
    # already-working arabic model, which would mask the bug rather than fix
    # it). Only "en" should resolve to the dedicated english model.
    assert _LANG_MODELS["en"] == "en"
    assert _LANG_MODELS["ar"] == "arabic"
    assert _LANG_MODELS["ar+en"] == "arabic"
