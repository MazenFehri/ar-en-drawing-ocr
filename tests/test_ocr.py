import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pipeline.ocr import run_ocr, OcrWord, _LANG_MODELS, _merge_by_script, _overlap

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
    assert _LANG_MODELS["en"] == ("en",)
    assert _LANG_MODELS["ar"] == ("arabic",)


def test_mixed_hint_runs_both_models_single_script_hints_run_one():
    # The arabic model alone mangles Latin (reverses tokens, fragments
    # identifiers) while reporting high confidence, so "ar+en" must run both.
    # "ar"/"en" stay single-model — no second model's worth of latency.
    assert _LANG_MODELS["ar+en"] == ("en", "arabic")
    assert len(_LANG_MODELS["ar"]) == 1
    assert len(_LANG_MODELS["en"]) == 1


# ---------------------------------------------------------------- merge logic

def _word(text, x, y, w=100, h=25, conf=0.9):
    return OcrWord(text=text, confidence=conf, bbox_px={"x": x, "y": y, "w": w, "h": h})


def test_merge_picks_arabic_model_for_arabic_and_en_model_for_latin():
    # Same two regions seen by both models. The arabic label must come from the
    # arabic model; the Latin identifier must come from the en model, whose
    # reading is the correct one ("+created_at", not the reversed "at_+created").
    arabic_model = [_word("غرفة النوم", 10, 20), _word("at_+created", 10, 200)]
    en_model = [_word("iic ii", 10, 20), _word("+created_at", 10, 200)]

    merged = _merge_by_script(arabic_model, en_model)

    by_y = {w.bbox_px["y"]: w.text for w in merged}
    assert by_y[20] == "غرفة النوم"
    assert by_y[200] == "+created_at"
    assert len(merged) == 2


def test_merge_keeps_regions_only_one_model_found():
    arabic_model = [_word("مدخل رئيسي", 10, 20)]
    en_model = [_word("MAIN ENTRANCE", 10, 400)]

    merged = _merge_by_script(arabic_model, en_model)

    assert sorted(w.text for w in merged) == sorted(["مدخل رئيسي", "MAIN ENTRANCE"])


def test_merge_treats_barely_overlapping_boxes_as_separate_regions():
    # 10px of overlap on a 100px box is two neighbouring labels, not one region.
    arabic_model = [_word("KITCHEN", 0, 0, w=100, h=25)]
    en_model = [_word("BEDROOM", 90, 0, w=100, h=25)]

    merged = _merge_by_script(arabic_model, en_model)

    assert len(merged) == 2


def test_merge_resolves_one_line_box_against_several_word_boxes():
    # Measured on sample_drawing.png: the en model returns a whole line as one
    # box while the arabic model splits it into words. Plain IoU never reaches
    # 0.5 on those pairs (0.24-0.47), which used to leave BOTH readings in the
    # output as overlapping duplicate text. The line and its words are one
    # region and must resolve to a single model.
    arabic_model = [
        _word("GROUND", 704, 65, w=145, h=28),
        _word("FLOOR", 862, 62, w=118, h=28),
        _word("PLAN", 985, 63, w=91, h=30),
    ]
    en_model = [_word("GROUND FLOOR PLAN", 704, 63, w=370, h=29)]

    merged = _merge_by_script(arabic_model, en_model)

    assert [w.text for w in merged] == ["GROUND FLOOR PLAN"]


def test_merge_resolves_arabic_line_to_the_arabic_models_word_boxes():
    # Same geometry, Arabic content: the en model's single-box reading of the
    # Arabic title is garbage ('ojY1 gLjI bbu', measured) and must not survive.
    arabic_model = [
        _word("الأرضي", 93, 56, w=104, h=48),
        _word("الطابق", 197, 56, w=78, h=45),
        _word("مخطط", 274, 58, w=81, h=34),
    ]
    en_model = [_word("ojY1 gLjI bbu", 124, 56, w=226, h=39, conf=0.53)]

    merged = _merge_by_script(arabic_model, en_model)

    assert [w.text for w in merged] == ["الأرضي", "الطابق", "مخطط"]


def test_merge_keeps_latin_read_by_only_the_arabic_model():
    # A group with no en member at all must still emit something — the "Latin
    # regions come from the en model" rule has nothing to fall back on here.
    arabic_model = [_word("D-04", 10, 10)]
    en_model = [_word("KITCHEN", 10, 400)]

    merged = _merge_by_script(arabic_model, en_model)

    assert sorted(w.text for w in merged) == ["D-04", "KITCHEN"]


def test_merge_carries_flagged_and_confidence_from_the_chosen_model():
    arabic_model = [OcrWord("entrnce", 0.48, {"x": 0, "y": 0, "w": 100, "h": 25}, flagged=True)]
    en_model = [OcrWord("ENTRANCE", 0.99, {"x": 0, "y": 0, "w": 100, "h": 25}, flagged=False)]

    merged = _merge_by_script(arabic_model, en_model)

    assert len(merged) == 1
    assert merged[0].text == "ENTRANCE"
    assert merged[0].confidence == pytest.approx(0.99)
    assert merged[0].flagged is False


def test_merge_of_empty_inputs():
    assert _merge_by_script([], []) == []
    assert [w.text for w in _merge_by_script([], [_word("ONLY", 0, 0)])] == ["ONLY"]
    assert [w.text for w in _merge_by_script([_word("ONLY", 0, 0)], [])] == ["ONLY"]


def test_overlap_is_relative_to_the_smaller_box():
    box = {"x": 0, "y": 0, "w": 10, "h": 10}
    assert _overlap(box, box) == pytest.approx(1.0)
    assert _overlap(box, {"x": 20, "y": 0, "w": 10, "h": 10}) == 0.0
    # Half-overlapping equal boxes: 50 / 100.
    assert _overlap(box, {"x": 5, "y": 0, "w": 10, "h": 10}) == pytest.approx(0.5)
    # Fully contained scores 1.0 however much bigger the outer box is — this is
    # the case IoU gets wrong (it would score 0.1 here).
    assert _overlap(box, {"x": 0, "y": 0, "w": 100, "h": 10}) == pytest.approx(1.0)


@patch("pipeline.ocr._get_ocr")
def test_mixed_hint_calls_both_models_single_hint_calls_one(mock_get_ocr):
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = [[]]
    mock_get_ocr.return_value = mock_ocr

    run_ocr(np.ones((200, 300, 3), dtype=np.uint8) * 255, language_hint="ar+en")
    assert [c.args[0] for c in mock_get_ocr.call_args_list] == ["en", "arabic"]

    mock_get_ocr.reset_mock()
    run_ocr(np.ones((200, 300, 3), dtype=np.uint8) * 255, language_hint="ar")
    assert [c.args[0] for c in mock_get_ocr.call_args_list] == ["arabic"]

    mock_get_ocr.reset_mock()
    run_ocr(np.ones((200, 300, 3), dtype=np.uint8) * 255, language_hint="en")
    assert [c.args[0] for c in mock_get_ocr.call_args_list] == ["en"]
