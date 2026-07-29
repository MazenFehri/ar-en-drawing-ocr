import numpy as np
import pytest
from unittest.mock import patch

from pipeline.ocr import (
    ARABIC_REC_MODEL,
    ARABIC_ROUTING_CONFIDENCE,
    DET_MODEL,
    LATIN_REC_MODEL,
    OcrWord,
    _CROP_PAD_PX,
    _LANG_RECOGNISERS,
    _crop_polygon,
    _pick,
    _poly_bbox,
    run_ocr,
)

# Crops carry a _CROP_PAD_PX margin on all four sides, so a WxH detection yields a
# (W + 2*pad) x (H + 2*pad) crop. Written in terms of the constant rather than baked in,
# so retuning the padding doesn't silently invalidate the geometry these tests pin.
_PAD2 = 2 * _CROP_PAD_PX

# 3.x hands back result objects, not 2.x's [[[pts, (text, conf)]]] nesting: the detector
# reports every polygon on the page in one `dt_polys`, and the recogniser reports one
# `rec_text`/`rec_score` per crop it was given. They are dict-like, so dicts stand in.


def _quad(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


class _FakeDetector:
    def __init__(self, polys):
        self.polys = polys

    def predict(self, image):
        return [{"dt_polys": [np.array(p, dtype=np.float32) for p in self.polys]}]


class _FakeRecogniser:
    """Answers with `readings` in order, and records the size of every batch it saw."""

    def __init__(self, readings):
        self.readings = list(readings)
        self.batch_sizes = []

    def predict(self, crops):
        crops = list(crops)
        self.batch_sizes.append(len(crops))
        out = []
        for _ in crops:
            if not self.readings:
                break
            text, score = self.readings.pop(0)
            out.append({"rec_text": text, "rec_score": score})
        return out


def _wire(polys, latin=(), arabic=()):
    """Patch _get_predictor and return (fakes_by_model, models_requested, getter)."""
    fakes = {
        DET_MODEL: _FakeDetector(polys),
        LATIN_REC_MODEL: _FakeRecogniser(latin),
        ARABIC_REC_MODEL: _FakeRecogniser(arabic),
    }
    requested = []

    def get_predictor(model_name):
        requested.append(model_name)
        return fakes[model_name]

    return fakes, requested, get_predictor


IMAGE = np.ones((200, 300, 3), dtype=np.uint8) * 255

POLYS = [
    _quad(10, 20, 100, 25),
    _quad(120, 20, 80, 25),
    _quad(10, 60, 140, 25),
]


def test_run_ocr_returns_ocrword_list():
    _, _, get = _wire(
        POLYS,
        latin=[("iic ii", 0.31), ("3.5m", 0.88), ("entrnce", 0.48)],
        arabic=[("غرفة النوم", 0.92), ("3.Sm", 0.71), ("ecnrtne", 0.90)],
    )
    with patch("pipeline.ocr._get_predictor", get):
        words = run_ocr(IMAGE)
    assert len(words) == 3
    assert all(isinstance(w, OcrWord) for w in words)


def test_ocrword_fields():
    _, _, get = _wire(
        POLYS,
        latin=[("iic ii", 0.31), ("3.5m", 0.88), ("entrnce", 0.48)],
        arabic=[("غرفة النوم", 0.92), ("3.Sm", 0.71), ("ecnrtne", 0.90)],
    )
    with patch("pipeline.ocr._get_predictor", get):
        words = run_ocr(IMAGE)
    assert words[0].text == "غرفة النوم"
    assert words[0].confidence == pytest.approx(0.92)
    assert words[0].bbox_px["x"] == 10
    assert words[0].bbox_px["w"] == 100


def test_low_confidence_flagged():
    # "entrnce" is the Latin reading kept (the Arabic model returned Latin there, so it
    # loses), and 0.48 is under the caller's threshold.
    _, _, get = _wire(
        POLYS,
        latin=[("iic ii", 0.31), ("3.5m", 0.88), ("entrnce", 0.48)],
        arabic=[("غرفة النوم", 0.92), ("3.Sm", 0.71), ("ecnrtne", 0.90)],
    )
    with patch("pipeline.ocr._get_predictor", get):
        words = run_ocr(IMAGE, confidence_threshold=0.75)
    flagged = [w for w in words if w.flagged]
    assert len(flagged) == 1
    assert flagged[0].text == "entrnce"


def test_empty_result():
    _, _, get = _wire([], latin=[], arabic=[])
    with patch("pipeline.ocr._get_predictor", get):
        assert run_ocr(IMAGE) == []


def test_arabic_survives_the_round_trip():
    # The whole point of the second recogniser: the Latin model cannot emit Arabic
    # script, so without routing this label would come out as its "iic ii" garbage.
    _, _, get = _wire(
        [_quad(10, 20, 100, 25)],
        latin=[("iic ii", 0.31)],
        arabic=[("غرفة النوم", 0.92)],
    )
    with patch("pipeline.ocr._get_predictor", get):
        words = run_ocr(IMAGE)
    assert [w.text for w in words] == ["غرفة النوم"]


# ------------------------------------------------------------------ model routing

def test_en_hint_maps_to_latin_model_not_arabic():
    # Regression guard for the language_hint=en 500 (root cause was a missing runtime
    # download of a model that only "en" needs, not a mapping bug — but the tempting
    # quick "fix" is to silently route "en" through the already-working arabic model,
    # which would mask the bug rather than fix it). It would also be actively wrong:
    # the arabic recogniser reverses and fragments Latin tokens.
    assert _LANG_RECOGNISERS["en"] == (LATIN_REC_MODEL,)
    assert _LANG_RECOGNISERS["ar"] == (ARABIC_REC_MODEL,)
    assert ARABIC_REC_MODEL not in _LANG_RECOGNISERS["en"]


def test_mixed_hint_considers_both_single_script_hints_one():
    assert _LANG_RECOGNISERS["ar+en"] == (LATIN_REC_MODEL, ARABIC_REC_MODEL)
    assert len(_LANG_RECOGNISERS["ar"]) == 1
    assert len(_LANG_RECOGNISERS["en"]) == 1


def test_single_script_hints_never_build_the_other_recogniser():
    for hint, wanted, unwanted in (
        ("en", LATIN_REC_MODEL, ARABIC_REC_MODEL),
        ("ar", ARABIC_REC_MODEL, LATIN_REC_MODEL),
    ):
        _, requested, get = _wire(
            POLYS,
            latin=[("a", 0.1), ("b", 0.1), ("c", 0.1)],
            arabic=[("a", 0.1), ("b", 0.1), ("c", 0.1)],
        )
        with patch("pipeline.ocr._get_predictor", get):
            run_ocr(IMAGE, language_hint=hint)
        # Deliberately low confidences: even so, a single-script hint must skip the
        # routing entirely rather than fall through to the other model.
        assert requested == [DET_MODEL, wanted], hint
        assert unwanted not in requested, hint


def test_detection_runs_once_for_the_mixed_hint():
    # The reason _merge_by_script/_overlap/the union-find are gone: there is one
    # segmentation now, so there is nothing left to reconcile.
    _, requested, get = _wire(
        POLYS,
        latin=[("iic ii", 0.31), ("3.5m", 0.98), ("KITCHEN", 0.99)],
        arabic=[("غرفة النوم", 0.92)],
    )
    with patch("pipeline.ocr._get_predictor", get):
        run_ocr(IMAGE, language_hint="ar+en")
    assert requested.count(DET_MODEL) == 1


def test_only_low_confidence_crops_reach_the_arabic_recogniser():
    # The cost argument for the design. Two of three crops read confidently as Latin, so
    # the Arabic recogniser is handed exactly the one that did not — not the whole page.
    fakes, _, get = _wire(
        POLYS,
        latin=[("iic ii", 0.31), ("3.5m", 0.98), ("KITCHEN", 0.99)],
        arabic=[("غرفة النوم", 0.92)],
    )
    with patch("pipeline.ocr._get_predictor", get):
        words = run_ocr(IMAGE)
    assert fakes[LATIN_REC_MODEL].batch_sizes == [3]
    assert fakes[ARABIC_REC_MODEL].batch_sizes == [1]
    assert sorted(w.text for w in words) == sorted(["غرفة النوم", "3.5m", "KITCHEN"])


def test_arabic_recogniser_is_skipped_entirely_when_nothing_is_doubted():
    fakes, requested, get = _wire(
        POLYS,
        latin=[("LIVING ROOM", 0.98), ("3.5m", 0.98), ("KITCHEN", 0.99)],
    )
    with patch("pipeline.ocr._get_predictor", get):
        run_ocr(IMAGE)
    assert fakes[ARABIC_REC_MODEL].batch_sizes == []
    assert ARABIC_REC_MODEL not in requested


def test_routing_threshold_is_exclusive_at_the_boundary():
    # Exactly at the threshold is "confident enough"; below it is not.
    fakes, _, get = _wire(
        [_quad(10, 20, 100, 25), _quad(10, 60, 100, 25)],
        latin=[("AT", ARABIC_ROUTING_CONFIDENCE),
               ("UNDER", ARABIC_ROUTING_CONFIDENCE - 0.001)],
        arabic=[("الصالة", 0.95)],
    )
    with patch("pipeline.ocr._get_predictor", get):
        words = run_ocr(IMAGE)
    assert fakes[ARABIC_REC_MODEL].batch_sizes == [1]
    assert sorted(w.text for w in words) == sorted(["AT", "الصالة"])


# ---------------------------------------------------------------- per-crop choice

def test_pick_takes_arabic_only_when_it_actually_returned_arabic():
    assert _pick(("iic ii", 0.31), ("غرفة النوم", 0.92)) == ("غرفة النوم", 0.92)


def test_pick_keeps_latin_even_when_arabic_is_more_confident():
    # The arabic model's RTL decoder mangles Latin ("+created_at" -> "at_+created") while
    # reporting ~0.92, so its confidence must not win a numeric contest against a genuine
    # Latin reading. This is the guard that stops the old v4 failure sneaking back in
    # through the routing path.
    assert _pick(("+created_at", 0.62), ("at_+created", 0.92)) == ("+created_at", 0.62)
    assert _pick(("transaction_id", 0.55), ("id_transaction", 0.99))[0] == "transaction_id"


def test_pick_keeps_latin_when_arabic_returns_nothing():
    assert _pick(("entrnce", 0.48), ("", 0.0)) == ("entrnce", 0.48)


def test_empty_readings_are_dropped_not_emitted_as_blank_words():
    _, _, get = _wire(
        [_quad(10, 20, 100, 25), _quad(10, 60, 100, 25)],
        latin=[("", 0.0), ("KITCHEN", 0.99)],
        arabic=[("", 0.0)],
    )
    with patch("pipeline.ocr._get_predictor", get):
        words = run_ocr(IMAGE)
    assert [w.text for w in words] == ["KITCHEN"]


def test_short_recogniser_batch_does_not_misalign_later_words():
    # A recogniser yielding fewer results than crops must not shift every subsequent
    # reading onto the wrong polygon — that would be a silent, page-wide corruption.
    _, _, get = _wire(POLYS, latin=[("FIRST", 0.99)])  # one reading for three crops
    with patch("pipeline.ocr._get_predictor", get):
        words = run_ocr(IMAGE)
    assert [w.text for w in words] == ["FIRST"]
    assert words[0].bbox_px == {"x": 10, "y": 20, "w": 100, "h": 25}


def test_words_come_back_in_page_order():
    _, _, get = _wire(
        [_quad(10, 300, 60, 20), _quad(200, 10, 60, 20), _quad(10, 10, 60, 20)],
        latin=[("LAST", 0.99), ("MIDDLE", 0.99), ("FIRST", 0.99)],
    )
    with patch("pipeline.ocr._get_predictor", get):
        words = run_ocr(np.ones((400, 300, 3), dtype=np.uint8) * 255)
    assert [w.text for w in words] == ["FIRST", "MIDDLE", "LAST"]


# -------------------------------------------------------------------- geometry

def test_poly_bbox_is_the_axis_aligned_hull_of_the_polygon():
    poly = np.array([[10, 20], [110, 24], [110, 49], [10, 45]], dtype=np.float32)
    assert _poly_bbox(poly) == {"x": 10, "y": 20, "w": 100, "h": 29}


def test_crop_of_an_axis_aligned_quad_has_the_quads_dimensions():
    crop = _crop_polygon(IMAGE, np.array(_quad(10, 20, 120, 30), dtype=np.float32))
    assert (crop.shape[1], crop.shape[0]) == (120 + _PAD2, 30 + _PAD2)


def test_crop_of_a_tilted_quad_is_unrotated_not_bbox_cropped():
    # A 100x20 line tilted 30 degrees. Its axis-aligned bbox is ~97x77 — taking that
    # would hand the recogniser a near-square containing the text running diagonally
    # across it plus two wedges of whatever is beside it. The perspective transform must
    # give back the strip's own 100x20 instead.
    import math
    cx, cy, half_w, half_h, ang = 150.0, 100.0, 50.0, 10.0, math.radians(30)
    cos, sin = math.cos(ang), math.sin(ang)
    corners = [
        [cx + dx * cos - dy * sin, cy + dx * sin + dy * cos]
        for dx, dy in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h))
    ]
    poly = np.array(corners, dtype=np.float32)

    # 100*cos30 + 20*sin30 = 96.6 wide, 100*sin30 + 20*cos30 = 67.3 tall: the
    # axis-aligned bbox really is nearly square, and 3.4x the strip's own height.
    bbox = _poly_bbox(poly)
    assert (bbox["w"], bbox["h"]) == (98, 68)

    crop = _crop_polygon(np.ones((300, 300, 3), dtype=np.uint8) * 255, poly)
    assert crop.shape[1] == pytest.approx(100 + _PAD2, abs=2)
    assert crop.shape[0] == pytest.approx(20 + _PAD2, abs=2)
    assert crop.shape[0] < bbox["h"] / 2  # nothing like the bbox crop


def test_vertical_crop_is_stood_upright():
    # A tall, narrow detection is vertical text; the recogniser only reads horizontal
    # strips, so the crop must come back wider than it is tall.
    crop = _crop_polygon(
        np.ones((300, 300, 3), dtype=np.uint8) * 255,
        np.array(_quad(20, 20, 25, 150), dtype=np.float32),
    )
    assert (crop.shape[1], crop.shape[0]) == (150 + _PAD2, 25 + _PAD2)


def test_degenerate_polygons_are_dropped_rather_than_crashing_the_page():
    assert _crop_polygon(IMAGE, np.array([[10, 10]] * 4, dtype=np.float32)) is None
    assert _crop_polygon(IMAGE, np.array([[10, 10], [11, 10]], dtype=np.float32)) is None


def test_a_dropped_crop_does_not_shift_the_remaining_words():
    # The degenerate polygon sits first; the readings must still land on the right boxes.
    _, _, get = _wire(
        [[[10, 10]] * 4, _quad(10, 60, 100, 25)],
        latin=[("KITCHEN", 0.99)],
    )
    with patch("pipeline.ocr._get_predictor", get):
        words = run_ocr(IMAGE)
    assert [w.text for w in words] == ["KITCHEN"]
    assert words[0].bbox_px == {"x": 10, "y": 60, "w": 100, "h": 25}
