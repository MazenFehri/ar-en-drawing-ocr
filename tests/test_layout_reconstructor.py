import pytest
from pipeline.layout_reconstructor import (
    reconstruct_layout, to_relative_bbox, classify_page, median_text_height_px,
    FULL_WIDTH_LINE_FRACTION, FULL_WIDTH_LINE_MINIMUM,
)
from pipeline.ocr import OcrWord
from pipeline.shape_detector import ShapeResult
from models.elements import BBox, TextElement, SimpleShapeElement, ComplexShapeElement

IMG_W, IMG_H = 1000, 1500


def make_word(text, x, y, w=100, h=20, conf=0.9, flagged=False):
    return OcrWord(text=text, confidence=conf, bbox_px={"x": x, "y": y, "w": w, "h": h}, flagged=flagged)


def test_to_relative_bbox():
    bbox = to_relative_bbox({"x": 100, "y": 150, "w": 200, "h": 50}, IMG_W, IMG_H)
    assert isinstance(bbox, BBox)
    assert bbox.x == pytest.approx(0.1)
    assert bbox.y == pytest.approx(0.1)
    assert bbox.w == pytest.approx(0.2)
    assert bbox.h == pytest.approx(50 / 1500)


def test_arabic_line_reads_right_to_left():
    """On one line, "غرفة النوم" has غرفة rightmost — it must come out first.

    Input is what the recogniser hands back, which under arabic_PP-OCRv5_mobile_rec is
    logical (Unicode storage) order. Only the *order of the boxes along the line* is this
    function's business; the strings themselves pass through untouched.
    """
    words = [
        make_word("النوم", 100, 50, 80, 24),   # left on the page
        make_word("غرفة", 200, 52, 70, 24),    # right on the page
    ]
    ordered = reconstruct_layout(words, [], IMG_W, IMG_H)
    assert [el.content for el in ordered] == ["غرفة", "النوم"]


def test_arabic_text_is_not_reordered_on_the_way_through():
    """The double-correction guard, asserted by codepoint.

    Under PP-OCRv4 this layer reversed every Arabic string, because that model emitted
    visual order. The v5 arabic recogniser emits logical order, so the reversal had to go
    — and if it ever comes back, the damage is invisible in a rendered diff (reversed
    Arabic still renders as Arabic, still passes is_arabic and detect_language). Compare
    codepoints, not glyphs.
    """
    phrase = "مخطط الطابق الأرضي"
    assert [ord(c) for c in phrase][:4] == [0x0645, 0x062E, 0x0637, 0x0637]  # م خ ط ط

    ordered = reconstruct_layout([make_word(phrase, 100, 50, 300, 24)], [], IMG_W, IMG_H)

    assert [ord(c) for c in ordered[0].content] == [ord(c) for c in phrase]
    assert ordered[0].content != phrase[::-1]
    assert ordered[0].language == "arabic"


def test_english_line_reads_left_to_right():
    words = [
        make_word("ROOM", 300, 50, 80, 24),
        make_word("LIVING", 200, 52, 90, 24),
    ]
    ordered = reconstruct_layout(words, [], IMG_W, IMG_H)
    assert [el.content for el in ordered] == ["LIVING", "ROOM"]


def test_separate_lines_stay_top_to_bottom():
    words = [
        make_word("SECOND", 400, 300, 90, 24),
        make_word("FIRST", 100, 50, 80, 24),
    ]
    ordered = reconstruct_layout(words, [], IMG_W, IMG_H)
    assert [el.content for el in ordered] == ["FIRST", "SECOND"]


def test_reading_order_top_to_bottom():
    words = [
        make_word("bottom", 10, 200),
        make_word("top", 10, 50),
        make_word("middle", 10, 120),
    ]
    elements = reconstruct_layout(words, [], IMG_W, IMG_H)
    texts = [e.content for e in elements if isinstance(e, TextElement)]
    assert texts == ["top", "middle", "bottom"]


def test_elements_have_relative_bbox():
    words = [make_word("test", 100, 200, 300, 50)]
    elements = reconstruct_layout(words, [], IMG_W, IMG_H)
    assert isinstance(elements[0].bbox, BBox)
    assert 0.0 <= elements[0].bbox.x <= 1.0
    assert 0.0 <= elements[0].bbox.y <= 1.0


def test_simple_shape_in_output():
    shape = ShapeResult(shape_type="circle", bbox_px={"x": 50, "y": 50, "w": 80, "h": 80}, confidence=0.97)
    elements = reconstruct_layout([], [shape], IMG_W, IMG_H)
    assert len(elements) == 1
    assert isinstance(elements[0], SimpleShapeElement)


def test_complex_shape_in_output():
    shape = ShapeResult(shape_type="complex", bbox_px={"x": 50, "y": 50, "w": 80, "h": 80}, confidence=0.7, crop=None)
    elements = reconstruct_layout([], [shape], IMG_W, IMG_H)
    assert len(elements) == 1
    assert isinstance(elements[0], ComplexShapeElement)


def test_language_detection_arabic():
    words = [make_word("غرفة النوم", 10, 50)]
    elements = reconstruct_layout(words, [], IMG_W, IMG_H)
    assert elements[0].language == "arabic"


def test_language_detection_english():
    words = [make_word("entrance", 10, 50)]
    elements = reconstruct_layout(words, [], IMG_W, IMG_H)
    assert elements[0].language == "english"


# --- container_shape_id: shape-membership association (Img2UML-style) ------

def test_word_inside_shape_gets_container_id():
    word = make_word("Bank", 60, 60, 40, 20)
    box = ShapeResult(shape_type="complex", bbox_px={"x": 50, "y": 50, "w": 300, "h": 300}, confidence=0.7)
    elements = reconstruct_layout([word], [box], IMG_W, IMG_H)
    text_el = next(e for e in elements if isinstance(e, TextElement))
    assert text_el.container_shape_id == "shape_000"


def test_word_outside_every_shape_has_no_container():
    """A relationship/cardinality label floating between two class boxes on a
    diagram — the common case that caps how much shape-context can help."""
    word = make_word("0..1", 900, 900, 40, 20)
    box = ShapeResult(shape_type="complex", bbox_px={"x": 50, "y": 50, "w": 300, "h": 300}, confidence=0.7)
    elements = reconstruct_layout([word], [box], IMG_W, IMG_H)
    text_el = next(e for e in elements if isinstance(e, TextElement))
    assert text_el.container_shape_id is None


def test_word_in_nested_shapes_picks_innermost():
    word = make_word("+id", 110, 110, 20, 10)
    outer = ShapeResult(shape_type="complex", bbox_px={"x": 50, "y": 50, "w": 400, "h": 400}, confidence=0.7)
    inner = ShapeResult(shape_type="complex", bbox_px={"x": 100, "y": 100, "w": 60, "h": 40}, confidence=0.7)
    # outer listed first: selection must be by area, not by list order
    elements = reconstruct_layout([word], [outer, inner], IMG_W, IMG_H)
    text_el = next(e for e in elements if isinstance(e, TextElement))
    assert text_el.container_shape_id == "shape_001"  # the smaller, inner box


def test_page_of_short_labels_is_a_drawing():
    """class-diagram.png shape: 0 lines wider than 40% of the page."""
    words = [make_word(f"+field_{i}", 100, 50 + i * 40, w=120, h=27) for i in range(20)]
    assert classify_page(words, IMG_W) == "drawing"


def test_page_of_full_width_lines_is_text():
    """The worksheets' shape: 7 lines spanning most of the page width."""
    words = [make_word(f"line {i}", 60, 50 + i * 40, w=800, h=15) for i in range(7)]
    assert classify_page(words, IMG_W) == "text"


def test_a_couple_of_wide_lines_is_still_a_drawing():
    """A drawing's title block is wide but there is only one of it — the minimum
    line count, not the width alone, is what does the separating."""
    wide = [make_word("a wide title block", 60, 50, w=800, h=27)
            for _ in range(FULL_WIDTH_LINE_MINIMUM - 1)]
    narrow = [make_word("label", 100, 200 + i * 40, w=120, h=27) for i in range(15)]
    assert classify_page(wide + narrow, IMG_W) == "drawing"


def test_classify_page_on_no_words_is_a_drawing():
    assert classify_page([], IMG_W) == "drawing"


def test_classify_page_boundary_is_strictly_wider_than_the_fraction():
    at_threshold = [make_word("x", 0, i * 40, w=int(IMG_W * FULL_WIDTH_LINE_FRACTION), h=15)
                    for i in range(10)]
    assert classify_page(at_threshold, IMG_W) == "drawing"
    over = [make_word("x", 0, i * 40, w=int(IMG_W * FULL_WIDTH_LINE_FRACTION) + 5, h=15)
            for i in range(10)]
    assert classify_page(over, IMG_W) == "text"


def test_median_text_height_of_no_words_is_zero():
    assert median_text_height_px([]) == 0.0


def test_median_text_height_is_the_median_not_the_mean():
    words = [make_word("a", 0, 0, h=14), make_word("b", 0, 40, h=15),
             make_word("c", 0, 80, h=200)]
    assert median_text_height_px(words) == 15.0


def test_word_partially_overlapping_shape_is_not_contained():
    """Strict containment only — a word straddling a box edge doesn't count,
    same rule the Phase 1 measurement used."""
    word = make_word("edge", 280, 280, 60, 40)
    box = ShapeResult(shape_type="rect", bbox_px={"x": 50, "y": 50, "w": 250, "h": 250}, confidence=0.9)
    elements = reconstruct_layout([word], [box], IMG_W, IMG_H)
    text_el = next(e for e in elements if isinstance(e, TextElement))
    assert text_el.container_shape_id is None
