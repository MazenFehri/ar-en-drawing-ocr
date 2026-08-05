import pytest
from pipeline.layout_reconstructor import reconstruct_layout, to_relative_bbox
from pipeline.ocr import OcrWord
from pipeline.shape_detector import ShapeResult
from models.elements import BBox, TextElement, SimpleShapeElement, ComplexShapeElement

IMG_W, IMG_H = 1000, 1500


def make_word(text, x, y, w=100, h=20, conf=0.9, flagged=False):
    return OcrWord(text=text, confidence=conf, bbox_px={"x": x, "y": y, "w": w, "h": h}, flagged=flagged)


def test_a_marking_column_is_lifted_out_of_the_sentences():
    """Geometry from test1.jpeg, scaled to this page: cells at 87.5% of the width, each
    3.2% wide, each sharing its vertical centre with a body line beside it. That overlap is
    the bug — without the split they sort *into* the sentences.
    """
    body = [
        make_word("السند 1 للاحتفال بعيد ميلاد", 90, 200, w=740),
        make_word("أحسب المبلغ الذي أحضرته رانية", 420, 400, w=410),
        make_word("أحسب نقود أحمد", 600, 600, w=230),
    ]
    cells = [make_word("د", 875, y, w=32) for y in (190, 390, 590, 790)]

    elements = reconstruct_layout(body + cells, [], IMG_W, IMG_H)

    assert [e.content for e in elements[:3]] == [w.text for w in body], "prose stays contiguous"
    assert [e.content for e in elements[3:]] == ["د"] * 4, "cells follow, top to bottom"
    assert [e.margin_column for e in elements] == [False] * 3 + [True] * 4
    # Lifted out of the reading order, not out of the document.
    assert elements[3].bbox.x == pytest.approx(0.875)


def test_narrow_marks_in_the_middle_of_the_page_are_left_in_place():
    """The same worksheet's coin tokens are exactly as narrow as the marking cells.

    Width cannot be what separates them, which is why the edge band and the alignment
    tolerance carry the decision.
    """
    words = [make_word("أواصل تمثيل هذا المبلغ", 300, 400, w=400)] + [
        make_word(text, x, 400, w=32) for text, x in (("30", 460), ("2", 540), ("½", 620))
    ]
    elements = reconstruct_layout(words, [], IMG_W, IMG_H)
    assert not any(e.margin_column for e in elements)


def test_a_table_row_label_column_is_not_a_margin():
    """Regression, found by running this on the real pages rather than on fixtures.

    test3's S1/S2/S3 row labels satisfy every condition but the one that matters — narrow,
    hard against the left edge, aligned to the pixel, three of them. What makes them body
    text is that they share their horizontal span with the prose. An earlier version of this
    rule swallowed them, and 19 of class-diagram's UML attributes with them.
    """
    words = [make_word("Find initial basic feasible solution", 10, 500, w=880)] + [
        make_word(text, 20, y, w=35) for text, y in (("S1", 100), ("S2", 200), ("S3", 300))
    ]
    elements = reconstruct_layout(words, [], IMG_W, IMG_H)
    assert not any(e.margin_column for e in elements)


def test_two_cells_at_an_edge_are_not_a_column():
    words = [
        make_word("north elevation", 100, 100, w=600),
        make_word("A", 950, 200, w=30),
        make_word("B", 950, 300, w=30),
    ]
    elements = reconstruct_layout(words, [], IMG_W, IMG_H)
    assert not any(e.margin_column for e in elements)


def test_edge_labels_that_do_not_share_an_x_are_not_a_column():
    """A drawing may carry several small labels near a border. A printed marking column
    shares an x to within a pixel or two; scattered annotations do not."""
    words = [make_word("floor plan", 100, 100, w=600)] + [
        make_word(text, x, y, w=15)
        for text, x, y in (("A", 870, 200), ("B", 940, 400), ("C", 985, 600))
    ]
    elements = reconstruct_layout(words, [], IMG_W, IMG_H)
    assert not any(e.margin_column for e in elements)


def test_a_numeric_disagreement_ships_marked_however_confident_the_line_is():
    """The confidence score cannot carry this signal, so the highlight must not depend on it.

    The Arabic recogniser reports a healthy score on a line it silently deleted a number
    from — that score is its opinion of the Arabic, which it read correctly. A page whose
    only defect is a missing amount would otherwise look clean at every threshold.
    """
    word = make_word("مبلغ مي الواحدة", 10, 10, conf=0.97)
    word.digit_disagreement = True
    word.digits_recovered = []

    element = reconstruct_layout([word], [], IMG_W, IMG_H)[0]
    assert element.digit_disagreement is True
    assert element.highlight == "red"


def test_a_clean_line_is_not_marked():
    element = reconstruct_layout([make_word("غرفة النوم", 10, 10, conf=0.61)], [], IMG_W, IMG_H)[0]
    assert element.digit_disagreement is False
    assert element.digits_recovered == []
    # Low confidence alone is the LLM stage's business, not this one's.
    assert element.highlight is None


def test_a_recovered_number_travels_to_the_element_without_marking_it():
    """Recovery is the fix working — it is reported, but it is not a warning."""
    word = make_word("فاحضر مجدي 27250 مي", 10, 10)
    word.digits_recovered = ["27250"]

    element = reconstruct_layout([word], [], IMG_W, IMG_H)[0]
    assert element.digits_recovered == ["27250"]
    assert element.highlight is None


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


def test_word_partially_overlapping_shape_is_not_contained():
    """Strict containment only — a word straddling a box edge doesn't count,
    same rule the Phase 1 measurement used."""
    word = make_word("edge", 280, 280, 60, 40)
    box = ShapeResult(shape_type="rect", bbox_px={"x": 50, "y": 50, "w": 250, "h": 250}, confidence=0.9)
    elements = reconstruct_layout([word], [box], IMG_W, IMG_H)
    text_el = next(e for e in elements if isinstance(e, TextElement))
    assert text_el.container_shape_id is None


def test_median_text_height_is_the_median_not_the_mean():
    from pipeline.layout_reconstructor import median_text_height_px
    # One huge outlier: a mean would be dragged over the low-resolution threshold,
    # a median is not. That is the whole reason this is a median.
    words = [make_word("a", 0, 0, w=50, h=14) for _ in range(5)]
    words.append(make_word("big", 0, 200, w=50, h=400))
    assert median_text_height_px(words) == 14.0


def test_median_text_height_of_no_words_is_zero():
    from pipeline.layout_reconstructor import median_text_height_px
    assert median_text_height_px([]) == 0.0
