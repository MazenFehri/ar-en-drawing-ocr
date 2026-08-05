import json

from models.elements import (
    BBox, TextElement, SimpleShapeElement, PolylineShapeElement, ComplexShapeElement,
    LLMCorrection,
)
from pipeline.sidecar import build_sidecar


def test_sidecar_carries_quality_score():
    side = build_sidecar([], 100, 200, 5, quality_score=0.42)
    assert side["stats"]["quality_score"] == 0.42


def make_text(id_, text, conf, corrected=None, highlight=None):
    correction = LLMCorrection(original=text, corrected=corrected, certainty=0.95) if corrected else None
    return TextElement(
        id=id_, bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.03),
        content=corrected or text, language="english",
        confidence=conf, llm_correction=correction, highlight=highlight
    )


def test_sidecar_has_required_keys():
    result = build_sidecar([], 1000, 1500, 350)
    assert "elements" in result
    assert "stats" in result
    assert "page_dimensions" in result


# --- review queue ----------------------------------------------------------------------

def _reviewable(id_, conf, highlight="red", disagreement=False, margin=False):
    return TextElement(
        id=id_, bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.03), content="x", language="english",
        confidence=conf, highlight=highlight,
        digit_disagreement=disagreement, margin_column=margin,
    )


def test_review_queue_is_empty_when_nothing_needs_review():
    side = build_sidecar([make_text("t0", "clean", 0.99)], 100, 200, 5)
    assert side["stats"]["review_queue"] == []


def test_review_queue_puts_numeric_disagreement_first_however_confident():
    """A wrong number makes a maths worksheet wrong; a wrong letter makes it ugly.

    The disagreeing line here is the *most* confident of the three, so ordering by confidence
    alone would bury it at the back — which is the whole reason this key exists.
    """
    elements = [
        _reviewable("t0", 0.20),
        _reviewable("t1", 0.55),
        _reviewable("t2", 0.97, disagreement=True),
    ]
    assert build_sidecar(elements, 100, 200, 5)["stats"]["review_queue"] == ["t2", "t0", "t1"]


def test_review_queue_sinks_marking_cells_to_the_back():
    """Measured on test1: 8 of 13 flagged items are marking cells, and they are the least
    fixable. In plain confidence order they would fill the front of the queue."""
    elements = [
        _reviewable("cell0", 0.19, margin=True),
        _reviewable("prose", 0.60),
        _reviewable("cell1", 0.27, margin=True),
    ]
    queue = build_sidecar(elements, 100, 200, 5)["stats"]["review_queue"]
    assert queue == ["prose", "cell0", "cell1"]


def test_review_queue_orders_the_rest_least_confident_first():
    elements = [_reviewable("t0", 0.70), _reviewable("t1", 0.30), _reviewable("t2", 0.50)]
    assert build_sidecar(elements, 100, 200, 5)["stats"]["review_queue"] == ["t1", "t2", "t0"]


def test_sidecar_page_dimensions():
    result = build_sidecar([], 1000, 1500, 350)
    assert result["page_dimensions"]["width_px"] == 1000
    assert result["page_dimensions"]["height_px"] == 1500


def test_sidecar_stats_counts():
    elements = [
        make_text("t0", "hello", 0.9),
        make_text("t1", "wrld", 0.5, corrected="world", highlight="yellow"),
        SimpleShapeElement(id="s0", bbox=BBox(x=0.3, y=0.2, w=0.1, h=0.1), shape="circle", confidence=0.97),
        ComplexShapeElement(id="cs0", bbox=BBox(x=0.5, y=0.3, w=0.15, h=0.1)),
    ]
    result = build_sidecar(elements, 1000, 1500, 250)
    assert result["stats"]["text_elements"] == 2
    assert result["stats"]["simple_shapes"] == 1
    assert result["stats"]["complex_shapes"] == 1
    assert result["stats"]["llm_corrections"] == 1
    assert result["stats"]["processing_time_ms"] == 250


def test_sidecar_element_serialization():
    elements = [make_text("t0", "hello", 0.9)]
    result = build_sidecar(elements, 500, 700, 100)
    el = result["elements"][0]
    assert el["id"] == "t0"
    assert el["type"] == "text"
    assert "bbox" in el
    assert "confidence" in el


def test_sidecar_total_elements():
    elements = [
        make_text("t0", "hello", 0.9),
        SimpleShapeElement(id="s0", bbox=BBox(x=0.3, y=0.2, w=0.1, h=0.1), shape="circle", confidence=0.97),
    ]
    result = build_sidecar(elements, 800, 600, 100)
    assert result["stats"]["total_elements"] == 2


def test_sidecar_polyline_shape_fields():
    elements = [
        PolylineShapeElement(
            id="pl0", bbox=BBox(x=0.1, y=0.2, w=0.3, h=0.4),
            points=[(0.0, 0.0), (0.5, 0.75), (1.0, 1.0)], confidence=0.8,
        )
    ]
    result = build_sidecar(elements, 800, 600, 100)
    el = result["elements"][0]
    assert el["type"] == "polyline_shape"
    assert el["shape"] == "polyline"
    # Counted apart from complex_shapes: a polyline is vector art, not an embedded image.
    assert result["stats"]["polyline_shapes"] == 1
    assert result["stats"]["complex_shapes"] == 0
    assert result["stats"]["simple_shapes"] == 0
    # Points must be JSON-serialisable lists (this dict goes out over the wire).
    assert el["points"] == [[0.0, 0.0], [0.5, 0.75], [1.0, 1.0]]
    json.dumps(result)


def test_sidecar_complex_shape_fields():
    elements = [
        ComplexShapeElement(id="cs0", bbox=BBox(x=0.5, y=0.3, w=0.15, h=0.1),
                            llm_label="door swing", llm_label_certainty=0.88)
    ]
    result = build_sidecar(elements, 800, 600, 100)
    el = result["elements"][0]
    assert el["type"] == "complex_shape"
    assert el["llm_label"] == "door swing"
    assert el["embedded_as"] == "image"


def test_low_resolution_is_reported_from_the_median_text_height():
    from pipeline.layout_reconstructor import LOW_RESOLUTION_TEXT_HEIGHT_PX
    under = build_sidecar([], 800, 600, 10,
                          median_text_height_px=LOW_RESOLUTION_TEXT_HEIGHT_PX - 1)["stats"]
    over = build_sidecar([], 800, 600, 10,
                         median_text_height_px=LOW_RESOLUTION_TEXT_HEIGHT_PX + 1)["stats"]
    assert under["low_resolution"] is True
    assert over["low_resolution"] is False
    assert over["median_text_height_px"] == LOW_RESOLUTION_TEXT_HEIGHT_PX + 1


def test_a_page_with_no_text_is_not_reported_as_low_resolution():
    """0.0 means "no text found", not "text too small to read". A drawing with no
    labels is a normal input here, and flagging it would tell the caller to go rescan
    a sheet that is perfectly fine."""
    stats = build_sidecar([], 800, 600, 10, median_text_height_px=0.0)["stats"]
    assert stats["low_resolution"] is False


def test_an_older_caller_that_passes_no_height_gets_no_low_resolution_claim():
    stats = build_sidecar([], 800, 600, 10)["stats"]
    assert stats["low_resolution"] is False
    assert stats["median_text_height_px"] is None
