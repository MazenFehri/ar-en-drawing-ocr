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
