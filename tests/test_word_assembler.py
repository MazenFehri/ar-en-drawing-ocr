import io
import re
import numpy as np
import pytest
from docx import Document
from models.elements import BBox, TextElement, SimpleShapeElement, ComplexShapeElement
from pipeline.word_assembler import assemble_document, _page_geometry


def make_text_el(id_, text, x=0.1, y=0.05, lang="english", highlight=None):
    return TextElement(
        id=id_, bbox=BBox(x=x, y=y, w=0.2, h=0.03),
        content=text, language=lang, confidence=0.9, highlight=highlight
    )


def test_assemble_returns_bytes():
    elements = [make_text_el("t0", "entrance")]
    result = assemble_document(elements, crop_images={})
    assert isinstance(result, bytes)
    assert len(result) > 100


def test_assemble_is_valid_docx():
    elements = [make_text_el("t0", "entrance")]
    docx_bytes = assemble_document(elements, crop_images={})
    doc = Document(io.BytesIO(docx_bytes))
    assert doc is not None


def test_assemble_with_circle_shape():
    elements = [
        SimpleShapeElement(id="s0", bbox=BBox(x=0.3, y=0.2, w=0.1, h=0.1),
                           shape="circle", confidence=0.97)
    ]
    result = assemble_document(elements, crop_images={})
    assert isinstance(result, bytes)


def test_assemble_with_complex_shape_crop():
    crop = np.ones((50, 80, 3), dtype=np.uint8) * 128
    elements = [ComplexShapeElement(id="cs0", bbox=BBox(x=0.5, y=0.3, w=0.15, h=0.1))]
    result = assemble_document(elements, crop_images={"cs0": crop})
    assert isinstance(result, bytes)


def test_assemble_arabic_text():
    elements = [make_text_el("t0", "غرفة النوم", lang="arabic")]
    result = assemble_document(elements, crop_images={})
    assert isinstance(result, bytes)


def test_assemble_yellow_highlight():
    elements = [make_text_el("t0", "entrance", highlight="yellow")]
    result = assemble_document(elements, crop_images={})
    assert isinstance(result, bytes)


def test_assemble_multiple_elements():
    elements = [
        make_text_el("t0", "kitchen", x=0.05, y=0.1),
        make_text_el("t1", "bedroom", x=0.5, y=0.3),
        SimpleShapeElement(id="s0", bbox=BBox(x=0.3, y=0.2, w=0.1, h=0.1),
                           shape="triangle", confidence=0.95),
    ]
    result = assemble_document(elements, crop_images={})
    assert isinstance(result, bytes)


def _last_extent(docx_bytes):
    """Pull the (cx, cy) of the last <wp:extent> in the run's raw XML."""
    doc = Document(io.BytesIO(docx_bytes))
    xml = doc.paragraphs[0].runs[0]._r.xml
    matches = re.findall(r'<wp:extent cx="(\d+)" cy="(\d+)"/>', xml)
    cx, cy = matches[-1]
    return int(cx), int(cy)


def test_simple_shape_is_outline_only_not_solid_blue():
    # Bug 2: shapes must be transparent-fill + thin black stroke, not Word's
    # default solid-blue-fill style (which would hide anything drawn under it).
    elements = [SimpleShapeElement(id="s0", bbox=BBox(x=0.1, y=0.1, w=0.1, h=0.1),
                                    shape="rect", confidence=0.9)]
    result = assemble_document(elements, crop_images={})
    doc = Document(io.BytesIO(result))
    xml = doc.paragraphs[0].runs[0]._r.xml
    assert "<a:noFill/>" in xml
    assert "srgbClr val=\"000000\"" in xml
    # noFill must appear straight after prstGeom, before ln, per schema order
    assert re.search(r"</a:prstGeom>\s*<a:noFill/>\s*<a:ln", xml)


def test_thin_line_is_not_inflated_to_a_box():
    # Bug 3: a near-zero-height horizontal line must keep its thin axis
    # instead of being forced to the old 0.1in-both-axes minimum.
    elements = [SimpleShapeElement(id="s0", bbox=BBox(x=0.1, y=0.1, w=0.5, h=0.0),
                                    shape="line", confidence=0.9)]
    result = assemble_document(elements, crop_images={})
    cx, cy = _last_extent(result)
    assert cy < 91_440
    assert cx > 91_440  # the long axis is untouched


def test_degenerate_point_still_gets_a_visible_floor():
    # Both axes collapsed: this is the one case the old 0.1in floor exists for.
    elements = [SimpleShapeElement(id="s0", bbox=BBox(x=0.1, y=0.1, w=0.0, h=0.0),
                                    shape="rect", confidence=0.9)]
    result = assemble_document(elements, crop_images={})
    cx, cy = _last_extent(result)
    assert cx == 91_440
    assert cy == 91_440


def test_page_geometry_none_matches_legacy_constants():
    from pipeline.word_assembler import _MARGIN, _USE_W, _USE_H
    doc = Document()
    assert _page_geometry(doc, None) == (_MARGIN, _MARGIN, _USE_W, _USE_H)


def test_page_geometry_letterbox_preserves_source_aspect():
    doc = Document()
    for aspect in (2.0, 1.3333, 0.5, 1.0):
        _, _, box_w, box_h = _page_geometry(doc, aspect)
        assert box_w / box_h == pytest.approx(aspect, rel=1e-3)


def test_letterbox_keeps_square_shape_square_on_landscape_page():
    # Source is 2:1 landscape. A square shape in *pixel* space has
    # bbox.h == 2 * bbox.w (h is normalized by the shorter source dimension).
    # After correct letterboxing it must render back out as cx == cy.
    elements = [SimpleShapeElement(id="s0", bbox=BBox(x=0.1, y=0.1, w=0.1, h=0.2),
                                    shape="rect", confidence=0.9)]
    result = assemble_document(elements, crop_images={}, page_aspect=2.0)
    cx, cy = _last_extent(result)
    assert cx == cy
