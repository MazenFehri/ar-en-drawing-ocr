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


def _sz_values(docx_bytes):
    """Pull every <w:sz w:val="N"/> from the run's raw XML, in order."""
    doc = Document(io.BytesIO(docx_bytes))
    xml = doc.paragraphs[0].runs[0]._r.xml
    return [int(v) for v in re.findall(r'<w:sz w:val="(\d+)"/>', xml)]


def test_font_size_scales_with_box_height():
    # Bug 1: box size tracks the OCR bbox, but font size was hardcoded to
    # 10pt (w:sz=20) regardless of box height. A short box must now get a
    # smaller font than a tall one.
    elements = [
        make_text_el("t0", "small", x=0.05, y=0.05, lang="english"),
    ]
    elements[0].bbox.h = 0.01
    small_sz = _sz_values(assemble_document(elements, crop_images={}))[0]

    elements[0].bbox.h = 0.2
    large_sz = _sz_values(assemble_document(elements, crop_images={}))[0]

    assert large_sz > small_sz


def test_font_size_has_a_minimum_clamp():
    # A near-zero-height text box (tiny OCR artefact) must not shrink the
    # font below the legibility floor (4pt == w:sz 8).
    from pipeline.word_assembler import _MIN_FONT_HALF_PT
    elements = [make_text_el("t0", "x", x=0.05, y=0.05)]
    elements[0].bbox.h = 0.0001
    sz = _sz_values(assemble_document(elements, crop_images={}))[0]
    assert sz == _MIN_FONT_HALF_PT


def test_font_size_has_a_maximum_clamp():
    # A huge detected box (e.g. a title) must not produce absurd type;
    # capped at 72pt == w:sz 144.
    from pipeline.word_assembler import _MAX_FONT_HALF_PT
    elements = [make_text_el("t0", "TITLE", x=0.05, y=0.05)]
    elements[0].bbox.h = 5.0
    sz = _sz_values(assemble_document(elements, crop_images={}))[0]
    assert sz == _MAX_FONT_HALF_PT


def test_font_size_keys_off_height_not_width_for_wide_short_label():
    # A long-but-short label (wide box, small height) must get a small
    # font driven by height alone, not inflated by the box's width/area.
    from pipeline.word_assembler import _MIN_FONT_HALF_PT
    elements = [make_text_el("t0", "a very long room label", x=0.02, y=0.05)]
    elements[0].bbox.w = 0.9
    elements[0].bbox.h = 0.01
    sz = _sz_values(assemble_document(elements, crop_images={}))[0]
    assert sz < 20  # well under the old hardcoded 10pt-equivalent-ish range
    assert sz >= _MIN_FONT_HALF_PT


def test_text_box_body_pr_has_zeroed_insets_and_no_wrap():
    elements = [make_text_el("t0", "entrance")]
    result = assemble_document(elements, crop_images={})
    doc = Document(io.BytesIO(result))
    xml = doc.paragraphs[0].runs[0]._r.xml
    assert 'wrap="none"' in xml
    assert 'lIns="0"' in xml
    assert 'tIns="0"' in xml
    assert 'rIns="0"' in xml
    assert 'bIns="0"' in xml


def test_document_with_varied_box_heights_is_valid_zip_and_xml():
    # End-to-end guard against schema/order mistakes: build a doc with
    # tiny/normal/large boxes and Arabic+English content, then confirm the
    # result is still a well-formed .docx (valid zip, parseable document.xml).
    import zipfile
    from lxml import etree as ET

    elements = [
        make_text_el("t0", "tiny", x=0.05, y=0.05, lang="english"),
        make_text_el("t1", "normal room", x=0.05, y=0.2, lang="english"),
        make_text_el("t2", "غرفة كبيرة", x=0.05, y=0.5, lang="arabic"),
    ]
    elements[0].bbox.h = 0.005
    elements[1].bbox.h = 0.03
    elements[2].bbox.h = 0.15

    result = assemble_document(elements, crop_images={})
    zf = zipfile.ZipFile(io.BytesIO(result))
    assert zf.testzip() is None  # valid zip, no corrupt members
    xml_bytes = zf.read("word/document.xml")
    ET.fromstring(xml_bytes)  # raises if malformed
