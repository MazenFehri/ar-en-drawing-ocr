import io
import re
import numpy as np
import pytest
from docx import Document
from models.elements import (
    BBox, TextElement, SimpleShapeElement, PolylineShapeElement, ComplexShapeElement,
)
from pipeline.word_assembler import assemble_document, _page_geometry, _Z_GRAPHIC, _PATH_SPACE


def make_text_el(id_, text, x=0.1, y=0.05, lang="english", highlight=None):
    return TextElement(
        id=id_, bbox=BBox(x=x, y=y, w=0.2, h=0.03),
        content=text, language=lang, confidence=0.9, highlight=highlight
    )


def _body_xml(elements):
    doc = Document(io.BytesIO(assemble_document(elements, crop_images={})))
    return doc.element.body.xml


def test_an_arabic_run_is_marked_complex_script():
    """`w:bidi` sets the paragraph's base direction; `w:rtl` marks the run as complex-script.

    Only the second makes the `w:szCs` written beside it mean anything — complex-script font
    size applies to runs Word treats as complex-script, and without `w:rtl` an Arabic run was
    never one, so szCs was being emitted and ignored on every Arabic line produced.
    """
    xml = _body_xml([make_text_el("t0", "غرفة المعيشة", lang="arabic")])
    assert "<w:bidi/>" in xml
    assert "<w:rtl/>" in xml


def test_a_latin_run_is_not_marked_rtl():
    xml = _body_xml([make_text_el("t0", "KITCHEN", lang="english")])
    assert "<w:rtl/>" not in xml
    assert "<w:bidi/>" not in xml


def test_run_properties_are_in_schema_order():
    """CT_RPr is a sequence, not a bag: sz, szCs, highlight, rtl. Word tolerates any order;
    LibreOffice, validators and the .NET caller's parser are entitled not to."""
    xml = _body_xml([make_text_el("t0", "غرفة", lang="arabic", highlight="red")])
    rpr = re.search(r"<w:rPr>.*?</w:rPr>", xml, re.S).group(0)
    positions = [rpr.index(tag) for tag in ("<w:sz ", "<w:szCs ", "<w:highlight ", "<w:rtl/>")]
    assert positions == sorted(positions), rpr


def test_digits_in_an_arabic_line_are_left_exactly_as_stored():
    """No directional isolates are injected. The digits already resolve correctly under
    UAX#9 inside an RTL run, so wrapping them would be a second correction applied to a
    correct string — the same mistake as the visual-order reversal this pipeline deleted."""
    xml = _body_xml([make_text_el("t0", "مجدي ب 8500 مي", lang="arabic")])
    assert "8500" in xml
    for control in ("⁦", "⁧", "⁨", "⁩", "‪", "‫", "‬"):
        assert control not in xml


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


def _relative_heights(docx_bytes):
    """relativeHeight per anchor, in document order."""
    doc = Document(io.BytesIO(docx_bytes))
    xml = doc.element.body.xml
    return [int(v) for v in re.findall(r'relativeHeight="(\d+)"', xml)]


def test_text_stacks_above_shapes_and_images():
    """A label inside a box must read on top of it. Every anchor used to carry
    the same relativeHeight, so stacking fell back to document order — and
    reconstruct_layout emits text first, graphics second, which painted the
    shape over the text and hid it. Graphics must sit in a lower band.
    """
    crop = np.ones((50, 80, 3), dtype=np.uint8) * 128
    # Deliberately in the order reconstruct_layout produces: text, then graphics.
    elements = [
        make_text_el("t0", "kitchen", x=0.32, y=0.22),
        SimpleShapeElement(id="s0", bbox=BBox(x=0.3, y=0.2, w=0.3, h=0.2),
                           shape="rect", confidence=0.93),
        ComplexShapeElement(id="cs0", bbox=BBox(x=0.3, y=0.2, w=0.3, h=0.2)),
    ]
    heights = _relative_heights(assemble_document(elements, crop_images={"cs0": crop}))
    assert len(heights) == 3, f"expected one anchor per element, got {heights}"
    text_z, shape_z, image_z = heights
    assert text_z > shape_z, "shape would paint over the text inside it"
    assert text_z > image_z, "embedded image would paint over the text inside it"


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


def _polyline_el(points, id_="p0"):
    return PolylineShapeElement(
        id=id_, bbox=BBox(x=0.1, y=0.1, w=0.4, h=0.3), points=points, confidence=0.8,
    )


def test_polyline_is_a_vector_freeform_not_a_picture():
    # The user-visible bug: connector lines were rasterised. A polyline must come out as
    # a real DrawingML freeform (a:custGeom with an a:path of a:lnTo points), so it stays
    # editable vector art in Word — never a <pic:pic>.
    result = assemble_document([_polyline_el([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])],
                               crop_images={})
    doc = Document(io.BytesIO(result))
    xml = doc.paragraphs[0].runs[0]._r.xml
    assert "<a:custGeom>" in xml
    assert "<a:prstGeom" not in xml
    assert "<pic:pic" not in xml
    assert xml.count("<a:moveTo>") == 1
    assert xml.count("<a:lnTo>") == 2
    # Open path: closing it would draw a phantom chord back to the start.
    assert "<a:close/>" not in xml


def test_polyline_points_map_into_the_path_coordinate_space():
    result = assemble_document([_polyline_el([(0.0, 0.0), (0.5, 0.25), (1.0, 1.0)])],
                               crop_images={})
    doc = Document(io.BytesIO(result))
    xml = doc.paragraphs[0].runs[0]._r.xml
    pts = re.findall(r'<a:pt x="(\d+)" y="(\d+)"/>', xml)
    assert pts == [
        ("0", "0"),
        (str(_PATH_SPACE // 2), str(_PATH_SPACE // 4)),
        (str(_PATH_SPACE), str(_PATH_SPACE)),
    ]
    assert f'<a:path w="{_PATH_SPACE}" h="{_PATH_SPACE}" fill="none">' in xml


def test_polyline_is_outline_only_and_sits_below_text():
    result = assemble_document(
        [_polyline_el([(0.0, 0.0), (1.0, 1.0)]), make_text_el("t0", "label")],
        crop_images={},
    )
    doc = Document(io.BytesIO(result))
    xml = doc.paragraphs[0].runs[0]._r.xml
    # Same outline-only styling as the preset shapes; schema order noFill then ln.
    assert re.search(r"</a:custGeom>\s*<a:noFill/>\s*<a:ln", xml)
    assert 'srgbClr val="000000"' in xml
    # Graphics z-band, so text still reads on top (deliberate earlier fix).
    heights = _relative_heights(result)
    assert heights[0] == _Z_GRAPHIC
    assert heights[1] > _Z_GRAPHIC


def test_polyline_with_too_few_points_is_skipped_not_emitted_broken():
    # A one-point "path" has nothing to draw; emitting a custGeom with a lone moveTo
    # would be a shape Word can't render. Skip it rather than write junk.
    result = assemble_document([_polyline_el([(0.5, 0.5)])], crop_images={})
    doc = Document(io.BytesIO(result))
    assert "<a:custGeom>" not in doc.element.body.xml


def test_polyline_document_is_valid_zip_and_xml():
    # A malformed a:custGeom makes Word refuse the whole file, so validate rather
    # than assume.
    import zipfile
    from lxml import etree as ET

    elements = [
        _polyline_el([(0.0, 0.0), (0.2, 0.9), (0.7, 0.1), (1.0, 1.0)], id_="p0"),
        _polyline_el([(0.0, 1.0), (1.0, 0.0)], id_="p1"),
        make_text_el("t0", "connector label"),
    ]
    result = assemble_document(elements, crop_images={}, page_aspect=1.4)
    zf = zipfile.ZipFile(io.BytesIO(result))
    assert zf.testzip() is None
    ET.fromstring(zf.read("word/document.xml"))


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
