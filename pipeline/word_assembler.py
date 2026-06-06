import io
import tempfile
import os
from lxml import etree
import numpy as np
from docx import Document
from models.elements import Element, TextElement, SimpleShapeElement, ComplexShapeElement
from utils.image_utils import ndarray_to_png_bytes

# A4 page geometry in EMU (914400 EMU = 1 inch)
_PAGE_W = 7_559_670
_PAGE_H = 10_692_720
_MARGIN = 914_400
_USE_W = _PAGE_W - 2 * _MARGIN
_USE_H = _PAGE_H - 2 * _MARGIN

# DrawingML preset geometry names for each simple shape type
_WORD_SHAPE = {
    "circle": "ellipse",
    "ellipse": "ellipse",
    "triangle": "triangle",
    "rect": "rect",
    "square": "rect",
    "line": "line",
}

_HIGHLIGHT_MAP = {"yellow": "yellow", "red": "red"}

_id_counter = [0]


def assemble_document(elements: list[Element], crop_images: dict[str, np.ndarray]) -> bytes:
    """Assemble a Word document with absolutely positioned elements.

    elements: list of TextElement, SimpleShapeElement, ComplexShapeElement
    crop_images: mapping of element id -> numpy BGR image (for ComplexShapeElement)
    Returns: .docx file as bytes
    """
    _id_counter[0] = 0
    doc = Document()
    # Remove the default empty paragraph Word adds
    for p in list(doc.paragraphs):
        p._element.getparent().remove(p._element)

    para = doc.add_paragraph()
    run = para.add_run()

    for el in elements:
        left, top, w, h = _emu_coords(el.bbox)
        xml_str = None

        if isinstance(el, TextElement):
            xml_str = _text_box_xml(
                el.content, left, top, w, h,
                is_rtl=(el.language == "arabic"),
                highlight=el.highlight,
            )
        elif isinstance(el, SimpleShapeElement):
            prst = _WORD_SHAPE.get(el.shape, "rect")
            xml_str = _shape_xml(prst, left, top, w, h)
        elif isinstance(el, ComplexShapeElement):
            crop = crop_images.get(el.id)
            if crop is not None:
                xml_str = _image_xml(doc, ndarray_to_png_bytes(crop), left, top, w, h)

        if xml_str is not None:
            drawing_el = etree.fromstring(xml_str)
            run._r.append(drawing_el)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _emu_coords(bbox) -> tuple[int, int, int, int]:
    left = _MARGIN + int(bbox.x * _USE_W)
    top = _MARGIN + int(bbox.y * _USE_H)
    width = max(int(bbox.w * _USE_W), 91_440)   # min 0.1 inch
    height = max(int(bbox.h * _USE_H), 91_440)
    return left, top, width, height


def _next_id() -> int:
    _id_counter[0] += 1
    return _id_counter[0]


def _anchor_wrap(left: int, top: int, cx: int, cy: int, inner_xml: str) -> str:
    eid = _next_id()
    return (
        '<w:drawing xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<wp:anchor distT="0" distB="0" distL="0" distR="0" simplePos="0" '
        'relativeHeight="251659264" behindDoc="0" locked="0" layoutInCell="1" allowOverlap="1" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
        '<wp:simplePos x="0" y="0"/>'
        f'<wp:positionH relativeFrom="page"><wp:posOffset>{left}</wp:posOffset></wp:positionH>'
        f'<wp:positionV relativeFrom="page"><wp:posOffset>{top}</wp:posOffset></wp:positionV>'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        '<wp:wrapNone/>'
        f'<wp:docPr id="{eid}" name="Element{eid}"/>'
        '<wp:cNvGraphicFramePr/>'
        + inner_xml +
        '</wp:anchor></w:drawing>'
    )


def _text_box_xml(text: str, left: int, top: int, cx: int, cy: int,
                  is_rtl: bool, highlight) -> str:
    ns_a = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
    ns_wps = 'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"'
    ns_w = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    bidi_tag = "<w:bidi/>" if is_rtl else ""
    hl_tag = (
        f'<w:highlight w:val="{_HIGHLIGHT_MAP[highlight]}"/>'
        if highlight and highlight in _HIGHLIGHT_MAP
        else ""
    )
    # Escape XML special chars in text
    safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _anchor_wrap(left, top, cx, cy,
        f'<a:graphic {ns_a}>'
        '<a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">'
        f'<wps:wsp {ns_wps}>'
        '<wps:cNvSpPr txBox="1"/>'
        '<wps:spPr>'
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        '<a:noFill/><a:ln><a:noFill/></a:ln>'
        '</wps:spPr>'
        '<wps:txbx>'
        f'<w:txbxContent {ns_w}>'
        f'<w:p><w:pPr>{bidi_tag}</w:pPr>'
        f'<w:r><w:rPr>{hl_tag}<w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
        f'<w:t xml:space="preserve">{safe_text}</w:t>'
        '</w:r></w:p>'
        '</w:txbxContent>'
        '</wps:txbx>'
        '<wps:bodyPr/>'
        '</wps:wsp>'
        '</a:graphicData>'
        '</a:graphic>'
    )


def _shape_xml(prst: str, left: int, top: int, cx: int, cy: int) -> str:
    ns_a = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
    ns_wps = 'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"'
    return _anchor_wrap(left, top, cx, cy,
        f'<a:graphic {ns_a}>'
        '<a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">'
        f'<wps:wsp {ns_wps}>'
        '<wps:cNvSpPr/>'
        '<wps:spPr>'
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>'
        '</wps:spPr>'
        '<wps:bodyPr/>'
        '</wps:wsp>'
        '</a:graphicData>'
        '</a:graphic>'
    )


def _image_xml(doc: Document, img_bytes: bytes, left: int, top: int, cx: int, cy: int) -> str:
    """Add image to document part and return the drawing XML referencing it.

    Strategy:
    1. Write image bytes to a temp file.
    2. Use doc.add_picture() to register the image with the document part and get a relationship.
    3. Extract the rId from the inline drawing element that python-docx inserts.
    4. Remove the inline drawing element from the document (we only wanted the rId).
    5. Build and return a floating anchor XML that references the same rId.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(img_bytes)
        tmp_path = f.name
    try:
        # add_picture returns an InlineShape; it also appends a paragraph to the doc body
        inline_shape = doc.add_picture(tmp_path)
    finally:
        os.unlink(tmp_path)

    # python-docx appends the picture inside a new paragraph at the end of the body.
    # Extract the rId from the <a:blip r:embed="..."> element inside that drawing.
    NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
    body = doc.element.body

    # The last paragraph added by add_picture
    last_para = body.findall(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
    )[-1]

    # Find the blip element to get rId
    blip = last_para.find(f".//{{{NS_A}}}blip")
    rId = blip.get(f"{{{NS_R}}}embed") if blip is not None else None

    # If blip not found, fall back to getting the last relationship key
    if rId is None:
        rId = list(doc.part.rels.keys())[-1]

    # Remove the paragraph python-docx added (we only needed the rId)
    body.remove(last_para)

    ns_a = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
    ns_pic = 'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
    ns_r = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    eid = _next_id()
    return _anchor_wrap(left, top, cx, cy,
        f'<a:graphic {ns_a}>'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic {ns_pic}>'
        f'<pic:nvPicPr><pic:cNvPr id="{eid}" name="img{eid}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip {ns_r} r:embed="{rId}"/>'
        '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        '<pic:spPr>'
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        '</pic:spPr>'
        '</pic:pic>'
        '</a:graphicData>'
        '</a:graphic>'
    )
