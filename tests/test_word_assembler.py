import io
import numpy as np
import pytest
from docx import Document
from models.elements import BBox, TextElement, SimpleShapeElement, ComplexShapeElement
from pipeline.word_assembler import assemble_document


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
