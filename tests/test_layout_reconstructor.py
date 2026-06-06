import pytest
from pipeline.layout_reconstructor import reconstruct_layout, to_relative_bbox
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
