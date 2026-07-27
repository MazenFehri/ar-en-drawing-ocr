import numpy as np
import cv2
import pytest
from pipeline.shape_detector import detect_shapes, ShapeResult

MIN_TEST_AREA = 400  # Must match the module's MIN_AREA_PX

def make_canvas(h=300, w=300):
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def test_detect_circle():
    img = make_canvas()
    cv2.circle(img, (150, 150), 60, (0, 0, 0), -1)
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert len(shapes) == 1
    assert shapes[0].shape_type == "circle"


def test_detect_triangle():
    img = make_canvas()
    pts = np.array([[150, 30], [30, 270], [270, 270]], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 0))
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert len(shapes) == 1
    assert shapes[0].shape_type == "triangle"


def test_detect_rectangle():
    img = make_canvas()
    cv2.rectangle(img, (40, 80), (260, 220), (0, 0, 0), -1)
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert len(shapes) == 1
    assert shapes[0].shape_type in ("rect", "square")


def test_detect_ellipse():
    img = make_canvas(300, 400)
    # Draw a wide ellipse (axes 120x60)
    cv2.ellipse(img, (200, 150), (120, 60), 0, 0, 360, (0, 0, 0), -1)
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert len(shapes) == 1
    assert shapes[0].shape_type in ("ellipse", "circle")  # both are acceptable for oval shapes


def test_shape_result_has_bbox():
    img = make_canvas()
    cv2.circle(img, (150, 150), 60, (0, 0, 0), -1)
    shapes = detect_shapes(img, text_bboxes_px=[])
    s = shapes[0]
    assert "x" in s.bbox_px
    assert "y" in s.bbox_px
    assert "w" in s.bbox_px
    assert "h" in s.bbox_px
    assert s.bbox_px["w"] > 0
    assert s.bbox_px["h"] > 0


def test_complex_shape_has_crop():
    img = make_canvas(400, 400)
    pts = np.array([[200, 50], [320, 150], [300, 300], [150, 350], [80, 200]], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 0))
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert len(shapes) == 1
    assert shapes[0].shape_type == "complex"
    assert shapes[0].crop is not None
    assert isinstance(shapes[0].crop, np.ndarray)


def test_noise_filtered():
    img = make_canvas()
    # Tiny 4x4 dot — below MIN_AREA_PX
    cv2.rectangle(img, (10, 10), (14, 14), (0, 0, 0), -1)
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert len(shapes) == 0


def test_text_region_excluded():
    img = make_canvas()
    cv2.circle(img, (150, 150), 60, (0, 0, 0), -1)
    # Declare the circle's area as a text region — should be masked out
    text_bboxes = [{"x": 80, "y": 80, "w": 140, "h": 140}]
    shapes = detect_shapes(img, text_bboxes_px=text_bboxes)
    assert len(shapes) == 0


def test_shape_result_dataclass():
    img = make_canvas()
    cv2.circle(img, (150, 150), 60, (0, 0, 0), -1)
    shapes = detect_shapes(img, text_bboxes_px=[])
    s = shapes[0]
    assert hasattr(s, "shape_type")
    assert hasattr(s, "bbox_px")
    assert hasattr(s, "confidence")
    assert hasattr(s, "crop")


def test_nested_shape_detected():
    # Solid outer square with a "room" cut out (white hole), and a separate small
    # symbol (circle) sitting inside that hole, not touching the outer square.
    # RETR_EXTERNAL used to see one connected blob and drop the inner circle entirely.
    img = make_canvas()
    cv2.rectangle(img, (30, 30), (270, 270), (0, 0, 0), -1)
    cv2.rectangle(img, (70, 70), (230, 230), (255, 255, 255), -1)
    cv2.circle(img, (150, 150), 30, (0, 0, 0), -1)
    shapes = detect_shapes(img, text_bboxes_px=[])
    # Both the outer square and the nested circle must be present now.
    assert len(shapes) >= 2
    inner = [s for s in shapes if s.bbox_px["w"] < 100 and s.bbox_px["h"] < 100]
    assert len(inner) >= 1
    assert any(s.shape_type == "circle" for s in inner)


def test_hollow_stroke_not_duplicated():
    # A rectangle drawn as an outline (e.g. a wall drawn as two parallel lines) gives
    # an outer contour and an inner "hole" contour of the same stroke. We should keep
    # only one shape for it, not double-report outside+inside edges.
    img = make_canvas()
    cv2.rectangle(img, (30, 30), (270, 270), (0, 0, 0), 6)
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert len(shapes) == 1


def test_thin_line_classified():
    img = make_canvas()
    cv2.line(img, (20, 150), (280, 150), (0, 0, 0), 2)
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert len(shapes) == 1
    assert shapes[0].shape_type == "line"


def test_diagonal_line_classified():
    img = make_canvas()
    cv2.line(img, (20, 20), (280, 280), (0, 0, 0), 2)
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert len(shapes) == 1
    assert shapes[0].shape_type == "line"


def test_residual_ink_produces_complex_shape():
    # A cluster of small strokes, each individually below the noise floor, but
    # together clearly meaningful ink (e.g. hatching or a broken/dashed leader).
    # None of them survives as its own contour, so they must come back as one
    # "complex" catch-all shape rather than vanishing.
    img = make_canvas()
    for i in range(6):
        x = 40 + i * 8
        cv2.rectangle(img, (x, 100), (x + 3, 103), (0, 0, 0), -1)
    shapes = detect_shapes(img, text_bboxes_px=[])
    complex_shapes = [s for s in shapes if s.shape_type == "complex"]
    assert len(complex_shapes) >= 1
    assert complex_shapes[0].crop is not None
