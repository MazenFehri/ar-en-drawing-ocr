import numpy as np
import cv2
import pytest
from pipeline.shape_detector import detect_shapes, ShapeResult, _classify

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


def test_text_in_filled_box_does_not_produce_phantom_rects():
    # A colour-filled box reads as solid ink under Otsu, so blanking each text bbox out
    # of `binary` punches holes in it that RETR_CCOMP reports as nested contours. Left
    # in, every word inside the box came back as its own "rect" and the document drew an
    # empty rectangle around each label. Only the box itself should be reported.
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    pts = np.array([[200, 50], [320, 150], [300, 300], [150, 350], [80, 200]], np.int32)
    cv2.fillPoly(img, [pts], (120, 60, 20))
    text_bboxes = [
        {"x": 170, "y": 190, "w": 40, "h": 15},
        {"x": 170, "y": 220, "w": 60, "h": 15},
    ]
    for tb in text_bboxes:
        cv2.rectangle(
            img, (tb["x"], tb["y"]), (tb["x"] + tb["w"], tb["y"] + tb["h"]), (0, 0, 0), -1,
        )

    shapes = detect_shapes(img, text_bboxes_px=text_bboxes)

    for tb in text_bboxes:
        for s in shapes:
            hugs_text = (
                abs(s.bbox_px["x"] - tb["x"]) <= 4 and abs(s.bbox_px["y"] - tb["y"]) <= 4 and
                abs(s.bbox_px["w"] - tb["w"]) <= 8 and abs(s.bbox_px["h"] - tb["h"]) <= 8
            )
            assert not hugs_text, f"phantom shape reported at text bbox {tb}: {s.bbox_px}"
    # The box itself must still survive — dropping the holes must not drop real content.
    assert any(s.crop is not None for s in shapes)


def test_text_inside_complex_shape_not_baked_into_crop():
    # Regression test for: crops were cut from the raw image, so a "complex" shape
    # whose bbox enclosed OCR'd text (e.g. a colour-filled class-diagram box) carried
    # the words into the embedded picture — duplicating them alongside the real OCR
    # text run. The box's colour fill must survive; only the text pixels must not.
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    fill_color = (120, 60, 20)  # BGR, dark enough to read as ink, distinct from black/white
    pts = np.array([[200, 50], [320, 150], [300, 300], [150, 350], [80, 200]], np.int32)
    cv2.fillPoly(img, [pts], fill_color)

    # Dark "text" blobs sitting well inside the box, away from its edges.
    text_bboxes = [
        {"x": 170, "y": 190, "w": 40, "h": 15},
        {"x": 170, "y": 220, "w": 60, "h": 15},
    ]
    for tb in text_bboxes:
        cv2.rectangle(
            img, (tb["x"], tb["y"]), (tb["x"] + tb["w"], tb["y"] + tb["h"]), (0, 0, 0), -1,
        )
    original = img.copy()

    shapes = detect_shapes(img, text_bboxes_px=text_bboxes)
    complex_shapes = [s for s in shapes if s.shape_type == "complex" and s.crop is not None]
    assert len(complex_shapes) >= 1

    for shape in complex_shapes:
        sx, sy = shape.bbox_px["x"], shape.bbox_px["y"]
        crop = shape.crop
        for tb in text_bboxes:
            lx0, ly0 = tb["x"] - sx, tb["y"] - sy
            lx1, ly1 = lx0 + tb["w"], ly0 + tb["h"]
            if lx0 < 0 or ly0 < 0 or lx1 > crop.shape[1] or ly1 > crop.shape[0]:
                continue  # this text box isn't inside this particular shape's crop
            patch = crop[ly0:ly1, lx0:lx1]
            # No dark text pixels should have survived into the crop.
            assert not np.any(np.all(patch < 50, axis=-1)), "text pixels leaked into crop"
            # The erased patch should read as the box's fill colour, not a white punch-out.
            median_patch = np.median(patch.reshape(-1, 3), axis=0)
            assert np.allclose(median_patch, fill_color, atol=20), (
                f"expected fill colour {fill_color}, got {median_patch}"
            )

    # detect_shapes must not mutate the caller's image while building the crop source.
    assert np.array_equal(img, original)


def _ink_contour(h, w):
    """Contour of a solid h x w ink block, as detect_shapes would find it."""
    binary = np.zeros((h + 20, w + 20), dtype=np.uint8)
    binary[10:10 + h, 10:10 + w] = 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


@pytest.mark.parametrize("h,w", [(1, 300), (300, 1), (2, 300), (5, 300)])
def test_thin_stroke_is_a_line_not_a_raster(h, w):
    # A perfectly thin axis-aligned stroke makes cv2.minAreaRect report a short side of
    # exactly 0.0. The old `short_side > 0` guard then skipped the line test entirely and
    # the hairline fell through to "complex" — i.e. got embedded as a picture one pixel
    # tall. The 2px/5px cases are here so the fix doesn't regress what already worked.
    shape_type, confidence = _classify(_ink_contour(h, w))
    assert shape_type == "line", f"{w}x{h} stroke classified as {shape_type}"
    assert confidence == 0.9


def test_hairline_end_to_end_is_a_line():
    img = make_canvas()
    cv2.line(img, (20, 150), (280, 150), (0, 0, 0), 1)
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert [s.shape_type for s in shapes] == ["line"]
    assert shapes[0].crop is None  # no raster embedding for a line


def test_open_elbow_is_a_polyline_with_points():
    # An L-shaped connector: an open stroke, so the contour traces out and back and
    # encloses ~no area. There was no shape type for that, so it became a raster image.
    img = make_canvas(400, 400)
    pts = np.array([[50, 50], [50, 300], [350, 300]], np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=(0, 0, 0), thickness=2)

    shapes = detect_shapes(img, text_bboxes_px=[])
    polylines = [s for s in shapes if s.shape_type == "polyline"]
    assert len(polylines) == 1, [s.shape_type for s in shapes]

    poly = polylines[0]
    assert poly.crop is None  # a path, not a picture
    assert poly.points is not None and len(poly.points) >= 3
    # Points are fractions of the shape's own bbox, so they survive px -> relative -> EMU.
    assert all(0.0 <= px <= 1.0 and 0.0 <= py <= 1.0 for px, py in poly.points)
    # The elbow corner (bottom-left of the bbox) must actually be in the path.
    assert any(px < 0.15 and py > 0.85 for px, py in poly.points)


def test_curved_open_stroke_is_a_polyline():
    img = make_canvas(400, 400)
    curve = np.array(
        [[40 + i * 6, int(200 + 140 * np.sin(i / 55.0 * np.pi))] for i in range(55)],
        np.int32,
    )
    cv2.polylines(img, [curve], isClosed=False, color=(0, 0, 0), thickness=2)
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert any(s.shape_type == "polyline" for s in shapes), [s.shape_type for s in shapes]


def test_filled_blob_still_rasterised():
    # The other half of the fix: raster embedding must survive for genuinely image-like
    # regions. A solid, high-extent blob is not a stroke and must still get a crop.
    img = make_canvas(400, 400)
    pts = np.array([[200, 50], [320, 150], [300, 300], [150, 350], [80, 200]], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 0))
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert len(shapes) == 1
    assert shapes[0].shape_type == "complex"
    assert shapes[0].crop is not None
    assert shapes[0].points is None


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
