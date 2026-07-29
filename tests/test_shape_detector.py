import numpy as np
import cv2
import pytest
from pipeline.shape_detector import (
    detect_shapes, ShapeResult, _Seg, _merge_collinear, _assemble_rectangles,
)

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


@pytest.mark.parametrize("h,w", [(1, 300), (300, 1), (2, 300), (5, 300)])
def test_thin_stroke_is_a_line_not_a_raster(h, w):
    # A thin axis-aligned stroke used to be classified from its contour, where
    # minAreaRect reports a short side of exactly 0.0 and the hairline fell through to
    # "complex" — i.e. got embedded as a picture one pixel tall. Strokes now come from
    # the segment detector instead, so this asserts the end-to-end outcome rather than
    # a contour heuristic that no longer exists.
    img = make_canvas(h + 60, w + 60)
    img[30:30 + h, 30:30 + w] = 0
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert [s.shape_type for s in shapes] == ["line"], f"{w}x{h} stroke -> {shapes}"
    assert shapes[0].crop is None


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


def test_merge_collinear_collapses_both_sides_of_one_stroke():
    # An edge detector reports a drawn stroke twice, once per side. Left unmerged, a box
    # edge is two segments and nothing lines up into a rectangle.
    merged = _merge_collinear([_Seg(20, 100, 280, 100), _Seg(20, 103, 280, 103)])
    assert len(merged) == 1
    assert merged[0].length == pytest.approx(260, abs=3)


def test_merge_collinear_bridges_a_small_gap():
    # A dashed or partly-occluded edge arrives in fragments; they are one edge.
    merged = _merge_collinear([_Seg(20, 100, 140, 100), _Seg(150, 100, 280, 100)])
    assert len(merged) == 1
    assert merged[0].length == pytest.approx(260, abs=3)


def test_merge_collinear_keeps_a_real_gap_apart():
    merged = _merge_collinear([_Seg(20, 100, 140, 100), _Seg(220, 100, 280, 100)])
    assert len(merged) == 2


def test_rectangle_assembled_from_four_separate_segments():
    # The whole point of the rewrite: a rectangle drawn as four strokes has no single
    # closed contour to classify, so contour tracing could never recover it.
    segs = [
        _Seg(40, 40, 240, 40), _Seg(240, 40, 240, 200),
        _Seg(240, 200, 40, 200), _Seg(40, 200, 40, 40),
    ]
    rects, used = _assemble_rectangles(segs)
    assert len(rects) == 1
    assert rects[0].shape_type == "rect"
    assert used == {0, 1, 2, 3}
    assert rects[0].bbox_px["w"] == pytest.approx(200, abs=2)
    assert rects[0].bbox_px["h"] == pytest.approx(160, abs=2)


def test_rotated_rectangle_becomes_a_closed_freeform():
    # ShapeResult has no rotation field, so an axis-aligned "rect" would misdraw a
    # tilted box. It goes out as a closed path instead, which is exact.
    corners = [(150, 40), (280, 130), (190, 260), (60, 170)]
    segs = [_Seg(*corners[i], *corners[(i + 1) % 4]) for i in range(4)]
    rects, used = _assemble_rectangles(segs)
    assert len(rects) == 1
    assert rects[0].shape_type == "polyline"
    assert rects[0].points is not None and len(rects[0].points) == 5
    assert rects[0].points[0] == rects[0].points[-1]  # closed


def test_box_touching_a_connector_still_yields_a_rectangle():
    # class-diagram.png's actual failure mode, minimised: the moment a connector line
    # touches a box border the two are one connected region, so findContours returned a
    # snake around both and 0 of 11 class boxes came back as rectangles.
    img = make_canvas(400, 400)
    cv2.rectangle(img, (60, 60), (300, 220), (0, 0, 0), 2)
    cv2.line(img, (300, 140), (380, 140), (0, 0, 0), 2)
    cv2.line(img, (180, 220), (180, 380), (0, 0, 0), 2)
    shapes = detect_shapes(img, text_bboxes_px=[])
    rects = [s for s in shapes if s.shape_type in ("rect", "square")]
    assert len(rects) == 1, [s.shape_type for s in shapes]
    assert rects[0].bbox_px["w"] == pytest.approx(240, abs=8)
    assert rects[0].bbox_px["h"] == pytest.approx(160, abs=8)
    assert not any(s.shape_type == "complex" for s in shapes), "line art was rasterised"


def test_forward_diagonal_is_a_freeform_not_a_mirrored_line():
    # prstGeom="line" always runs top-left -> bottom-right of its box, so a "/" segment
    # would render mirrored. Same bbox, wrong picture — that one goes out as a path.
    img = make_canvas(320, 320)
    cv2.line(img, (30, 280), (290, 40), (0, 0, 0), 2)
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert [s.shape_type for s in shapes] == ["polyline"], [s.shape_type for s in shapes]
    pts = shapes[0].points
    assert len(pts) == 2
    # Bottom-left to top-right, in bbox fractions.
    assert {(round(x), round(y)) for x, y in pts} == {(0, 1), (1, 0)}


def test_textured_photographic_region_still_rasterises():
    # The segment pass must not shred a photo into edges. An irregular textured region
    # is not line art and has to keep arriving as one picture with its pixels intact.
    rng = np.random.default_rng(7)
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    texture = cv2.GaussianBlur(rng.integers(0, 200, size=(400, 400, 3), dtype=np.uint8), (7, 7), 0)
    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(
        [[110, 130], [250, 120], [300, 210], [265, 300], [170, 315], [95, 245], [140, 200]],
        np.int32)], 255)
    img[mask > 0] = texture[mask > 0]

    shapes = detect_shapes(img, text_bboxes_px=[])
    rasters = [s for s in shapes if s.shape_type == "complex" and s.crop is not None]
    assert len(rasters) == 1, [(s.shape_type, s.bbox_px) for s in shapes]
    assert rasters[0].bbox_px["w"] > 150 and rasters[0].bbox_px["h"] > 150


def test_circle_touching_a_line_is_one_circle_not_two():
    # A circle fused to a leader line fails the region gate (the contour is a snake),
    # so it comes from the ellipse detector — which fits one ellipse per side of the
    # drawn stroke and would otherwise emit two concentric circles for one drawn one.
    img = make_canvas(400, 400)
    cv2.circle(img, (200, 200), 90, (0, 0, 0), 2)
    cv2.line(img, (290, 200), (390, 200), (0, 0, 0), 2)
    shapes = detect_shapes(img, text_bboxes_px=[])
    circles = [s for s in shapes if s.shape_type in ("circle", "ellipse")]
    assert len(circles) == 1, [(s.shape_type, s.bbox_px) for s in shapes]
    assert circles[0].bbox_px["w"] == pytest.approx(182, abs=12)


def test_lsd_fallback_still_finds_the_box(monkeypatch):
    # Three opencv distributions fight over site-packages/cv2 and only the contrib one
    # carries ximgproc, so the fallback is a real code path, not defensive decoration.
    import pipeline.shape_detector as sd
    monkeypatch.setattr(sd, "HAVE_EDGE_DRAWING", False)
    img = make_canvas(400, 400)
    cv2.rectangle(img, (60, 60), (300, 220), (0, 0, 0), 2)
    cv2.line(img, (300, 140), (380, 140), (0, 0, 0), 2)
    shapes = detect_shapes(img, text_bboxes_px=[])
    rects = [s for s in shapes if s.shape_type in ("rect", "square")]
    assert len(rects) == 1, [s.shape_type for s in shapes]


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
