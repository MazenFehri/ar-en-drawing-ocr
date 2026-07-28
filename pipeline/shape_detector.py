import math
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

MIN_AREA_PX = 400  # Ignore contours smaller than 20×20 px
MIN_DIAGONAL_PX = 60  # ...unless the bbox diagonal alone says it's a real (thin) stroke
LINE_ASPECT_RATIO = 8.0  # minAreaRect long/short side above this => "line"
LINE_MAX_THICKNESS_PX = 12  # ...and the short side must actually be thin

# An open stroke (connector, leader, curved association line) is traced out and back,
# so the contour encloses essentially no area. Measured on class-diagram.png the real
# connectors came back at extent 0.000-0.068 and solidity 0.001-0.122, while a filled
# blob (the thing that genuinely wants to be a raster picture) sits near 1.0 on both.
POLYLINE_MAX_EXTENT = 0.10
POLYLINE_MAX_SOLIDITY = 0.30
POLYLINE_EPSILON_FRAC = 0.005  # approxPolyDP tolerance, as a fraction of arc length
POLYLINE_MAX_POINTS = 64  # cap so one noisy squiggle can't emit a thousand-point path
POLYLINE_MIN_POINTS = 2  # fewer than this is not a path; fall back to a raster crop

CLAIMED_STROKE_PX = 3  # pen width used to mark a traced contour's own ink as claimed

RESIDUAL_DILATE_PX = 9  # merge nearby ink fragments into one blob before componentizing
RESIDUAL_MIN_AREA_PX = 150  # noise floor for residual (unclassified) ink blobs
RESIDUAL_MAX_SHAPES = 40  # hard cap so a noisy scan can't spam hundreds of crops

TEXT_ERASE_RING_PX = 6  # ring width sampled just outside a text bbox for its fill colour
TEXT_HOLE_TOL_PX = 4  # how closely a hole contour must hug a masked text bbox to be ours


@dataclass
class ShapeResult:
    # "circle", "ellipse", "triangle", "rect", "square", "line", "polyline", "complex"
    shape_type: str
    bbox_px: dict     # {x, y, w, h} in pixel coordinates
    confidence: float
    crop: Optional[np.ndarray] = None  # Only set for complex shapes
    # Only set for "polyline": the simplified path, as (x, y) fractions of bbox_px.
    # Relative on purpose — the rest of the pipeline maps px -> relative -> EMU, and
    # a path stored in absolute pixels would not survive that.
    points: Optional[list[tuple[float, float]]] = None


def detect_shapes(image: np.ndarray, text_bboxes_px: list[dict]) -> list[ShapeResult]:
    """Detect non-text shapes in an image.

    text_bboxes_px: list of {x, y, w, h} dicts in pixel coords — these regions are masked.
    Returns list of ShapeResult, one per detected contour, plus a handful of
    "complex" catch-all boxes for any leftover ink that isn't text or a classified shape.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    # Otsu instead of a fixed 200 cutoff: architectural scans are mostly bimodal
    # (dark ink on a light page), and Otsu picks the split point automatically instead
    # of assuming a clean white background. Cheaper and more robust than adaptive
    # thresholding for this content; doesn't handle uneven lighting/shadows, but that's
    # not the failure mode we're fixing here.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Mask text regions so OCR text regions don't appear as shapes
    for tb in text_bboxes_px:
        x, y, w, h = int(tb["x"]), int(tb["y"]), int(tb["w"]), int(tb["h"])
        cv2.rectangle(binary, (x, y), (x + w, y + h), 0, -1)

    # Crops (below) are cut from this, not from `image`: masking `binary` only keeps
    # text out of the CONTOUR search, it doesn't stop a "complex" shape's bbox from
    # enclosing text and dragging the words along into the embedded picture, so the
    # doc ends up with the text twice (real OCR run + baked into the shape image).
    # Built once up front and reused for every crop below, not patched per-crop.
    crop_source = _erase_text_for_crop(image, text_bboxes_px)

    # ponytail: RETR_CCOMP (not RETR_EXTERNAL) so interior content (inner walls, doors,
    # furniture symbols) inside an outer outline is no longer discarded — that was the
    # critical bug. RETR_CCOMP gives a 2-level hierarchy (outer boundaries / holes) which
    # lets us drop the "hole" contour of a stroke drawn as two parallel lines (e.g. a wall)
    # instead of double-reporting outside+inside edges of the same stroke. Ceiling: this
    # only strips exact holes, not visually-duplicate contours from other causes (e.g.
    # anti-aliasing halos); if that shows up on real scans, dedupe by IoU of bbox instead.
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0] if hierarchy is not None else []

    results = []
    claimed_mask = np.zeros(binary.shape, dtype=np.uint8)
    for idx, cnt in enumerate(contours):
        parent = hierarchy[idx][3] if len(hierarchy) else -1
        if parent != -1 and _is_hole_of_parent(cnt, contours[parent]):
            # This is the inner edge of a stroke whose outer edge we already keep — skip it.
            continue
        if parent != -1 and _is_text_mask_hole(cnt, text_bboxes_px):
            # Our own doing: blanking a text bbox out of `binary` punches a hole in any
            # surrounding solid ink (a colour-filled label box reads as ink under Otsu),
            # and RETR_CCOMP reports that hole as a nested contour. Left in, every word
            # inside a filled box comes back as a phantom "rect" and the document gets an
            # empty rectangle outlined around each label.
            continue

        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        diagonal = math.hypot(w, h)
        if area < MIN_AREA_PX and diagonal < MIN_DIAGONAL_PX:
            # Area alone punishes thin strokes (a 2px x 150px dimension line has area
            # ~300 but is clearly real ink); accept on diagonal/arc-length as a fallback.
            continue

        bbox_px = {"x": x, "y": y, "w": w, "h": h}
        shape_type, confidence = _classify(cnt)

        points = None
        if shape_type == "polyline":
            points = _polyline_points(cnt, bbox_px)
            if len(points) < POLYLINE_MIN_POINTS:
                # Nothing left to draw as a path — better a picture than an empty shape.
                shape_type, points = "complex", None

        crop = crop_source[y:y + h, x:x + w].copy() if shape_type == "complex" else None
        results.append(ShapeResult(
            shape_type=shape_type,
            bbox_px=bbox_px,
            confidence=confidence,
            crop=crop,
            points=points,
        ))
        cv2.drawContours(claimed_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        # ponytail: FILLED alone claims a solid shape's ink, but an open stroke encloses
        # no area, so its own pixels stayed unclaimed and came back a second time as a
        # residual "complex" blob — the same connector emitted twice, once as a path and
        # once as a raster. Stroking the contour claims the ink we actually traced.
        # Ceiling: a fixed 3px pen, so a stroke drawn thicker leaves a thin unclaimed
        # halo; upgrade path is sizing the pen off the contour's minAreaRect short side.
        cv2.drawContours(claimed_mask, [cnt], -1, 255, thickness=CLAIMED_STROKE_PX)

    results.extend(_residual_shapes(crop_source, binary, claimed_mask))
    return results


def _erase_text_for_crop(image: np.ndarray, text_bboxes_px: list[dict]) -> np.ndarray:
    """Return a copy of `image` with every text bbox filled in, for cutting crops from.

    Not plain white: these boxes are often colour-filled (e.g. a UML class-diagram
    header bar), and a white punch-out on a coloured box looks broken. Instead sample
    the median colour of a thin ring just outside the bbox and fill with that, so the
    patch blends into whatever's actually there — white paper stays white, a blue box
    stays blue.
    """
    if not text_bboxes_px:
        return image
    crop_source = image.copy()
    img_h, img_w = image.shape[:2]
    for tb in text_bboxes_px:
        x, y, w, h = int(tb["x"]), int(tb["y"]), int(tb["w"]), int(tb["h"])
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, img_w), min(y + h, img_h)
        if x1 <= x0 or y1 <= y0:
            continue

        ring_x0, ring_y0 = max(x0 - TEXT_ERASE_RING_PX, 0), max(y0 - TEXT_ERASE_RING_PX, 0)
        ring_x1, ring_y1 = min(x1 + TEXT_ERASE_RING_PX, img_w), min(y1 + TEXT_ERASE_RING_PX, img_h)
        ring_region = crop_source[ring_y0:ring_y1, ring_x0:ring_x1]
        ring_mask = np.ones(ring_region.shape[:2], dtype=bool)
        ring_mask[y0 - ring_y0:y1 - ring_y0, x0 - ring_x0:x1 - ring_x0] = False
        ring_pixels = ring_region[ring_mask]
        if ring_pixels.size == 0:
            # ponytail: bbox has no surroundings to sample (e.g. text fills the whole
            # image). Ceiling: leaves the text pixels untouched in that corner case;
            # upgrade path is falling back to the image's own overall median colour.
            continue

        fill_color = np.median(ring_pixels, axis=0)
        crop_source[y0:y1, x0:x1] = fill_color
    return crop_source


def _is_text_mask_hole(cnt, text_bboxes_px: list[dict]) -> bool:
    """True if this hole contour is just the outline of a text bbox we blanked ourselves.

    The traced hole sits a pixel or so outside the rectangle we filled, so this matches on
    all four edges within a small tolerance rather than on containment.
    ponytail: a real rectangle drawn tightly around a label, within TEXT_HOLE_TOL_PX of the
    OCR word box on every side, would also be dropped. OCR boxes hug the glyphs and a drawn
    box has visible padding, so the tolerance keeps them apart in practice. Upgrade path if
    real boxes start vanishing: only skip when the hole has no ink of its own inside it.
    """
    x, y, w, h = cv2.boundingRect(cnt)
    for tb in text_bboxes_px:
        tx, ty, tw, th = int(tb["x"]), int(tb["y"]), int(tb["w"]), int(tb["h"])
        if (
            abs(x - tx) <= TEXT_HOLE_TOL_PX and abs(y - ty) <= TEXT_HOLE_TOL_PX and
            abs((x + w) - (tx + tw)) <= TEXT_HOLE_TOL_PX and
            abs((y + h) - (ty + th)) <= TEXT_HOLE_TOL_PX
        ):
            return True
    return False


def _is_hole_of_parent(cnt, parent_cnt) -> bool:
    """True if cnt's bbox is nearly identical to its parent's — i.e. it's the inner
    edge of the same drawn stroke (double line), not genuinely nested content."""
    x, y, w, h = cv2.boundingRect(cnt)
    px, py, pw, ph = cv2.boundingRect(parent_cnt)
    if pw == 0 or ph == 0:
        return False
    # "Nearly identical" bbox on all four edges (within 15% of parent size)
    tol_w, tol_h = pw * 0.15, ph * 0.15
    return (
        abs(x - px) <= tol_w and abs(y - py) <= tol_h and
        abs((x + w) - (px + pw)) <= tol_w and abs((y + h) - (py + ph)) <= tol_h
    )


def _classify(contour) -> tuple[str, float]:
    rect = cv2.minAreaRect(contour)
    (rw, rh) = rect[1]
    long_side, short_side = max(rw, rh), min(rw, rh)
    # A perfectly thin axis-aligned stroke gives minAreaRect a short side of exactly
    # 0.0 (proven: a 300x1 ink run reports (299.0, 0.0)). Guarding on `short_side > 0`
    # made every hairline fall through to "complex" and get embedded as a raster image
    # one pixel tall. Zero means "1 pixel thick", not "unclassifiable", so floor it —
    # which also keeps the division safe.
    short_side = max(short_side, 1.0)
    if long_side / short_side >= LINE_ASPECT_RATIO and short_side <= LINE_MAX_THICKNESS_PX:
        return "line", 0.9

    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = w * h
    # Extent: ratio of contour area to its bounding box area.
    # Rectangles fill ~100% of their bbox; ellipses fill ~π/4 ≈ 78%.
    extent = cv2.contourArea(contour) / bbox_area if bbox_area > 0 else 1.0

    # ponytail: an open path (a curved connector, a leader, an elbow) is traced out and
    # back, so it encloses ~no area and both extent and solidity collapse toward zero.
    # That's the whole test, and it has to run before the vertex counting below or an
    # L-shaped stroke gets counted as a 3-vertex "triangle". Ceiling: two thresholds
    # tuned on one class of drawing, so a genuinely *filled* sliver (a very thin wedge)
    # would also read as a path. Upgrade path is comparing the contour's arc length
    # against twice its skeleton length to prove the retrace directly.
    if extent < POLYLINE_MAX_EXTENT and _solidity(contour) < POLYLINE_MAX_SOLIDITY:
        return "polyline", 0.8

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        return "triangle", 0.95

    if vertices == 4:
        if extent < 0.85:
            # Low extent means the shape is curved (ellipse/circle), not a flat rect
            circ = _circularity(contour)
            if circ > 0.85:
                return "circle", float(circ)
            return "ellipse", float(circ)
        ar = w / h if h > 0 else 1.0
        if 0.9 <= ar <= 1.1:
            return "square", 0.93
        return "rect", 0.93

    # For 5+ vertices: use circularity + solidity
    circ = _circularity(contour)
    solidity = _solidity(contour)

    if circ > 0.85:
        return "circle", float(circ)
    if circ > 0.80 and solidity > 0.92:
        # High circularity + high solidity → smooth convex curve = ellipse
        return "ellipse", float(circ)
    return "complex", 0.70


def _polyline_points(contour, bbox_px: dict) -> list[tuple[float, float]]:
    """Simplify an open-stroke contour to a short path, as (x, y) fractions of its bbox.

    Fractions, not pixels, because the pipeline maps px -> relative -> EMU
    (layout_reconstructor.to_relative_bbox, word_assembler._emu_coords) and an absolute
    path would not survive that; 0..1 of the shape's own box does.

    ponytail: the contour is the stroke's *outline*, so it walks the path out and back
    and the emitted path retraces itself. That draws correctly (a hairline stroke's two
    sides are the same line) at roughly double the point count. Ceiling: a thick stroke
    renders as its outline rather than its centreline. Upgrade path is skeletonising the
    stroke and emitting the centreline once, with a real width.
    """
    peri = cv2.arcLength(contour, False)
    eps = max(peri * POLYLINE_EPSILON_FRAC, 1.0)
    approx = cv2.approxPolyDP(contour, eps, False)
    for _ in range(8):  # bounded: eps only grows, so the point count only shrinks
        if len(approx) <= POLYLINE_MAX_POINTS:
            break
        eps *= 1.6
        approx = cv2.approxPolyDP(contour, eps, False)

    x, y = bbox_px["x"], bbox_px["y"]
    w, h = max(bbox_px["w"], 1), max(bbox_px["h"], 1)
    points: list[tuple[float, float]] = []
    for px, py in approx.reshape(-1, 2):
        pt = (
            min(max((float(px) - x) / w, 0.0), 1.0),
            min(max((float(py) - y) / h, 0.0), 1.0),
        )
        if not points or pt != points[-1]:
            points.append(pt)
    return points


def _circularity(contour) -> float:
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return 0.0
    return (4 * math.pi * area) / (peri ** 2)


def _solidity(contour) -> float:
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return cv2.contourArea(contour) / hull_area


def _residual_shapes(crop_source: np.ndarray, binary: np.ndarray, claimed_mask: np.ndarray) -> list[ShapeResult]:
    """Catch-all for ink that's neither OCR text nor a contour we classified above.

    This is the fix for "drawings silently vanish": dilate whatever's left, group it
    into connected components, and emit each as a "complex" shape with its crop so the
    word assembler embeds it as an image at the right spot, even though we can't name it.

    crop_source: the text-erased image (see _erase_text_for_crop), not the raw one —
    a residual blob's bbox can enclose text too, same duplication bug as _classify's crop.
    """
    residual = cv2.bitwise_and(binary, cv2.bitwise_not(claimed_mask))
    if not np.any(residual):
        return []

    # ponytail: fixed 9px dilation kernel merges nearby fragments (e.g. a dashed line's
    # dashes, or a symbol's disjoint strokes) into one bbox. Ceiling: a hand-picked
    # constant that doesn't scale with image resolution; upgrade path is to size the
    # kernel off image DPI/dimensions if this misses on very high-res scans.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RESIDUAL_DILATE_PX, RESIDUAL_DILATE_PX))
    dilated = cv2.dilate(residual, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    boxes = []
    for label in range(1, num_labels):  # label 0 is background
        area = stats[label, cv2.CC_STAT_AREA]
        if area < RESIDUAL_MIN_AREA_PX:
            continue
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        boxes.append((area, x, y, w, h))

    # Cap count: keep the biggest blobs first so a noisy scan can't flood the doc.
    boxes.sort(key=lambda b: b[0], reverse=True)
    boxes = boxes[:RESIDUAL_MAX_SHAPES]

    results = []
    for area, x, y, w, h in boxes:
        crop = crop_source[y:y + h, x:x + w].copy()
        results.append(ShapeResult(
            shape_type="complex",
            bbox_px={"x": x, "y": y, "w": w, "h": h},
            confidence=0.5,
            crop=crop,
        ))
    return results
