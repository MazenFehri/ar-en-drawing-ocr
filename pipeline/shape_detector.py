"""Detect non-text graphics and hand them to the document builder as vectors.

Two passes, because a drawing contains two structurally different things:

* **Regions** — a closed outline or a filled blob. `cv2.findContours` traces region
  boundaries, which is exactly the right tool: solidity, extent and vertex count all
  mean something for a closed shape.
* **Strokes** — a wall, a connector, a leader, a dimension line, the four edges of a
  box that other lines happen to touch. A contour *cannot* represent one. Tracing a
  stroke walks the pen out and back, so `minAreaRect` reports a short side of 0.0 and
  `extent` collapses to ~0.001. Worse, the moment a connector touches a box border the
  two become one connected region and the box's contour is a snake around half the page
  — which is why a class diagram with eleven boxes yielded zero rectangles. Strokes
  therefore go through a line-segment detector: detect segments, merge collinear ones,
  assemble primitives from them, claim their ink.

Ink that neither pass claims still falls through to a raster crop (`_residual_shapes`),
so nothing silently vanishes.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np

# ---------------------------------------------------------------- region pass
MIN_AREA_PX = 400  # Ignore contours smaller than 20x20 px
MIN_DIAGONAL_PX = 60  # ...unless the bbox diagonal alone says it's a real (thin) stroke

# Contour area over convex-hull area. A closed outline (a room, a circle, a triangle)
# and a filled blob both sit at ~1.0, because contourArea of an outer boundary ignores
# the hole inside it. A stroke, or a box fused to the connectors touching it, is a thin
# snake whose hull is enormous next to its own area — measured at 0.001-0.12 on
# class-diagram.png's connectors. 0.6 is comfortably between the two populations.
# This is the gate that decides "contours can describe this" vs "hand it to the
# segment detector", and it is the whole fix for the fused-box case.
REGION_MIN_SOLIDITY = 0.60

# The other half of the same gate, measured on the region's own extent rather than its
# shape: an ink run this thin has no interior to describe, whatever its solidity says.
# A 300x5 dimension bar is solid and convex, so solidity alone happily calls it a region
# and then approxPolyDP — whose epsilon is 4% of a 610px perimeter, i.e. five times the
# bar's own height — collapses it to two vertices and it lands in "complex". Routing on
# minAreaRect's short side retires the old `short_side > 0` special case as well: a 1px
# hairline reports exactly 0.0, which is simply below the floor, not a division hazard.
REGION_MIN_THICKNESS_PX = 14

TEXT_ERASE_RING_PX = 6  # ring width sampled just outside a text bbox for its fill colour
TEXT_HOLE_TOL_PX = 4  # how closely a hole contour must hug a masked text bbox to be ours

# --------------------------------------------------------------- segment pass
# EdgeDrawing over LSD: measured on class-diagram.png inside the production image,
# EdgeDrawing returned 330 segments in 33 ms against LSD's 385 in 90 ms, and throws in
# ellipse detection for free. Research's warning about 122 spurious ellipses came from
# the raw file; on the *preprocessed* page (denoised, CLAHE'd, text erased) the same
# parameters find 1. LSD stays as the fallback for an environment whose cv2 has no
# ximgproc — see _detect_segments.
ED_GRADIENT_THRESHOLD = 36
ED_ANCHOR_THRESHOLD = 8
ED_MIN_PATH_LENGTH = 20
ED_MIN_LINE_LENGTH = 10
ED_LINE_FIT_ERROR = 1.4
ED_MAX_ERROR = 1.3
ED_MAX_LINE_GAP = 6.0

# A stroke has two sides, so every drawn edge comes back as a pair of near-coincident
# segments; a dashed or partly-occluded edge comes back in fragments. Merging on
# (angle, perpendicular offset) collapses both cases into one segment, which is what
# turns four hairlines back into one box edge.
MERGE_ANGLE_TOL_DEG = 3.0
# 9 px, not the ~2 px a hairline stroke is drawn at: EdgeDrawing fits a line to a run of
# anchor points, and on class-diagram.png the two fitted sides of one 2 px box border
# came back 7.0 px apart. Anything under this is one edge.
MERGE_OFFSET_TOL_PX = 9.0
# 25 px, sized off what actually interrupts an edge in these drawings: an arrowhead
# landing on a box border knocks a ~14 px hole in it, and the border either side has to
# rejoin or the box has no corner to assemble from. Ceiling: two genuinely separate
# collinear strokes closer than this (a dimension line broken around its own text)
# become one. Upgrade path is requiring ink along the bridged span before joining.
MERGE_GAP_TOL_PX = 25.0
MIN_SEGMENT_PX = 18.0  # shorter than this is glyph residue or a corner nub

RECT_PERP_TOL_DEG = 7.0  # how far from 90 deg two sides may meet
RECT_PARALLEL_TOL_DEG = 7.0  # how far from parallel opposite sides may be
# 20 px: an arrow converging on a box corner eats the last stretch of the border, so the
# detected side stops short of where the corner geometrically is. Measured worst case on
# class-diagram.png was 15 px (the User box, six arrows landing on one edge).
RECT_CORNER_TOL_PX = 20.0
RECT_MIN_SIDE_PX = 24.0
RECT_MIN_FILL = 0.85  # quad area / minAreaRect area — rejects skewed near-rectangles
RECT_MAX_IOU = 0.55  # suppress near-duplicate rectangles
RECT_AXIS_TOL_DEG = 2.5  # beyond this a rectangle is emitted as a rotated freeform

AXIS_TOL_DEG = 2.5  # how near horizontal/vertical a segment counts as axis-aligned
CHAIN_JOIN_TOL_PX = 10.0  # endpoints closer than this are the same corner
POLYLINE_MAX_POINTS = 64  # cap so one noisy squiggle can't emit a thousand-point path

ELLIPSE_MIN_AXIS_PX = 12.0  # below this it's noise, not a drawn circle
ELLIPSE_MAX_IOU = 0.5  # the two sides of one drawn circle are two fits of the same circle
SEGMENT_CLAIMED_FRAC = 0.7  # a segment this covered by already-claimed ink is a duplicate

# ponytail: pipeline/preprocessor.py already caps the longest side at MAX_DIM_PX = 2000
# and shape detection only ever sees its output, so in the pipeline this resize is a
# no-op — it exists for direct callers. Detection at 2000 px was measured at 33 ms;
# research measured full-res LSD at 603 ms, so the cap matters if one ever arrives.
# Ceiling: a fixed pixel budget rather than one derived from stroke width or DPI.
SEGMENT_MAX_DIM_PX = 2000

CLAIMED_STROKE_PX = 5  # pen width used to mark an emitted primitive's own ink as claimed
CLAIM_DILATE_PX = 5  # grown once more before subtracting, to absorb anti-aliasing halo

RESIDUAL_DILATE_PX = 9  # merge nearby ink fragments into one blob before componentizing
RESIDUAL_MIN_AREA_PX = 150  # noise floor for residual (unclassified) ink blobs
RESIDUAL_MAX_SHAPES = 40  # hard cap so a noisy scan can't spam hundreds of crops


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


@dataclass
class _Seg:
    """A straight segment in image pixels, with the derived values used everywhere."""
    x1: float
    y1: float
    x2: float
    y2: float
    theta: float = field(init=False)   # direction mod pi, so a segment and its reverse match
    length: float = field(init=False)

    def __post_init__(self):
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        self.theta = math.atan2(dy, dx) % math.pi
        self.length = math.hypot(dx, dy)

    @property
    def p1(self) -> tuple[float, float]:
        return (self.x1, self.y1)

    @property
    def p2(self) -> tuple[float, float]:
        return (self.x2, self.y2)


def detect_shapes(image: np.ndarray, text_bboxes_px: list[dict]) -> list[ShapeResult]:
    """Detect non-text shapes in an image.

    text_bboxes_px: list of {x, y, w, h} dicts in pixel coords — these regions are masked.
    Returns list of ShapeResult: closed regions classified from their contours, strokes
    assembled from detected line segments, plus a handful of "complex" catch-all boxes
    for any leftover ink that is neither text nor a shape we could name.
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
    # text out of the shape search, it doesn't stop a "complex" shape's bbox from
    # enclosing text and dragging the words along into the embedded picture, so the
    # doc ends up with the text twice (real OCR run + baked into the shape image).
    # Built once up front and reused for every crop below, not patched per-crop.
    crop_source = _erase_text_for_crop(image, text_bboxes_px)

    results: list[ShapeResult] = []
    claimed_mask = np.zeros(binary.shape, dtype=np.uint8)

    # Pass 1 — regions. Anything a contour can honestly describe.
    results.extend(_region_shapes(binary, crop_source, text_bboxes_px, claimed_mask))

    # Pass 2 — strokes. The segment detector runs on the *text-erased* image rather
    # than on `binary`: blanking a text box out of a binary mask leaves a hard-edged
    # rectangle behind, and an edge detector would dutifully assemble that rectangle
    # into a phantom box around every label. _erase_text_for_crop fills with the
    # surrounding median instead, so the patch has no gradient to detect.
    results.extend(_stroke_shapes(crop_source, claimed_mask))

    # Pass 3 — whatever is still unclaimed becomes a picture rather than vanishing.
    results.extend(_residual_shapes(crop_source, binary, claimed_mask))
    return results


# --------------------------------------------------------------------------- regions


def _region_shapes(
    binary: np.ndarray,
    crop_source: np.ndarray,
    text_bboxes_px: list[dict],
    claimed_mask: np.ndarray,
) -> list[ShapeResult]:
    """Closed outlines and filled blobs, classified from their contours."""
    # ponytail: RETR_CCOMP (not RETR_EXTERNAL) so interior content (inner walls, doors,
    # furniture symbols) inside an outer outline is no longer discarded — that was the
    # critical bug. RETR_CCOMP gives a 2-level hierarchy (outer boundaries / holes) which
    # lets us drop the "hole" contour of a stroke drawn as two parallel lines (e.g. a wall)
    # instead of double-reporting outside+inside edges of the same stroke. Ceiling: this
    # only strips exact holes, not visually-duplicate contours from other causes (e.g.
    # anti-aliasing halos); if that shows up on real scans, dedupe by IoU of bbox instead.
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0] if hierarchy is not None else []

    results: list[ShapeResult] = []
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

        if _solidity(cnt) < REGION_MIN_SOLIDITY or _min_thickness(cnt) < REGION_MIN_THICKNESS_PX:
            # Not a region: a stroke, or a shape fused to the strokes touching it.
            # Leave the ink unclaimed — the segment pass owns it.
            continue

        shape_type, confidence = _classify_region(cnt)
        crop = crop_source[y:y + h, x:x + w].copy() if shape_type == "complex" else None
        results.append(ShapeResult(
            shape_type=shape_type,
            bbox_px={"x": x, "y": y, "w": w, "h": h},
            confidence=confidence,
            crop=crop,
        ))
        cv2.drawContours(claimed_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        # FILLED alone claims a solid shape's ink, but a hollow outline encloses its own
        # stroke rather than covering it, so the boundary pixels stayed unclaimed and came
        # back a second time as a residual "complex" blob. Stroking claims what we traced.
        cv2.drawContours(claimed_mask, [cnt], -1, 255, thickness=CLAIMED_STROKE_PX)
    return results


def _classify_region(contour) -> tuple[str, float]:
    """Name a closed region. Strokes never reach here — see REGION_MIN_SOLIDITY."""
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = w * h
    # Extent: ratio of contour area to its bounding box area.
    # Rectangles fill ~100% of their bbox; ellipses fill ~pi/4 ~= 78%.
    extent = cv2.contourArea(contour) / bbox_area if bbox_area > 0 else 1.0

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


# --------------------------------------------------------------------------- strokes


def _stroke_shapes(crop_source: np.ndarray, claimed_mask: np.ndarray) -> list[ShapeResult]:
    """Rectangles, lines and open paths, assembled from detected line segments."""
    gray = (
        cv2.cvtColor(crop_source, cv2.COLOR_BGR2GRAY)
        if len(crop_source.shape) == 3 else crop_source
    )
    segments, ellipses = _detect_segments(gray)

    results: list[ShapeResult] = []

    # Ellipses first: their arcs also come back as short segments, and claiming the
    # ellipse's ink lets the segment filter below drop those as duplicates.
    emitted: list[dict] = []
    for shape_type, bbox, conf, draw in ellipses:
        if _claimed_fraction_bbox(bbox, claimed_mask) > SEGMENT_CLAIMED_FRAC:
            continue
        # A drawn circle has two sides and the detector fits one to each — measured on
        # sample_drawing.png, r=90 and r=86 around the same centre. Without this the
        # document gets two concentric circles where the page has one.
        if any(_iou(bbox, prev) > ELLIPSE_MAX_IOU for prev in emitted):
            continue
        emitted.append(bbox)
        results.append(ShapeResult(shape_type=shape_type, bbox_px=bbox, confidence=conf))
        draw(claimed_mask)

    merged = _merge_collinear(segments)
    merged = [s for s in merged if not _is_claimed(s, claimed_mask)]

    rects, used = _assemble_rectangles(merged)
    for shape in rects:
        results.append(shape)
        _claim_points(claimed_mask, shape, closed=True)

    remaining = [s for i, s in enumerate(merged) if i not in used]
    for shape in _chain_shapes(remaining):
        results.append(shape)
        _claim_points(claimed_mask, shape, closed=False)

    return results


# ponytail: opencv-python, opencv-contrib-python and opencv-python-headless all unpack
# into the same site-packages/cv2 and paddleocr requires two of them, so which build
# answers `import cv2` comes down to install order (the Dockerfile forces contrib to win,
# and asserts it). If a plain build ever lands on top, ximgproc disappears and this flag
# routes detection to LSD instead of crashing: measured on class-diagram.png, LSD found
# 385 segments in 90 ms against EdgeDrawing's 330 in 33 ms, so it is slower and noisier
# but not broken. Ceiling: no ellipse detection on that path, so circles fall through to
# a raster crop. Module-level and not inlined so tests can flip it and exercise both.
HAVE_EDGE_DRAWING = hasattr(cv2, "ximgproc")


def _edge_drawing_params():
    """cv2 renamed this class between releases; both spellings ship in 4.x."""
    try:
        return cv2.ximgproc.EdgeDrawing.Params()
    except AttributeError:
        return cv2.ximgproc_EdgeDrawing_Params()


def _detect_segments(gray: np.ndarray):
    """Return (segments, ellipses) in the *caller's* pixel coordinates.

    ellipses is a list of (shape_type, bbox_px, confidence, draw_fn) where draw_fn
    strokes the ellipse into a mask, so the caller can claim its ink without needing
    to know the geometry.
    """
    h, w = gray.shape[:2]
    scale = 1.0
    src = gray
    longest = max(h, w)
    if longest > SEGMENT_MAX_DIM_PX:
        scale = SEGMENT_MAX_DIM_PX / longest
        src = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    inv = 1.0 / scale

    src = np.ascontiguousarray(src)
    segs, ells = _edge_drawing(src) if HAVE_EDGE_DRAWING else _lsd(src)

    segments = [_Seg(a * inv, b * inv, c * inv, d * inv) for a, b, c, d in segs]
    ellipses = [_ellipse_shape(e, inv) for e in ells]
    return segments, [e for e in ellipses if e is not None]


def _edge_drawing(src: np.ndarray):
    ed = cv2.ximgproc.createEdgeDrawing()
    params = _edge_drawing_params()
    params.PFmode = False
    params.EdgeDetectionOperator = cv2.ximgproc.EDGE_DRAWING_SOBEL
    params.GradientThresholdValue = ED_GRADIENT_THRESHOLD
    params.AnchorThresholdValue = ED_ANCHOR_THRESHOLD
    params.MinPathLength = ED_MIN_PATH_LENGTH
    params.MinLineLength = ED_MIN_LINE_LENGTH
    params.NFAValidation = True
    params.Sigma = 1.0
    params.LineFitErrorThreshold = ED_LINE_FIT_ERROR
    params.MaxErrorThreshold = ED_MAX_ERROR
    params.MaxDistanceBetweenTwoLines = ED_MAX_LINE_GAP
    ed.setParams(params)
    ed.detectEdges(src)

    lines = ed.detectLines()
    segs = [] if lines is None else [tuple(float(v) for v in l) for l in lines.reshape(-1, 4)]
    ells = ed.detectEllipses()
    ells = [] if ells is None else [[float(v) for v in e] for e in ells.reshape(-1, 6)]
    return segs, ells


def _lsd(src: np.ndarray):
    lines = cv2.createLineSegmentDetector().detect(src)[0]
    segs = [] if lines is None else [tuple(float(v) for v in l) for l in lines.reshape(-1, 4)]
    return segs, []


def _ellipse_shape(row: list[float], inv: float):
    """EdgeDrawing packs a result as (cx, cy, r, a, b, angle): r is 0 for an ellipse and
    the radius for a circle, and the real semi-axes are (r + a, r + b)."""
    cx, cy, r, a, b, angle = row
    ax, by = (r + a) * inv, (r + b) * inv
    if min(ax, by) < ELLIPSE_MIN_AXIS_PX:
        return None
    rad = math.radians(angle)
    # Half-extents of the axis-aligned box around a rotated ellipse.
    hw = math.hypot(ax * math.cos(rad), by * math.sin(rad))
    hh = math.hypot(ax * math.sin(rad), by * math.cos(rad))
    cx, cy = cx * inv, cy * inv
    bbox = {
        "x": int(round(cx - hw)), "y": int(round(cy - hh)),
        "w": max(int(round(2 * hw)), 1), "h": max(int(round(2 * hh)), 1),
    }
    shape_type = "circle" if r > 0 else "ellipse"

    def draw(mask: np.ndarray):
        cv2.ellipse(
            mask, (int(round(cx)), int(round(cy))), (int(round(ax)), int(round(by))),
            angle, 0, 360, 255, CLAIMED_STROKE_PX,
        )

    return shape_type, bbox, 0.85, draw


# ponytail: EdgeDrawing's detectEllipses only ever returns whole circles/ellipses
# (it is EDCircles underneath), never partial arcs, so there is nothing an "arc"
# element type would carry that "ellipse" does not. Upgrade path if arcs matter
# (door swings in a floor plan are the obvious case): fit a circle to a chained
# stroke's points and emit start/sweep angles alongside the bbox.


def _merge_collinear(segments: list[_Seg]) -> list[_Seg]:
    """Collapse near-coincident and collinear-with-a-gap segments into single segments.

    Every drawn edge has two sides, so an edge detector reports it twice; a dashed or
    partly-occluded edge arrives in fragments. Both are the same operation: cluster on
    (angle, perpendicular offset), then merge the clusters' projections along the line.

    ponytail: greedy single-pass clustering seeded longest-first, so the dominant
    segment anchors each cluster and short noise attaches to it rather than the other
    way round. Ceiling: a genuinely curved stroke whose chord happens to stay inside
    MERGE_OFFSET_TOL_PX gets straightened. Upgrade path is splitting a cluster whose
    residual error exceeds the tolerance instead of accepting it whole.
    """
    clusters: list[dict] = []
    for s in sorted(segments, key=lambda s: -s.length):
        placed = False
        for c in clusters:
            if _angle_diff(s.theta, c["theta"]) > math.radians(MERGE_ANGLE_TOL_DEG):
                continue
            nx, ny = -math.sin(c["theta"]), math.cos(c["theta"])
            d1 = abs((s.x1 - c["ax"]) * nx + (s.y1 - c["ay"]) * ny)
            d2 = abs((s.x2 - c["ax"]) * nx + (s.y2 - c["ay"]) * ny)
            if max(d1, d2) > MERGE_OFFSET_TOL_PX:
                continue
            weight = c["weight"]
            mx, my = (s.x1 + s.x2) / 2, (s.y1 + s.y2) / 2
            c["ax"] = (c["ax"] * weight + mx * s.length) / (weight + s.length)
            c["ay"] = (c["ay"] * weight + my * s.length) / (weight + s.length)
            # Circular mean of an *undirected* angle: double it, average, halve back.
            c["sin2"] += s.length * math.sin(2 * s.theta)
            c["cos2"] += s.length * math.cos(2 * s.theta)
            c["theta"] = (0.5 * math.atan2(c["sin2"], c["cos2"])) % math.pi
            c["weight"] = weight + s.length
            c["members"].append(s)
            placed = True
            break
        if not placed:
            clusters.append({
                "ax": (s.x1 + s.x2) / 2, "ay": (s.y1 + s.y2) / 2,
                "theta": s.theta, "weight": s.length,
                "sin2": s.length * math.sin(2 * s.theta),
                "cos2": s.length * math.cos(2 * s.theta),
                "members": [s],
            })

    out: list[_Seg] = []
    for c in clusters:
        dx, dy = math.cos(c["theta"]), math.sin(c["theta"])
        spans = []
        for s in c["members"]:
            t1 = (s.x1 - c["ax"]) * dx + (s.y1 - c["ay"]) * dy
            t2 = (s.x2 - c["ax"]) * dx + (s.y2 - c["ay"]) * dy
            spans.append((min(t1, t2), max(t1, t2)))
        spans.sort()
        lo, hi = spans[0]
        for a, b in spans[1:]:
            if a <= hi + MERGE_GAP_TOL_PX:
                hi = max(hi, b)
            else:
                out.append(_span_to_seg(c, dx, dy, lo, hi))
                lo, hi = a, b
        out.append(_span_to_seg(c, dx, dy, lo, hi))
    return [s for s in out if s.length >= MIN_SEGMENT_PX]


def _span_to_seg(cluster: dict, dx: float, dy: float, lo: float, hi: float) -> _Seg:
    ax, ay = cluster["ax"], cluster["ay"]
    return _Seg(ax + dx * lo, ay + dy * lo, ax + dx * hi, ay + dy * hi)


def _assemble_rectangles(segs: list[_Seg]) -> tuple[list[ShapeResult], set[int]]:
    """Build rectangles out of two parallel pairs of segments that meet at four corners.

    This is the part a contour cannot do: the four edges of a box stay four edges even
    when a connector line fuses the box to half the page, because they are found and
    paired individually.
    """
    n = len(segs)
    if n < 4:
        return [], set()

    perp_tol = math.radians(RECT_PERP_TOL_DEG)
    par_tol = math.radians(RECT_PARALLEL_TOL_DEG)

    corners: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(_angle_diff(segs[i].theta, segs[j].theta) - math.pi / 2) > perp_tol:
                continue
            pt = _intersection(segs[i], segs[j])
            if pt is None or not _near_end(segs[i], pt) or not _near_end(segs[j], pt):
                continue
            corners[i][j] = pt
            corners[j][i] = pt

    candidates = []
    seen: set[frozenset] = set()
    for i in range(n):
        for k in range(i + 1, n):
            if _angle_diff(segs[i].theta, segs[k].theta) > par_tol:
                continue
            shared = sorted(set(corners[i]) & set(corners[k]))
            for a in range(len(shared)):
                for b in range(a + 1, len(shared)):
                    j, l = shared[a], shared[b]
                    if _angle_diff(segs[j].theta, segs[l].theta) > par_tol:
                        continue
                    members = frozenset((i, j, k, l))
                    if len(members) != 4 or members in seen:
                        continue
                    quad = [corners[i][j], corners[j][k], corners[k][l], corners[l][i]]
                    box = _validate_quad(quad)
                    if box is None:
                        continue
                    seen.add(members)
                    candidates.append({
                        "members": members, "quad": quad, "box": box,
                        "area": box[1][0] * box[1][1],
                        "pairs": (frozenset((i, k)), frozenset((j, l))),
                    })

    if not candidates:
        return [], set()

    # A UML class box, a titleblock and a table all share one pair of long sides across
    # several rectangles (the box plus each compartment). Keep only the largest per
    # shared pair: the outer box becomes a rectangle, the dividers stay lines.
    best: dict[frozenset, float] = {}
    for c in candidates:
        for p in c["pairs"]:
            best[p] = max(best.get(p, 0.0), c["area"])
    kept = [c for c in candidates if all(c["area"] >= best[p] - 1e-6 for p in c["pairs"])]
    kept.sort(key=lambda c: -c["area"])

    results: list[ShapeResult] = []
    used: set[int] = set()
    accepted_boxes: list[dict] = []
    for c in kept:
        bbox = _quad_bbox(c["quad"])
        if any(_iou(bbox, prev) > RECT_MAX_IOU for prev in accepted_boxes):
            continue
        accepted_boxes.append(bbox)
        used |= set(c["members"])
        angle = _axis_offset_deg(c["box"][2])
        if angle <= RECT_AXIS_TOL_DEG:
            ar = bbox["w"] / bbox["h"] if bbox["h"] else 1.0
            results.append(ShapeResult(
                shape_type="square" if 0.9 <= ar <= 1.1 else "rect",
                bbox_px=bbox, confidence=0.93,
            ))
        else:
            # ponytail: ShapeResult carries a bbox and a type, with no rotation field,
            # so an axis-aligned "rect" would be a lie about a tilted box. Emitted as a
            # closed freeform instead, which draws exactly right through the existing
            # PolylineShapeElement path. Ceiling: Word sees a generic path, not a
            # rectangle, so shape-aware tooling downstream can't tell it is one;
            # upgrade path is a rotation field on ShapeResult and an a:xfrm rot.
            results.append(ShapeResult(
                shape_type="polyline", bbox_px=bbox, confidence=0.9,
                points=_to_bbox_fractions(c["quad"] + [c["quad"][0]], bbox),
            ))
    return results, used


def _validate_quad(quad: list[tuple[float, float]]):
    """Reject four corners that don't actually enclose a rectangle. Returns minAreaRect."""
    for a, b in zip(quad, quad[1:] + quad[:1]):
        if math.dist(a, b) < RECT_MIN_SIDE_PX:
            return None
    pts = np.array(quad, dtype=np.float32)
    box = cv2.minAreaRect(pts)
    rect_area = box[1][0] * box[1][1]
    if rect_area <= 0:
        return None
    if cv2.contourArea(pts) / rect_area < RECT_MIN_FILL:
        return None
    return box


def _chain_shapes(segs: list[_Seg]) -> list[ShapeResult]:
    """Walk segments that share endpoints into paths, then emit each as a shape.

    An elbow connector is two segments, a curved association is a dozen; both are one
    stroke and belong in the document as one path, not as loose pieces.
    """
    nodes: list[list] = []  # [x, y, [(seg_index, end)]]

    def node_for(p: tuple[float, float]) -> int:
        for idx, node in enumerate(nodes):
            if math.dist((node[0], node[1]), p) <= CHAIN_JOIN_TOL_PX:
                return idx
        nodes.append([p[0], p[1], []])
        return len(nodes) - 1

    ends: list[tuple[int, int]] = []
    for i, s in enumerate(segs):
        a, b = node_for(s.p1), node_for(s.p2)
        nodes[a][2].append((i, 0))
        nodes[b][2].append((i, 1))
        ends.append((a, b))

    results: list[ShapeResult] = []
    visited: set[int] = set()

    def other(node_idx: int, seg_idx: int) -> int:
        a, b = ends[seg_idx]
        return b if a == node_idx else a

    def walk(start_node: int, seg_idx: int) -> list[tuple[float, float]]:
        path = [(nodes[start_node][0], nodes[start_node][1])]
        node, seg = start_node, seg_idx
        while True:
            visited.add(seg)
            node = other(node, seg)
            path.append((nodes[node][0], nodes[node][1]))
            nxt = [s for s, _ in nodes[node][2] if s != seg and s not in visited]
            if len(nodes[node][2]) != 2 or len(nxt) != 1:
                break
            seg = nxt[0]
        return path

    # Open chains first, from every junction/endpoint; whatever is left is a closed loop.
    for node_idx, node in enumerate(nodes):
        if len(node[2]) == 2:
            continue
        for seg_idx, _ in list(node[2]):
            if seg_idx not in visited:
                results.append(_path_shape(walk(node_idx, seg_idx)))
    for seg_idx in range(len(segs)):
        if seg_idx not in visited:
            results.append(_path_shape(walk(ends[seg_idx][0], seg_idx)))
    return [r for r in results if r is not None]


def _path_shape(path: list[tuple[float, float]]) -> Optional[ShapeResult]:
    path = _simplify(path)
    if len(path) < 2:
        return None
    bbox = _quad_bbox(path)
    if len(path) == 2 and _is_drawable_as_line(path[0], path[1]):
        return ShapeResult(shape_type="line", bbox_px=bbox, confidence=0.9)
    return ShapeResult(
        shape_type="polyline", bbox_px=bbox, confidence=0.85,
        points=_to_bbox_fractions(path, bbox),
    )


def _is_drawable_as_line(p1: tuple[float, float], p2: tuple[float, float]) -> bool:
    """True if DrawingML's `line` preset would draw this segment correctly.

    prstGeom="line" always runs from the shape box's top-left corner to its
    bottom-right, so it can only express an axis-aligned segment or one leaning "\".
    A "/" segment has the same bounding box and would render mirrored — that one goes
    out as a two-point freeform instead, which is exact.
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    theta = math.degrees(math.atan2(dy, dx)) % 180.0
    if min(theta, abs(theta - 90.0), abs(theta - 180.0)) <= AXIS_TOL_DEG:
        return True
    return dx * dy > 0  # both increase together => top-left to bottom-right


def _simplify(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop duplicate points and cap the point count, so one squiggle can't emit a
    thousand-point path."""
    pts: list[tuple[float, float]] = []
    for p in path:
        if not pts or math.dist(p, pts[-1]) > 0.5:
            pts.append(p)
    if len(pts) <= POLYLINE_MAX_POINTS:
        return pts
    arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    eps = max(cv2.arcLength(arr, False) * 0.005, 1.0)
    for _ in range(8):  # bounded: eps only grows, so the point count only shrinks
        approx = cv2.approxPolyDP(arr, eps, False).reshape(-1, 2)
        if len(approx) <= POLYLINE_MAX_POINTS:
            break
        eps *= 1.6
    return [(float(x), float(y)) for x, y in approx]


# ----------------------------------------------------------------------- geometry


def _angle_diff(a: float, b: float) -> float:
    """Smallest angle between two *undirected* directions, in [0, pi/2]."""
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def _intersection(s1: _Seg, s2: _Seg) -> Optional[tuple[float, float]]:
    d1x, d1y = s1.x2 - s1.x1, s1.y2 - s1.y1
    d2x, d2y = s2.x2 - s2.x1, s2.y2 - s2.y1
    den = d1x * d2y - d1y * d2x
    if abs(den) < 1e-9:
        return None
    t = ((s2.x1 - s1.x1) * d2y - (s2.y1 - s1.y1) * d2x) / den
    return (s1.x1 + t * d1x, s1.y1 + t * d1y)


def _near_end(s: _Seg, pt: tuple[float, float]) -> bool:
    """True if pt sits within RECT_CORNER_TOL_PX of one of s's endpoints, measured
    along the segment (so a corner just past the end still counts, an overshooting
    T-junction in the middle does not)."""
    if s.length == 0:
        return False
    dx, dy = (s.x2 - s.x1) / s.length, (s.y2 - s.y1) / s.length
    t = (pt[0] - s.x1) * dx + (pt[1] - s.y1) * dy
    perp = abs(-(pt[0] - s.x1) * dy + (pt[1] - s.y1) * dx)
    if perp > RECT_CORNER_TOL_PX:
        return False
    if t < -RECT_CORNER_TOL_PX or t > s.length + RECT_CORNER_TOL_PX:
        return False
    return min(abs(t), abs(s.length - t)) <= RECT_CORNER_TOL_PX


def _axis_offset_deg(angle: float) -> float:
    """How far a minAreaRect angle is from axis-aligned, in degrees (0..45)."""
    a = abs(angle) % 90.0
    return min(a, 90.0 - a)


def _quad_bbox(pts: list[tuple[float, float]]) -> dict:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x, y = int(math.floor(min(xs))), int(math.floor(min(ys)))
    w = max(int(math.ceil(max(xs))) - x, 1)
    h = max(int(math.ceil(max(ys))) - y, 1)
    return {"x": x, "y": y, "w": w, "h": h}


def _to_bbox_fractions(pts: list[tuple[float, float]], bbox: dict) -> list[tuple[float, float]]:
    """Fractions, not pixels, because the pipeline maps px -> relative -> EMU
    (layout_reconstructor.to_relative_bbox, word_assembler._emu_coords) and an absolute
    path would not survive that; 0..1 of the shape's own box does."""
    x, y = bbox["x"], bbox["y"]
    w, h = max(bbox["w"], 1), max(bbox["h"], 1)
    out: list[tuple[float, float]] = []
    for px, py in pts:
        pt = (
            min(max((px - x) / w, 0.0), 1.0),
            min(max((py - y) / h, 0.0), 1.0),
        )
        if not out or pt != out[-1]:
            out.append(pt)
    return out


def _iou(a: dict, b: dict) -> float:
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _circularity(contour) -> float:
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return 0.0
    return (4 * math.pi * area) / (peri ** 2)


def _min_thickness(contour) -> float:
    """Short side of the contour's minimum-area rectangle — how thin the ink run is."""
    (_, (w, h), _) = cv2.minAreaRect(contour)
    return min(w, h)


def _solidity(contour) -> float:
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return cv2.contourArea(contour) / hull_area


# -------------------------------------------------------------------------- claiming


def _is_claimed(s: _Seg, claimed_mask: np.ndarray) -> bool:
    """True if most of this segment already lies on ink some earlier pass claimed."""
    steps = max(int(s.length / 4), 4)
    hits = 0
    h, w = claimed_mask.shape[:2]
    for i in range(steps + 1):
        t = i / steps
        x = int(round(s.x1 + (s.x2 - s.x1) * t))
        y = int(round(s.y1 + (s.y2 - s.y1) * t))
        if 0 <= x < w and 0 <= y < h and claimed_mask[y, x]:
            hits += 1
    return hits / (steps + 1) > SEGMENT_CLAIMED_FRAC


def _claimed_fraction_bbox(bbox: dict, claimed_mask: np.ndarray) -> float:
    h, w = claimed_mask.shape[:2]
    x0, y0 = max(bbox["x"], 0), max(bbox["y"], 0)
    x1, y1 = min(bbox["x"] + bbox["w"], w), min(bbox["y"] + bbox["h"], h)
    if x1 <= x0 or y1 <= y0:
        return 1.0
    patch = claimed_mask[y0:y1, x0:x1]
    return float(np.count_nonzero(patch)) / patch.size


def _claim_points(claimed_mask: np.ndarray, shape: ShapeResult, closed: bool) -> None:
    """Stroke an emitted primitive's own ink into the claimed mask.

    A stroke encloses no area, so filling is useless here — the pen is the point. Any
    ink we drew over must be claimed or the residual pass emits it a second time as a
    raster crop of the very shape we just vectorised.
    """
    b = shape.bbox_px
    if shape.points:
        pts = [(b["x"] + px * b["w"], b["y"] + py * b["h"]) for px, py in shape.points]
    else:
        pts = [(b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"])]
        if shape.shape_type != "line":
            pts = [
                (b["x"], b["y"]), (b["x"] + b["w"], b["y"]),
                (b["x"] + b["w"], b["y"] + b["h"]), (b["x"], b["y"] + b["h"]),
            ]
            closed = True
    arr = np.array([[int(round(x)), int(round(y))] for x, y in pts], dtype=np.int32)
    cv2.polylines(claimed_mask, [arr], closed, 255, CLAIMED_STROKE_PX)


# --------------------------------------------------------------------------- shared


def _erase_text_for_crop(image: np.ndarray, text_bboxes_px: list[dict]) -> np.ndarray:
    """Return a copy of `image` with every text bbox filled in, for cutting crops from.

    Not plain white: these boxes are often colour-filled (e.g. a UML class-diagram
    header bar), and a white punch-out on a coloured box looks broken. Instead sample
    the median colour of a thin ring just outside the bbox and fill with that, so the
    patch blends into whatever's actually there — white paper stays white, a blue box
    stays blue. The segment detector reads this image too, and blending matters twice
    as much there: a flat punch-out would give it four crisp edges to assemble into a
    phantom rectangle around every word on the page.
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


def _residual_shapes(crop_source: np.ndarray, binary: np.ndarray, claimed_mask: np.ndarray) -> list[ShapeResult]:
    """Catch-all for ink that's neither OCR text nor a shape we named above.

    This is the fix for "drawings silently vanish": dilate whatever's left, group it
    into connected components, and emit each as a "complex" shape with its crop so the
    word assembler embeds it as an image at the right spot, even though we can't name it.
    Photographs, logos and rendered charts are supposed to arrive here — the complaint
    was never that this pass exists, only that line art was reaching it.

    crop_source: the text-erased image (see _erase_text_for_crop), not the raw one —
    a residual blob's bbox can enclose text too, same duplication bug as the crops above.
    """
    # ponytail: the claim pens above are centred on geometry fitted to the ink, so they
    # can leave a pixel or two of anti-aliased halo on either side. Growing the claim
    # once here is cheaper and steadier than widening every pen. Ceiling: a genuinely
    # separate mark within CLAIM_DILATE_PX of a detected stroke is absorbed with it.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLAIM_DILATE_PX, CLAIM_DILATE_PX))
    residual = cv2.bitwise_and(binary, cv2.bitwise_not(cv2.dilate(claimed_mask, kernel)))
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
