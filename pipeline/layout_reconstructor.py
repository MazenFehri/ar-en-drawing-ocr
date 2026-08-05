from statistics import median

from models.elements import (
    BBox, TextElement, SimpleShapeElement, PolylineShapeElement, ComplexShapeElement, Element,
)
from pipeline.ocr import OcrWord
from pipeline.shape_detector import ShapeResult
from utils.bidi import detect_language, is_arabic


# Median detected text height below which recognition should not be trusted.
#
# Low-DPI scans don't degrade gracefully in Arabic, they fail in one specific way: the
# dots that tell ق from ف and خ from ف are 1-2 pixels at this size, so they are never
# sampled and the letter becomes a coin flip. Measured 15px and 14px on two ~45 DPI
# worksheets against 27px on a drawing that reads correctly, so 20 separates them.
#
# ponytail: height in pixels stands in for DPI, which the upload does not tell us —
# fine while pages are whole scanned sheets, wrong for a tight crop of one large label.
# Upgrade path if that turns up: divide by the page's longest side instead of using an
# absolute count.
LOW_RESOLUTION_TEXT_HEIGHT_PX = 20.0


# A marking column — the strip of small boxes down the edge of an exam page where a teacher
# writes per-question marks — is not part of the prose, but it shares its vertical extent
# with the prose. Reading order groups words into lines by vertical centre, so without this
# those cells land *between* the words of the sentences beside them. Measured on test1.jpeg:
# ten cells at x=328 w=12 on a 375px page, interleaved as 'خا' 'د' 'محا' 'ح1' 'حا' among the
# body text, corrupting the element sequence the sidecar and the LLM's sentence context read.
#
# Four conditions together, because no single one discriminates. The same page carries coin
# tokens that are just as narrow (x=174-288, w=11-13) but sit mid-page, and a drawing can
# legitimately have a label near an edge.
# All four fractions are of page width. Width was measured against the *median word width*
# first, which fails on exactly the page this is for: the column's own cells, plus the coin
# tokens, drag test1's median down to 37px — below the 12px cells being looked for. A page
# dense in small marks makes its own median useless as a scale. Page width doesn't move.
MARGIN_BAND_FRAC = 0.15          # how far in from an edge the column may sit
MARGIN_MAX_WIDTH_FRAC = 0.08     # a cell is a mark, never a phrase
MARGIN_ALIGN_FRAC = 0.06         # how tightly the cells must share an x — the real filter
MARGIN_MIN_CELLS = 3             # a column, not a stray label

# Measured on test1.jpeg (375px wide) with these values: the ten cells at x=328 sit at
# centre 334 against a right band starting at 318.75, and the coin tokens — identically
# narrow — top out at centre 294.5 and fall outside it. ~24px of separation between the two
# groups, which is where the band fraction came from rather than from a round number.


def median_text_height_px(ocr_words: list[OcrWord]) -> float:
    """Median height of the detected text boxes, 0.0 when there is no text."""
    if not ocr_words:
        return 0.0
    return float(median(w.bbox_px["h"] for w in ocr_words))


def reconstruct_layout(
    ocr_words: list[OcrWord],
    shapes: list[ShapeResult],
    image_width: int,
    image_height: int,
) -> list[Element]:
    """Convert pixel-coordinate OCR words and shapes into relative-coord Element list, sorted top-to-bottom."""
    elements: list[Element] = []
    # id, bbox_px pairs — same ids the shape-element loop below assigns, computed
    # up front so word/shape containment can be checked before shape elements exist.
    shape_ids_px = [(f"shape_{j:03d}", shape.bbox_px) for j, shape in enumerate(shapes)]

    # A marking column is pulled out of the prose and appended after it, top to bottom,
    # rather than dropped — it is content, just not part of any sentence. See
    # MARGIN_MIN_CELLS. Identity, not equality: two blank cells compare equal as dataclasses.
    body, margin_words = _split_margin_column(ocr_words, image_width)
    in_margin = {id(w) for w in margin_words}
    ordered = _reading_order(body) + sorted(margin_words, key=lambda w: w.bbox_px["y"])

    for i, word in enumerate(ordered):
        bbox = to_relative_bbox(word.bbox_px, image_width, image_height)
        elements.append(TextElement(
            id=f"text_{i:03d}",
            bbox=bbox,
            # No bidi reordering here. arabic_PP-OCRv5_mobile_rec already returns logical
            # (Unicode storage) order, which is what the sidecar and Word both want — the
            # visual-order reversal that used to sit on this line was a v4 compensation
            # and would now corrupt correct text. See the note in utils/bidi.py.
            content=word.text,
            language=detect_language(word.text),
            confidence=word.confidence,
            container_shape_id=_innermost_container(word.bbox_px, shape_ids_px),
            digits_recovered=word.digits_recovered,
            digit_disagreement=word.digit_disagreement,
            margin_column=id(word) in in_margin,
            # Set here rather than in the LLM stage, which is optional and may never run.
            # A line the two recognisers disagree about numerically ships marked whatever
            # its confidence says — that score is the Arabic model's opinion of the Arabic
            # it read, and it stays high on exactly the lines it dropped a number from.
            highlight="red" if word.digit_disagreement else None,
        ))

    for j, shape in enumerate(shapes):
        bbox = to_relative_bbox(shape.bbox_px, image_width, image_height)
        if shape.shape_type == "complex":
            elements.append(ComplexShapeElement(id=f"shape_{j:03d}", bbox=bbox))
        elif shape.shape_type == "polyline":
            elements.append(PolylineShapeElement(
                id=f"shape_{j:03d}",
                bbox=bbox,
                # Already fractions of the shape's own bbox (see shape_detector.
                # _polyline_points), so the px -> relative step doesn't touch them.
                points=list(shape.points or []),
                confidence=shape.confidence,
            ))
        else:
            elements.append(SimpleShapeElement(
                id=f"shape_{j:03d}",
                bbox=bbox,
                shape=shape.shape_type,
                confidence=shape.confidence,
            ))

    return elements


def _split_margin_column(
    words: list[OcrWord], image_width: int,
) -> tuple[list[OcrWord], list[OcrWord]]:
    """Split (body, margin) — a narrow, edge-hugging, tightly-aligned column of ≥3 cells.

    All four conditions must hold at once. Width alone catches the coin tokens on the same
    worksheet; edge proximity alone catches any label near a border; and the alignment
    tolerance is what actually separates a printed column from scattered edge annotations,
    because a real marking column shares an x to within a couple of pixels.

    Returns everything as body when no column is found, which is the common case — a drawing
    has no marking column, and neither does any page this service was originally built for.
    """
    if len(words) < MARGIN_MIN_CELLS or image_width <= 0:
        return list(words), []

    def centre(word: OcrWord) -> float:
        return word.bbox_px["x"] + word.bbox_px["w"] / 2

    narrow_max = image_width * MARGIN_MAX_WIDTH_FRAC
    band = image_width * MARGIN_BAND_FRAC
    candidates = [
        w for w in words
        if w.bbox_px["w"] <= narrow_max
        and (centre(w) <= band or centre(w) >= image_width - band)
    ]
    if len(candidates) < MARGIN_MIN_CELLS:
        return list(words), []

    # The condition that actually earns its keep: a *margin* column sits outside the text
    # block, not merely near an edge. Without this, measured on the real pages, the rule ate
    # test3's row labels (S1/S2/S3) and 19 of class-diagram's UML attributes — all narrow,
    # edge-adjacent and tightly aligned, and all genuine content that belongs in the prose.
    # What separates them is that they share their horizontal span with body text; a marking
    # column does not, because it lives in the margin the body was laid out to avoid.
    is_candidate = {id(w) for w in candidates}
    block = [w for w in words if id(w) not in is_candidate]
    if not block:
        return list(words), []
    block_left = min(w.bbox_px["x"] for w in block)
    block_right = max(w.bbox_px["x"] + w.bbox_px["w"] for w in block)
    candidates = [
        w for w in candidates
        if w.bbox_px["x"] + w.bbox_px["w"] <= block_left or w.bbox_px["x"] >= block_right
    ]
    if len(candidates) < MARGIN_MIN_CELLS:
        return list(words), []

    # Cluster by x, keeping only runs that are both tightly aligned and deep enough. Two
    # columns (both page edges marked) are handled by this falling out as two clusters.
    tolerance = image_width * MARGIN_ALIGN_FRAC
    margin: list[OcrWord] = []
    cluster: list[OcrWord] = []
    for word in sorted(candidates, key=centre):
        if cluster and centre(word) - centre(cluster[0]) > tolerance:
            if len(cluster) >= MARGIN_MIN_CELLS:
                margin.extend(cluster)
            cluster = []
        cluster.append(word)
    if len(cluster) >= MARGIN_MIN_CELLS:
        margin.extend(cluster)

    if not margin:
        return list(words), []
    marked = {id(w) for w in margin}
    return [w for w in words if id(w) not in marked], margin


def _reading_order(words: list[OcrWord]) -> list[OcrWord]:
    """Order words top-to-bottom, then along each line in that line's own direction.

    Sorting by y alone leaves words sharing a line in whatever order OCR emitted them.
    Words are grouped into lines by vertical centre, then each line is sorted by x —
    reversed for Arabic lines, which read right-to-left.
    """
    if not words:
        return []

    heights = sorted(w.bbox_px["h"] for w in words)
    tolerance = max(heights[len(heights) // 2] * 0.6, 1.0)

    ordered: list[OcrWord] = []
    line: list[OcrWord] = []
    line_y = None
    for word in sorted(words, key=lambda w: w.bbox_px["y"]):
        centre = word.bbox_px["y"] + word.bbox_px["h"] / 2
        if line_y is not None and abs(centre - line_y) > tolerance:
            ordered.extend(_order_line(line))
            line = []
            line_y = None
        line.append(word)
        if line_y is None:
            line_y = centre
    ordered.extend(_order_line(line))
    return ordered


def _order_line(line: list[OcrWord]) -> list[OcrWord]:
    if not line:
        return line
    rtl = sum(1 for w in line if is_arabic(w.text)) * 2 > len(line)
    return sorted(line, key=lambda w: w.bbox_px["x"], reverse=rtl)


def _innermost_container(word_bbox_px: dict, shape_ids_px: list[tuple[str, dict]]) -> str | None:
    """id of the smallest-area shape whose bbox strictly contains word_bbox_px, or None.

    Shapes nest (e.g. an attribute label's box sits inside the class box, which can
    itself sit inside a package frame) — smallest area wins so a word binds to its
    immediate container, not some outer wrapper several levels up. Measured on
    class-diagram.png: 65% of words land inside at least one shape, nesting never
    goes past 2 deep, so "just take the smallest" doesn't need to be smarter than this.
    """
    best_id, best_area = None, None
    for shape_id, shape_bbox in shape_ids_px:
        if _bbox_contains(shape_bbox, word_bbox_px):
            area = shape_bbox["w"] * shape_bbox["h"]
            if best_area is None or area < best_area:
                best_id, best_area = shape_id, area
    return best_id


def _bbox_contains(outer: dict, inner: dict) -> bool:
    return (
        outer["x"] <= inner["x"] and outer["y"] <= inner["y"] and
        outer["x"] + outer["w"] >= inner["x"] + inner["w"] and
        outer["y"] + outer["h"] >= inner["y"] + inner["h"]
    )


def to_relative_bbox(bbox_px: dict, image_width: int, image_height: int) -> BBox:
    return BBox(
        x=bbox_px["x"] / image_width,
        y=bbox_px["y"] / image_height,
        w=bbox_px["w"] / image_width,
        h=bbox_px["h"] / image_height,
    )
