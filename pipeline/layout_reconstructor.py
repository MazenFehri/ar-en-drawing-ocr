from models.elements import (
    BBox, TextElement, SimpleShapeElement, PolylineShapeElement, ComplexShapeElement, Element,
)
from pipeline.ocr import OcrWord
from pipeline.shape_detector import ShapeResult
from utils.bidi import detect_language, is_arabic


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

    for i, word in enumerate(_reading_order(ocr_words)):
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
