from models.elements import BBox, TextElement, SimpleShapeElement, ComplexShapeElement, Element
from pipeline.ocr import OcrWord
from pipeline.shape_detector import ShapeResult
from utils.bidi import detect_language, is_arabic, to_logical_order


def reconstruct_layout(
    ocr_words: list[OcrWord],
    shapes: list[ShapeResult],
    image_width: int,
    image_height: int,
) -> list[Element]:
    """Convert pixel-coordinate OCR words and shapes into relative-coord Element list, sorted top-to-bottom."""
    elements: list[Element] = []

    for i, word in enumerate(_reading_order(ocr_words)):
        bbox = to_relative_bbox(word.bbox_px, image_width, image_height)
        elements.append(TextElement(
            id=f"text_{i:03d}",
            bbox=bbox,
            # OCR hands back visual order; the sidecar and Word both want logical.
            content=to_logical_order(word.text),
            language=detect_language(word.text),
            confidence=word.confidence,
        ))

    for j, shape in enumerate(shapes):
        bbox = to_relative_bbox(shape.bbox_px, image_width, image_height)
        if shape.shape_type == "complex":
            elements.append(ComplexShapeElement(id=f"shape_{j:03d}", bbox=bbox))
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


def to_relative_bbox(bbox_px: dict, image_width: int, image_height: int) -> BBox:
    return BBox(
        x=bbox_px["x"] / image_width,
        y=bbox_px["y"] / image_height,
        w=bbox_px["w"] / image_width,
        h=bbox_px["h"] / image_height,
    )
