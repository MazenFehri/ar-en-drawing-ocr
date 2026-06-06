from models.elements import BBox, TextElement, SimpleShapeElement, ComplexShapeElement, Element
from pipeline.ocr import OcrWord
from pipeline.shape_detector import ShapeResult
from utils.bidi import detect_language


def reconstruct_layout(
    ocr_words: list[OcrWord],
    shapes: list[ShapeResult],
    image_width: int,
    image_height: int,
) -> list[Element]:
    """Convert pixel-coordinate OCR words and shapes into relative-coord Element list, sorted top-to-bottom."""
    elements: list[Element] = []

    sorted_words = sorted(ocr_words, key=lambda w: w.bbox_px["y"])
    for i, word in enumerate(sorted_words):
        bbox = to_relative_bbox(word.bbox_px, image_width, image_height)
        elements.append(TextElement(
            id=f"text_{i:03d}",
            bbox=bbox,
            content=word.text,
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


def to_relative_bbox(bbox_px: dict, image_width: int, image_height: int) -> BBox:
    return BBox(
        x=bbox_px["x"] / image_width,
        y=bbox_px["y"] / image_height,
        w=bbox_px["w"] / image_width,
        h=bbox_px["h"] / image_height,
    )
