from typing import Literal, Optional, Union
from pydantic import BaseModel


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class LLMCorrection(BaseModel):
    original: str
    corrected: str
    certainty: float


class TextElement(BaseModel):
    id: str
    type: Literal["text"] = "text"
    bbox: BBox
    content: str
    language: Literal["arabic", "english", "mixed"]
    confidence: float
    llm_correction: Optional[LLMCorrection] = None
    highlight: Optional[Literal["yellow", "red"]] = None
    # Numeric integrity, from the two-recogniser digit check in pipeline/ocr.py.
    # digits_recovered: runs the donor put back into this line — the fix, working.
    # digit_disagreement: the donor saw a number here that could not be placed, so a value
    # is probably missing and nobody can say which. Independent of `confidence`: the Arabic
    # recogniser reports a healthy score on a line it silently deleted a number from, so a
    # confident line can still carry this. Consumers should force review when it is true.
    digits_recovered: list[str] = []
    digit_disagreement: bool = False
    # True for cells of an edge marking column (see layout_reconstructor.MARGIN_MIN_CELLS).
    # They are real content and keep their position in the document, but they belong to no
    # sentence — consumers building a text stream, or feeding neighbouring words to a model
    # as context, should skip them.
    margin_column: bool = False
    # id of the smallest detected shape (models/elements.ComplexShapeElement /
    # SimpleShapeElement) whose bbox strictly contains this word's bbox, e.g. the
    # class box a UML attribute label sits inside. None is the common case (most
    # words, and most drawings with no shapes at all) — set by
    # pipeline/layout_reconstructor.py, consumed by pipeline/llm_corrector.py to
    # find confident sibling text for a flagged word's correction prompt.
    container_shape_id: Optional[str] = None


class SimpleShapeElement(BaseModel):
    id: str
    type: Literal["simple_shape"] = "simple_shape"
    bbox: BBox
    shape: Literal["circle", "ellipse", "triangle", "rect", "square", "line"]
    confidence: float


class PolylineShapeElement(BaseModel):
    """An open stroke — a connector, leader or curved association line.

    Distinct from SimpleShapeElement because there is no DrawingML preset geometry for
    "arbitrary path": pipeline/word_assembler.py emits it as an a:custGeom freeform, so
    it stays a real vector shape in Word instead of a raster picture. Distinct from
    ComplexShapeElement because it is line art, not an image — raster embedding is
    reserved for genuinely image-like regions (a photo, a logo, a textured blob).

    points: the path as (x, y) fractions of this element's own bbox, in draw order.
    """
    id: str
    type: Literal["polyline_shape"] = "polyline_shape"
    bbox: BBox
    shape: Literal["polyline"] = "polyline"
    points: list[tuple[float, float]]
    confidence: float


class ComplexShapeElement(BaseModel):
    id: str
    type: Literal["complex_shape"] = "complex_shape"
    bbox: BBox
    shape: str = "unknown"
    embedded_as: Literal["image"] = "image"
    llm_label: Optional[str] = None
    llm_label_certainty: Optional[float] = None


Element = Union[TextElement, SimpleShapeElement, PolylineShapeElement, ComplexShapeElement]
