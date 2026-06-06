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


class SimpleShapeElement(BaseModel):
    id: str
    type: Literal["simple_shape"] = "simple_shape"
    bbox: BBox
    shape: Literal["circle", "ellipse", "triangle", "rect", "square", "line"]
    confidence: float


class ComplexShapeElement(BaseModel):
    id: str
    type: Literal["complex_shape"] = "complex_shape"
    bbox: BBox
    shape: str = "unknown"
    embedded_as: Literal["image"] = "image"
    llm_label: Optional[str] = None
    llm_label_certainty: Optional[float] = None


Element = Union[TextElement, SimpleShapeElement, ComplexShapeElement]
