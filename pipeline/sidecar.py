from models.elements import (
    Element, TextElement, SimpleShapeElement, PolylineShapeElement, ComplexShapeElement,
)
from pipeline.layout_reconstructor import LOW_RESOLUTION_TEXT_HEIGHT_PX


def build_sidecar(
    elements: list[Element],
    image_width: int,
    image_height: int,
    processing_time_ms: int,
    quality_score: float | None = None,
    llm_status: dict | None = None,
    shape_label_status: dict | None = None,
    median_text_height_px: float | None = None,
) -> dict:
    serialized = [_serialize(e) for e in elements]
    text_count = sum(1 for e in elements if isinstance(e, TextElement))
    simple_count = sum(1 for e in elements if isinstance(e, SimpleShapeElement))
    polyline_count = sum(1 for e in elements if isinstance(e, PolylineShapeElement))
    complex_count = sum(1 for e in elements if isinstance(e, ComplexShapeElement))
    correction_count = sum(
        1 for e in elements
        if isinstance(e, TextElement) and e.llm_correction is not None
    )
    return {
        "page_dimensions": {"width_px": image_width, "height_px": image_height},
        "elements": serialized,
        "stats": {
            "total_elements": len(elements),
            "text_elements": text_count,
            "simple_shapes": simple_count,
            # Counted separately from simple_shapes: these are vector freeforms with a
            # point list, not preset geometry, and separately from complex_shapes
            # because they are NOT embedded as images.
            "polyline_shapes": polyline_count,
            "complex_shapes": complex_count,
            "llm_corrections": correction_count,
            "processing_time_ms": processing_time_ms,
            # Laplacian sharpness in [0,1]. Low values mean the scan was blurry and
            # the confidences below should be read with that in mind.
            "quality_score": quality_score,
            # Without these, "llm_corrections: 0" is ambiguous: it reads the same
            # whether the model confirmed every word or the provider was down.
            # {"state": not_attempted|success|failed, "reason": str|None, "model": str|None}
            "llm_status": llm_status,
            "shape_label_status": shape_label_status,
            "median_text_height_px": median_text_height_px,
            # The honest "this page was too low-DPI to read reliably" signal. The
            # confidences alone don't say it: the Arabic recogniser reports ~0.65 on a
            # page whose letter dots were never sampled, which reads as merely mediocre
            # rather than as guessing.
            #
            # A height of 0.0 means no text was found at all, which is a different claim
            # and is deliberately not reported as this one — a drawing with no labels is
            # a normal input here, and telling its caller to go rescan a perfectly good
            # sheet would make the flag worthless. Falsy covers that and the None an
            # older caller passes; text_elements == 0 already distinguishes "blank".
            "low_resolution": bool(
                median_text_height_px
                and median_text_height_px < LOW_RESOLUTION_TEXT_HEIGHT_PX
            ),
        },
    }


def _serialize(el: Element) -> dict:
    d = {
        "id": el.id,
        "type": el.type,
        "bbox": el.bbox.model_dump(),
    }
    if isinstance(el, TextElement):
        d["content"] = el.content
        d["language"] = el.language
        d["confidence"] = el.confidence
        d["highlight"] = el.highlight
        d["llm_correction"] = el.llm_correction.model_dump() if el.llm_correction else None
    elif isinstance(el, SimpleShapeElement):
        d["shape"] = el.shape
        d["confidence"] = el.confidence
    elif isinstance(el, PolylineShapeElement):
        d["shape"] = el.shape
        d["confidence"] = el.confidence
        # Plain lists, not tuples: this dict goes straight out as JSON.
        d["points"] = [[float(px), float(py)] for px, py in el.points]
    elif isinstance(el, ComplexShapeElement):
        d["shape"] = el.shape
        d["embedded_as"] = el.embedded_as
        d["llm_label"] = el.llm_label
        d["llm_label_certainty"] = el.llm_label_certainty
    return d
