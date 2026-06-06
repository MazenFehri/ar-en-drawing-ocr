from models.elements import Element, TextElement, SimpleShapeElement, ComplexShapeElement


def build_sidecar(
    elements: list[Element],
    image_width: int,
    image_height: int,
    processing_time_ms: int,
) -> dict:
    serialized = [_serialize(e) for e in elements]
    text_count = sum(1 for e in elements if isinstance(e, TextElement))
    simple_count = sum(1 for e in elements if isinstance(e, SimpleShapeElement))
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
            "complex_shapes": complex_count,
            "llm_corrections": correction_count,
            "processing_time_ms": processing_time_ms,
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
    elif isinstance(el, ComplexShapeElement):
        d["shape"] = el.shape
        d["embedded_as"] = el.embedded_as
        d["llm_label"] = el.llm_label
        d["llm_label_certainty"] = el.llm_label_certainty
    return d
