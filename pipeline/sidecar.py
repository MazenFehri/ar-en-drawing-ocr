from models.elements import Element, TextElement, SimpleShapeElement, ComplexShapeElement


def build_sidecar(
    elements: list[Element],
    image_width: int,
    image_height: int,
    processing_time_ms: int,
    quality_score: float | None = None,
    llm_status: dict | None = None,
    shape_label_status: dict | None = None,
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
            # Laplacian sharpness in [0,1]. Low values mean the scan was blurry and
            # the confidences below should be read with that in mind.
            "quality_score": quality_score,
            # Without these, "llm_corrections: 0" is ambiguous: it reads the same
            # whether the model confirmed every word or the provider was down.
            # {"state": not_attempted|success|failed, "reason": str|None, "model": str|None}
            "llm_status": llm_status,
            "shape_label_status": shape_label_status,
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
