import time
from dataclasses import dataclass
import numpy as np
import cv2

from pipeline.preprocessor import preprocess, detect_quality
from pipeline.layout import segment_layout
from pipeline.shape_detector import detect_shapes
from pipeline.ocr import run_ocr, DEFAULT_LANGUAGE_HINT
from pipeline.llm_corrector import apply_corrections, label_complex_shapes, TOTAL_LLM_BUDGET_SECONDS
from pipeline.layout_reconstructor import reconstruct_layout
from pipeline.word_assembler import assemble_document
from pipeline.sidecar import build_sidecar
from utils.image_utils import ndarray_to_png_bytes
from app.config import settings


# Fraction of the shared LLM budget word correction may spend before shape
# labelling gets the rest. Half each: with PER_ATTEMPT_CAP_SECONDS at 15s, 30s
# still buys either stage two full attempts, so neither is cut to one shot.
# ponytail: a fixed split, blind to how much work each stage actually has — a
# page with two flagged words and fifty shapes still reserves half for the
# words. Upgrade path if that shows up: weight the split by len(flagged) vs
# len(crop_images), both known before either call.
CORRECTION_BUDGET_SHARE = 0.5


@dataclass
class PipelineResult:
    docx_bytes: bytes
    sidecar: dict


def process_image(
    image: np.ndarray,
    confidence_threshold: float = None,
    language_hint: str = DEFAULT_LANGUAGE_HINT,
    label_shapes: bool = None,
) -> PipelineResult:
    threshold = confidence_threshold if confidence_threshold is not None else settings.confidence_threshold
    want_labels = label_shapes if label_shapes is not None else settings.label_shapes
    start = time.time()

    # Sidecar reports the caller's original page size; element bboxes are relative
    # so they stay valid against it even though preprocessing may have downscaled.
    orig_h, orig_w = image.shape[:2]

    # Stage 1: Preprocess (may downscale — normalize coords against the result)
    preprocessed = preprocess(image)
    img_h, img_w = preprocessed.shape[:2]

    # Stage 2: Layout segmentation
    regions = segment_layout(preprocessed)
    text_region_bboxes = [r.bbox_px for r in regions if r.region_type == "text"]

    # Stage 3: OCR — runs before shape detection so its word boxes can mask text.
    words = run_ocr(preprocessed, confidence_threshold=threshold, language_hint=language_hint)

    # Stage 4: Shape detection. Mask both the layout text regions and the actual
    # OCR word boxes; without the word boxes every glyph cluster the layout model
    # missed comes back as a "complex" shape and gets duplicated into the document.
    shapes = detect_shapes(
        preprocessed,
        text_bboxes_px=text_region_bboxes + [w.bbox_px for w in words],
    )

    # Stage 5: Layout reconstruction (pixel -> relative coords, reading order)
    elements = reconstruct_layout(words, shapes, img_w, img_h)

    # Stage 6: LLM correction for flagged words. One deadline computed here and
    # passed into both this call and stage 6b's label_complex_shapes below —
    # without that they'd each get their own full TOTAL_LLM_BUDGET_SECONDS,
    # doubling the worst-case wait for a request that hits both. See the
    # comment on TOTAL_LLM_BUDGET_SECONDS in pipeline/llm_corrector.py.
    llm_deadline = time.monotonic() + TOTAL_LLM_BUDGET_SECONDS
    # Sharing one budget means whichever stage runs first can spend all of it.
    # Measured on a 167-element page: word correction succeeded but consumed the
    # lot, and shape labelling came back timed_out on 2 of 3 runs. Word
    # correction therefore gets a share, not the whole thing, so labelling
    # always has something left to work with.
    correction_deadline = min(
        llm_deadline, time.monotonic() + TOTAL_LLM_BUDGET_SECONDS * CORRECTION_BUDGET_SHARE,
    )
    _, img_encoded = cv2.imencode(".jpg", preprocessed)
    elements, llm_status = apply_corrections(
        elements, img_encoded.tobytes(), confidence_threshold=threshold, deadline=correction_deadline,
    )

    # Build crop map: shape_{j:03d} -> crop ndarray, for complex shapes
    crop_images: dict[str, np.ndarray] = {}
    for j, shape in enumerate(shapes):
        if shape.shape_type == "complex" and shape.crop is not None:
            crop_images[f"shape_{j:03d}"] = shape.crop

    # Stage 6b: name the complex shapes, so the sidecar says "door swing" not "unknown"
    shape_status = {"state": "not_attempted", "reason": "no_shapes_to_label", "model": None}
    if want_labels and crop_images:
        elements, shape_status = label_complex_shapes(
            elements,
            {eid: ndarray_to_png_bytes(crop) for eid, crop in crop_images.items()},
            deadline=llm_deadline,
        )

    # Stage 7: Word assembly. Pass the *original* aspect ratio so the page is
    # letterboxed to match the source — bboxes are relative fractions, and mapping
    # x by page width and y by page height independently stretches the drawing.
    docx_bytes = assemble_document(
        elements, crop_images=crop_images, page_aspect=orig_w / orig_h,
    )

    # Stage 8: JSON sidecar
    elapsed_ms = int((time.time() - start) * 1000)
    sidecar = build_sidecar(
        elements, orig_w, orig_h, elapsed_ms,
        quality_score=detect_quality(preprocessed),
        llm_status=llm_status,
        shape_label_status=shape_status,
    )

    return PipelineResult(docx_bytes=docx_bytes, sidecar=sidecar)
