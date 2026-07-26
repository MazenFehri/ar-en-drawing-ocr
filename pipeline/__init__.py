import time
from dataclasses import dataclass
import numpy as np
import cv2

from pipeline.preprocessor import preprocess, detect_quality
from pipeline.layout import segment_layout
from pipeline.shape_detector import detect_shapes
from pipeline.ocr import run_ocr
from pipeline.llm_corrector import apply_corrections
from pipeline.layout_reconstructor import reconstruct_layout
from pipeline.word_assembler import assemble_document
from pipeline.sidecar import build_sidecar
from app.config import settings


@dataclass
class PipelineResult:
    docx_bytes: bytes
    sidecar: dict


def process_image(
    image: np.ndarray,
    confidence_threshold: float = None,
) -> PipelineResult:
    threshold = confidence_threshold if confidence_threshold is not None else settings.confidence_threshold
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
    words = run_ocr(preprocessed, confidence_threshold=threshold)

    # Stage 4: Shape detection. Mask both the layout text regions and the actual
    # OCR word boxes; without the word boxes every glyph cluster the layout model
    # missed comes back as a "complex" shape and gets duplicated into the document.
    shapes = detect_shapes(
        preprocessed,
        text_bboxes_px=text_region_bboxes + [w.bbox_px for w in words],
    )

    # Stage 5: Layout reconstruction (pixel -> relative coords, reading order)
    elements = reconstruct_layout(words, shapes, img_w, img_h)

    # Stage 6: LLM correction for flagged words
    _, img_encoded = cv2.imencode(".jpg", preprocessed)
    elements = apply_corrections(elements, img_encoded.tobytes(), confidence_threshold=threshold)

    # Build crop map: shape_{j:03d} -> crop ndarray, for complex shapes
    crop_images: dict[str, np.ndarray] = {}
    for j, shape in enumerate(shapes):
        if shape.shape_type == "complex" and shape.crop is not None:
            crop_images[f"shape_{j:03d}"] = shape.crop

    # Stage 7: Word assembly
    docx_bytes = assemble_document(elements, crop_images=crop_images)

    # Stage 8: JSON sidecar
    elapsed_ms = int((time.time() - start) * 1000)
    sidecar = build_sidecar(
        elements, orig_w, orig_h, elapsed_ms,
        quality_score=detect_quality(preprocessed),
    )

    return PipelineResult(docx_bytes=docx_bytes, sidecar=sidecar)
