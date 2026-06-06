# Arabic Architectural OCR Pipeline — Design Spec

**Date:** 2026-06-06  
**Status:** Approved

---

## Overview

A Python microservice (FastAPI, Docker) that accepts an image of an Arabic/English architectural drawing (handwritten or printed) and returns a `.docx` Word document with all text, shapes, and symbols at approximately correct positions, plus a JSON sidecar describing every detected element with confidence scores and LLM corrections.

---

## Constraints

- Deployed as a REST API microservice, called from a .NET application
- Commercial cloud APIs allowed, but must be low-cost (target: < $0.005/image)
- Mixed Arabic (RTL) + English (LTR) text
- Approximately correct spatial positioning (not pixel-precise)
- Shapes: simple geometry → Word vector shapes; complex shapes → cropped images embedded in doc
- Low-confidence words: auto-corrected by LLM (Claude Haiku), highlighted yellow (certain) or red (uncertain)
- User corrections feed back into the system for continuous improvement

---

## Pipeline Stages

| Stage | Tool | Input | Output |
|---|---|---|---|
| 1. Preprocessing | OpenCV | Raw image | Deskewed, denoised, normalized image |
| 2. Layout Segmentation | PaddleOCR ppstructure | Preprocessed image | Text regions, shape regions, margin notes |
| 3. Shape Detection | OpenCV contours | Non-text regions | Simple shapes (type + bbox) or complex shape crops |
| 4. OCR | PaddleOCR (ar+en) | Text regions | Words with bounding boxes + confidence scores |
| 5. LLM Correction | Claude Haiku Vision | Full image + flagged words | Corrected words + certainty scores |
| 6. Layout Reconstruction | Custom | All elements | Reading-order list with relative positions |
| 7. Word Assembly | python-docx | Layout + elements | .docx file |
| 8. Sidecar Generation | Custom | All elements + corrections | JSON sidecar |

---

## Shape Classification Rule

```
contour detected
    │
    ├─ 3 vertices              → Word triangle shape
    ├─ 4 vertices
    │   ├─ aspect ratio ~1.0   → Word square
    │   └─ else                → Word rectangle
    ├─ 5+ vertices
    │   ├─ circularity > 0.85  → Word ellipse
    │   └─ circularity ≤ 0.85  → complex: crop + embed image
    │                             optionally: Claude Haiku label
    └─ overlaps text bbox       → discard (noise)
```

Circularity = `4π × area / perimeter²`

---

## LLM Correction Logic

| OCR Confidence | LLM Certainty | Word Doc Highlight |
|---|---|---|
| ≥ 0.75 | — | None (accepted) |
| 0.40–0.74 | ≥ 0.60 | Yellow |
| 0.40–0.74 | < 0.60 | Red |
| < 0.40 | ≥ 0.60 | Yellow |
| < 0.40 | < 0.60 | Red (keep original) |

The full image is sent to Claude Haiku Vision once per document to provide visual context for all corrections.

---

## API Contract

### POST /process
- **Request:** `multipart/form-data` — `image` file, optional `confidence_threshold` (default 0.75), `language_hint` (default `ar+en`), `label_shapes` (default true)
- **Response:** JSON with `document_id`, `docx` (binary stream), and `sidecar` object

### POST /feedback
- **Request:** JSON — `document_id`, array of `{element_id, user_final}` corrections
- **Response:** 204 No Content

### GET /health
- **Response:** `{"status": "ok", "models_loaded": true}`

---

## Sidecar Schema

```json
{
  "document_id": "uuid",
  "page_dimensions": {"width_px": 2480, "height_px": 3508},
  "elements": [
    {
      "id": "text_001",
      "type": "text | simple_shape | complex_shape",
      "bbox": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
      "content": "string (text only)",
      "language": "arabic | english | mixed",
      "confidence": 0.0,
      "llm_correction": {"original": "", "corrected": "", "certainty": 0.0},
      "highlight": "yellow | red | null",
      "shape": "circle | triangle | rect | unknown (shapes only)",
      "embedded_as": "image (complex shapes only)",
      "llm_label": "string (complex shapes only)"
    }
  ],
  "stats": {
    "total_elements": 0,
    "text_elements": 0,
    "simple_shapes": 0,
    "complex_shapes": 0,
    "llm_corrections": 0,
    "processing_time_ms": 0
  }
}
```

---

## Feedback Loop

| Layer | Trigger | Effect |
|---|---|---|
| Immediate | Every correction stored in PostgreSQL | Data accumulates |
| Few-shot (short-term) | ~50 corrections available | Top-N similar past corrections prepended to LLM prompt |
| Fine-tuning (long-term) | ~500+ correction pairs | PaddleOCR recognition model retrained on domain data |

---

## Technology Stack

| Concern | Choice |
|---|---|
| API framework | FastAPI + uvicorn |
| Image processing | OpenCV 4.9+, Pillow |
| OCR | PaddleOCR PP-OCRv4 |
| Layout analysis | PaddleOCR ppstructure |
| LLM correction | Claude Haiku (claude-haiku-4-5) via Anthropic SDK |
| Word generation | python-docx |
| Bidi text | python-bidi + arabic-reshaper |
| Feedback storage | PostgreSQL |
| Packaging | Docker |
| Integration | Called by .NET via HTTP (HttpClient + MultipartFormDataContent) |

---

## Cost Target

- Local processing (OCR + CV): $0.00
- Claude Haiku correction per image: ~$0.0002–$0.0004
- Claude Haiku Vision per complex shape: ~$0.0002
- **Total per image: ~$0.001–$0.002**
- At 10K images/month: ~$10–20 LLM costs

---

## Implementation Phases

1. **Core OCR** (Weeks 1–3): API skeleton, preprocessing, PaddleOCR, basic Word output
2. **Layout + Positioning** (Weeks 4–5): Reading order, RTL, absolute text box positioning
3. **Shape Detection** (Weeks 6–7): Contours, vector shapes, image crops, shape labeling
4. **LLM Correction** (Week 8): Claude Haiku Vision, highlighting, JSON sidecar
5. **Feedback Loop** (Week 9): `/feedback` endpoint, PostgreSQL, few-shot injection
6. **Hardening** (Week 10): Error handling, tiling for large images, load testing
