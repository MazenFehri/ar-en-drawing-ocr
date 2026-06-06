# Arabic Architectural Drawing OCR — Deep Research

> **Project:** Image-to-Word document pipeline for Arabic/English architectural drawings  
> **Date:** 2026-06-06  
> **Scope:** Mixed Arabic/English handwritten and printed images containing text, geometric shapes, complex shapes, and side notes → .docx + JSON sidecar

---

## Table of Contents

1. [Problem Space](#1-problem-space)
2. [Arabic OCR — State of the Art](#2-arabic-ocr--state-of-the-art)
3. [Handwritten vs Printed Text](#3-handwritten-vs-printed-text)
4. [Mixed Bidirectional Text (RTL + LTR)](#4-mixed-bidirectional-text-rtl--ltr)
5. [Document Layout Analysis](#5-document-layout-analysis)
6. [Shape and Symbol Detection](#6-shape-and-symbol-detection)
7. [LLM-Assisted Correction](#7-llm-assisted-correction)
8. [Word Document Generation](#8-word-document-generation)
9. [Feedback Loop and Fine-Tuning](#9-feedback-loop-and-fine-tuning)
10. [Cost Analysis](#10-cost-analysis)
11. [Architecture Decision](#11-architecture-decision)
12. [Challenges and Mitigations](#12-challenges-and-mitigations)
13. [Implementation Roadmap](#13-implementation-roadmap)
14. [References and Tools](#14-references-and-tools)

---

## 1. Problem Space

### 1.1 What the Input Looks Like

Architectural drawings in Arabic-speaking contexts have specific characteristics that make them harder than general document OCR:

- **Mixed scripts**: Arabic (RTL) and English (LTR) coexist on the same drawing — room labels in Arabic, dimension values in English/numeric, material codes mixed
- **Multiple fonts and weights**: Printed text from CAD tools, handwritten annotations, stamps, rubber-stamp fonts
- **Color variety**: Black ink on white, colored highlighter annotations, blue/red revision marks, faded photocopies
- **Geometric symbols**: Dimension lines, north arrows, grid references, section cut markers (standard architectural notation)
- **Complex shapes**: Floor plan outlines, furniture symbols (chairs, toilets, stairs), site boundary markers, tree symbols, car symbols in parking layouts
- **Side notes and legends**: Margin annotations, material legends, revision clouds, title block text
- **Variable quality**: Scanned at different DPIs (75–600 DPI), skewed, crumpled, low contrast

### 1.2 Why General-Purpose OCR Fails

Standard OCR engines (Tesseract, Google Vision used naively) fail on these inputs for several reasons:

1. **Arabic cursive segmentation**: Arabic letters are connected and change shape depending on their position (initial, medial, final, isolated). A character segmentation approach that works for Latin fails entirely for Arabic.
2. **Diacritics (Tashkeel)**: Short vowel marks appear above/below base letters and are easy to merge with adjacent characters or noise.
3. **Mixed directionality**: A single line may contain `غرفة النوم 3.5m × 4.2m` — Arabic words (RTL) followed by English measurements (LTR). Naive left-to-right bounding box ordering breaks reading order.
4. **Shape interference**: OCR engines see shape contours as noise or try to OCR them as text, corrupting output.
5. **Domain vocabulary**: Architectural terms like "Anteroom", "Fascia", "رواق", "فناء داخلي" are rare in general training data — the model guesses wrong characters and produces valid-but-wrong words.

### 1.3 Success Criteria

- Text preserved with correct Arabic/English content and reading order
- Simple shapes (circle, triangle, rectangle) converted to Word vector objects at approximate positions
- Complex shapes (house, car, stair symbol) cropped and embedded as images at approximate positions
- Low-confidence words auto-corrected by LLM, highlighted in the document
- JSON sidecar contains every element with bounding box, type, confidence, and correction details
- User corrections stored for feedback loop and eventual model improvement
- Cost under $0.005 per image at steady state

---

## 2. Arabic OCR — State of the Art

### 2.1 Open-Source Options

#### PaddleOCR (Recommended)
- **Developed by:** Baidu (PaddlePaddle team)
- **Architecture:** PP-OCRv3 — a three-stage pipeline: text detection (DBNet++), text direction classification, text recognition (SVTR/CRNN)
- **Arabic support:** Yes — `arabic` language pack, supports Arabic + Farsi + Urdu
- **Mixed language:** Can run two language models simultaneously (`arabic,en`)
- **Bounding boxes:** Returns polygon bounding boxes per text line, with confidence scores
- **Handwriting:** PP-OCRv3 handles some handwriting but degrades significantly on highly informal handwriting — PP-OCRv4 (newer) improves this
- **Performance:** ~50–200ms per image on CPU, ~10–30ms on GPU
- **License:** Apache 2.0 — free for commercial use
- **Key advantage:** Built-in layout analysis module (`ppstructure`) that separates text regions, table regions, and image regions before OCR

#### EasyOCR
- **Architecture:** CRAFT (text detection) + CRNN (recognition)
- **Arabic support:** Yes, but accuracy is lower than PaddleOCR for dense Arabic text
- **Mixed language:** Supports combining `ar` + `en` in one call
- **Bounding boxes:** Returns rectangular bounding boxes (less precise than PaddleOCR polygons)
- **Verdict:** Good for prototyping, not recommended for production Arabic OCR

#### Tesseract 5
- **Architecture:** LSTM-based (Tesseract 4+)
- **Arabic support:** Yes — `ara` language data, includes Arabic script
- **Handwriting:** Poor — trained primarily on printed text
- **RTL handling:** Has RTL mode but struggles with mixed RTL/LTR lines
- **Speed:** Slower than PaddleOCR on equivalent hardware
- **Verdict:** Legacy choice; PaddleOCR outperforms it on Arabic in most benchmarks

#### TrOCR (Microsoft)
- **Architecture:** Transformer-based (ViT encoder + GPT-2 decoder)
- **Arabic support:** Limited — base models are English-focused; Arabic would require fine-tuning
- **Handwriting:** Excellent for English handwriting; Arabic would need custom training
- **Verdict:** Good fine-tuning target for Arabic handwriting once a dataset is built

### 2.2 Commercial Cloud Options

#### Azure Document Intelligence (formerly Form Recognizer)
- **Arabic support:** Full — including handwritten Arabic (preview feature as of 2025)
- **Layout model:** Returns paragraphs, lines, words with bounding polygons and confidence
- **Mixed language:** Automatic detection
- **Pricing:** $1.50 per 1,000 pages (Read API) — approximately $0.0015 per image
- **Verdict:** Excellent accuracy, reasonable cost, but more expensive than free local OCR

#### Google Document AI
- **Arabic support:** Yes — Document OCR processor
- **Pricing:** $1.50 per 1,000 pages for the Document OCR processor
- **Layout:** Returns paragraphs, tokens with bounding boxes
- **Verdict:** Comparable to Azure, slightly less Arabic handwriting coverage

#### AWS Textract
- **Arabic support:** Limited — Textract supports Arabic for basic text extraction but not handwriting
- **Verdict:** Not recommended for this use case

#### ABBYY FineReader / Cloud OCR SDK
- **Arabic support:** Excellent — ABBYY has specialized Arabic OCR engines
- **Pricing:** $0.005–$0.01 per page — significantly more expensive
- **Verdict:** Best accuracy, highest cost; overkill given the LLM correction layer

### 2.3 Benchmark Comparison (Arabic Printed Text)

| Engine | CER (Character Error Rate) | WER (Word Error Rate) | Cost/1000 pages |
|---|---|---|---|
| PaddleOCR v4 | ~3–8% | ~8–15% | $0 |
| Azure Document Intelligence | ~1–4% | ~4–10% | $1.50 |
| Google Document AI | ~2–5% | ~5–12% | $1.50 |
| EasyOCR | ~8–15% | ~15–25% | $0 |
| Tesseract 5 | ~10–20% | ~20–35% | $0 |

*CER/WER vary significantly based on image quality, font, and whether text is printed vs. handwritten.*

### 2.4 Decision

**Use PaddleOCR PP-OCRv4** as the primary OCR engine:
- Free, runs locally
- Best open-source accuracy for Arabic
- Has layout analysis built in
- Bounding boxes suitable for position reconstruction
- Fine-tunable on custom data (critical for the feedback loop)

---

## 3. Handwritten vs Printed Text

### 3.1 The Handwritten Arabic Challenge

Handwritten Arabic is substantially harder than printed Arabic OCR for these reasons:

1. **No standard glyph shapes**: Each writer has their own letterforms. The same letter can look completely different across two writers.
2. **Ligatures**: Writers often join letters that are not normally joined in printed text, creating novel visual patterns.
3. **Ink variation**: Pressure, pen type, and speed all affect stroke width and shape.
4. **Overlapping strokes**: Especially for diacritics and dotted letters (ب، ت، ث، ن، ي).
5. **No baseline**: Handwritten text often doesn't sit on a consistent baseline.

### 3.2 Detection Strategy: Is This Handwritten?

Before choosing the OCR approach, classify each text region as printed or handwritten:

- **Stroke width variation:** Printed text has uniform stroke width; handwritten text varies
- **Baseline deviation:** Measure how much each text line deviates from a straight baseline
- **Character spacing regularity:** Printed text has more regular inter-character spacing

A simple classifier (logistic regression on these features) can distinguish printed from handwritten with ~90% accuracy.

### 3.3 Handling Handwritten Regions

For regions classified as handwritten:
1. Apply aggressive preprocessing: binarization (Sauvola thresholding), deskewing, morphological cleanup
2. Run PaddleOCR — it will do its best but confidence scores will be low
3. These words are very likely to fall into the LLM correction path
4. The LLM seeing the full image has additional context to make a good guess

### 3.4 Future: Fine-tuned Handwriting Model

After collecting enough user corrections (target: 500+ correction pairs), fine-tune PaddleOCR's recognition model specifically on the domain's handwriting. This should reduce handwriting WER from ~30–40% to ~10–15%.

---

## 4. Mixed Bidirectional Text (RTL + LTR)

### 4.1 The Bidi Problem

Unicode defines the Bidirectional Algorithm (UBA, Unicode Standard Annex #9) for rendering mixed RTL/LTR text. The challenge for OCR is not rendering but **detection and ordering**:

When a line contains: `غرفة النوم 3.5m × 4.2m الطابق الأول`

The visual order (left to right on the page) is:
`الأول الطابق 4.2m × 3.5m النوم غرفة`

But the logical order (correct reading order) is:
`غرفة النوم 3.5m × 4.2m الطابق الأول`

OCR engines that process boxes left-to-right will reverse the Arabic segments.

### 4.2 PaddleOCR's Bidi Handling

PaddleOCR detects text direction per-line and applies direction classification. However, it handles RTL vs LTR as a whole-line property — mixed lines can still be misordered.

**Mitigation:**
1. After OCR, apply the Python `python-bidi` library to each text line to reorder characters correctly
2. Use the `arabic-reshaper` library to reshape Arabic characters for correct glyph selection
3. For lines with mixed scripts, detect language per-word using `langdetect` and reconstruct logical order

### 4.3 Word Document RTL Support

`python-docx` supports RTL paragraphs via the `bidi` paragraph property. For mixed-direction text boxes, we must:
- Split text boxes by dominant direction (Arabic text box = RTL, English text box = LTR)
- Insert them in correct visual positions using absolute positioning

---

## 5. Document Layout Analysis

### 5.1 What Layout Analysis Does

Before OCR, the image must be segmented into meaningful regions:
- **Text blocks**: Dense areas of text (room labels, notes)
- **Title block**: Usually bottom-right corner of architectural drawings — contains project name, date, scale, revision
- **Dimension lines**: Thin lines with text at ends or midpoints
- **Shape/symbol regions**: Non-text graphic elements
- **Margin annotations (side notes)**: Text in margins, often at an angle
- **Legend/key areas**: Tables of symbols and their meanings

### 5.2 PaddleOCR PP-Structure

`ppstructure` (PaddleOCR's layout module) uses a layout detection model trained on document datasets. It returns region types: `text`, `title`, `figure`, `figure_caption`, `table`, `table_caption`.

**Limitation:** Trained primarily on academic papers and forms — not architectural drawings. May misclassify shape regions as figures or fail to detect architectural-specific regions.

**Mitigation:** Use ppstructure as a first-pass segmentation, then apply additional heuristics for architectural drawings:
- Dimension line detection (thin lines with text at ends)
- Title block detection (bottom-right rectangle with structured text)
- Legend detection (repeated symbol+text pairs)

### 5.3 LayoutParser

LayoutParser is a unified toolkit for document image analysis built on Detectron2. It supports custom model training on domain-specific layouts.

- **For this project:** Could train a custom layout model on architectural drawings if enough labeled examples are available
- **Timeline:** Would require ~200–500 labeled architectural drawings to train effectively
- **Short-term:** Use PaddleOCR layout + heuristics; plan LayoutParser fine-tuning as a future improvement

### 5.4 Reading Order Reconstruction

After segmentation, reading order must be reconstructed. For Arabic architectural drawings:
1. Group regions by horizontal bands (top to bottom)
2. Within each band, order RTL (right to left) for Arabic-dominant bands, LTR for others
3. Special handling for title blocks (read as a structured block, not flowing text)
4. Side notes attached to nearby elements (connect via proximity)

---

## 6. Shape and Symbol Detection

### 6.1 Architectural Symbol Taxonomy

Architectural drawings use a standardized set of symbols. For this system, we classify shapes into:

**Tier 1 — Simple Geometry (Convert to Word vector shapes):**
- Circle / Ellipse: column indicators, bubble tags, north arrow circle
- Triangle: slope indicators, section arrows
- Rectangle / Square: rooms, openings, structural elements
- Line: dimension lines, grid lines, boundary lines
- Polygon (N-sided): irregular room boundaries

**Tier 2 — Standard Architectural Symbols (Crop as image + label):**
- Door swing (arc + line)
- Window (parallel lines in wall gap)
- Staircase (parallel lines with arrow)
- North arrow (compass)
- Section cut marker (circle with line and arrows)
- Toilet / sink / bathtub (fixtures)
- Car / parking space

**Tier 3 — Complex / Irregular Shapes (Crop as image, no label):**
- Tree / vegetation
- Site boundary with irregular outline
- Furniture (sofa, chair, bed)
- Custom client-specific symbols

### 6.2 OpenCV Detection Pipeline

```
Input: Preprocessed binary image with text regions masked out

Step 1: Find contours
  cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

Step 2: Filter noise
  Discard contours with area < min_area_threshold (e.g., 100px²)
  Discard contours that overlap with detected text bounding boxes

Step 3: Geometric classification
  For each contour:
    a. Approximate polygon: approxPolyDP(contour, epsilon=0.04*perimeter, closed=True)
    b. Count vertices:
       3 → Triangle
       4 → check aspect_ratio:
           0.9–1.1 → Square
           else    → Rectangle
       5+ → compute circularity = 4π × area / perimeter²
           > 0.85 → Circle/Ellipse
           ≤ 0.85 → Complex shape

Step 4: Convexity analysis (for Tier 2 symbols)
  Compute convex hull defects
  Door swing: one significant defect + arc
  Staircase: multiple parallel subcontours

Step 5: Output
  Tier 1: {type, vertices, bbox} → Word vector shape
  Tier 2/3: {type, bbox, crop_path} → embedded image
```

### 6.3 Claude Haiku Vision for Shape Labeling

For Tier 2 and Tier 3 crops, optionally call Claude Haiku Vision with the prompt:

```
You are analyzing a cropped region from an architectural drawing.
In one word or short phrase, identify what this shape/symbol represents.
Examples: "door swing", "staircase", "tree", "car", "toilet", "north arrow"
If you cannot identify it, respond with "unknown".
```

This gives a human-readable label for the JSON sidecar at minimal cost (~$0.0002 per crop).

### 6.4 Masking Shapes Before OCR

A critical preprocessing step: after shape detection, **mask all shape regions** (fill with white) before running OCR. This prevents the OCR engine from trying to interpret shape edges as characters.

---

## 7. LLM-Assisted Correction

### 7.1 Why LLM Correction Works for OCR

OCR produces character sequences — it has no semantic understanding. An LLM knows that:
- "entrnce" near a door symbol must be "entrance"
- "غرفه" near a bed symbol is probably "غرفة" (bedroom) not "غرفة" (room) — subtle diacritic
- "3.5n" in a dimension context should be "3.5m"
- "Bedroon" is "Bedroom"

The LLM acts as a semantic post-processor that uses document context to resolve ambiguities that are impossible to resolve at the character level.

### 7.2 Model Selection

#### Claude Haiku (claude-haiku-4-5)
- **Input cost:** $0.00025 per 1K tokens (~$0.25/million)
- **Output cost:** $0.00125 per 1K tokens
- **Vision:** Yes (Haiku Vision for image context)
- **Arabic:** Good understanding of Arabic text and architectural terminology
- **Speed:** ~1–3 seconds for a correction pass
- **Context window:** 200K tokens

#### GPT-4o-mini
- **Input cost:** $0.00015 per 1K tokens
- **Output cost:** $0.0006 per 1K tokens
- **Vision:** Yes
- **Arabic:** Good, slightly less consistent than Claude on formal Arabic
- **Speed:** ~1–2 seconds

**Recommendation:** Claude Haiku for better Arabic handling and architectural domain understanding. GPT-4o-mini as fallback if Anthropic API is unavailable.

### 7.3 Prompt Design

The LLM receives:
1. The full image (as base64 in the API call) — gives visual context
2. All extracted text with confidence scores
3. Flagged words (confidence < 0.75)
4. Domain context hint
5. Few-shot examples from the correction database (after initial data collection)

```
System prompt:
  You are an expert in Arabic and English architectural drawing OCR correction.
  You specialize in floor plans, site plans, and architectural annotations.
  Common terms: غرفة النوم (bedroom), الصالة (living room), المطبخ (kitchen),
  الحمام (bathroom), مدخل (entrance), الفناء (courtyard), رواق (corridor),
  الدرج (stairs), نافذة (window), باب (door).
  Return only valid JSON, no explanation.

User prompt:
  Image: [base64 image]
  
  Extracted text with confidence scores:
  [
    {"word": "غرفة", "confidence": 0.92},
    {"word": "النون", "confidence": 0.43, "flagged": true},
    {"word": "3.5n", "confidence": 0.51, "flagged": true},
    {"word": "entrance", "confidence": 0.89}
  ]
  
  For each flagged word, provide the correction and your certainty (0.0–1.0).
  Return format:
  [
    {"original": "النون", "corrected": "النوم", "certainty": 0.96},
    {"original": "3.5n", "corrected": "3.5m", "certainty": 0.99}
  ]
```

### 7.4 Confidence Thresholds and Document Flagging

| OCR Confidence | LLM Certainty | Action | Document Highlight |
|---|---|---|---|
| ≥ 0.75 | — | Accept OCR result | None |
| 0.40–0.74 | ≥ 0.60 | Use LLM correction | Yellow |
| 0.40–0.74 | < 0.60 | Use LLM correction (uncertain) | Red |
| < 0.40 | ≥ 0.60 | Use LLM correction | Yellow |
| < 0.40 | < 0.60 | Keep original, flag for human review | Red |

### 7.5 Token Budget Management

To keep costs predictable:
- Maximum 50 flagged words per LLM call (batch if more)
- Truncate surrounding context to 20 words per flagged word
- Full image is sent once per document (not per word)
- Estimated tokens per image: 400–800 input, 100–200 output

At Claude Haiku pricing: ~$0.0002–0.0004 per image.

---

## 8. Word Document Generation

### 8.1 python-docx Capabilities and Limitations

`python-docx` is the standard Python library for generating .docx files. Relevant capabilities:

**Supported:**
- Absolute positioning of elements (text boxes, images, shapes) via `WD_STYLE` and XML manipulation
- Run-level formatting: bold, italic, color, highlight color, font, size
- RTL paragraph direction (`paragraph.paragraph_format.bidi = True`)
- Inline and floating images
- Basic shapes via DrawingML (rectangles, ellipses, lines, triangles)
- Text box with custom border and fill
- Table insertion

**Limitations:**
- Complex DrawingML shapes (arbitrary polygons) require direct XML injection
- Shape positioning uses EMUs (English Metric Units) — 914400 EMUs = 1 inch
- RTL text in text boxes requires explicit XML manipulation (python-docx doesn't expose all RTL properties in its API)
- No native support for complex shape types beyond basic geometry

### 8.2 Coordinate System Translation

The source image has pixel coordinates. The Word document uses:
- Page dimensions: typically A4 = 210mm × 297mm = 8267px × 11692px at 96 DPI
- Margins: typically 25mm on all sides
- Usable area: 160mm × 247mm

Conversion formula:
```
word_x_emu = (bbox_x_relative × page_width_mm × 36000)  # 36000 EMU per mm
word_y_emu = (bbox_y_relative × page_height_mm × 36000)
```

Where `bbox_x_relative` is the x coordinate normalized to [0, 1] relative to image width.

### 8.3 Text Box Insertion Strategy

For each detected text region:
```python
from docx.shared import Pt, RGBColor, Emu
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def add_text_box(doc, text, x_emu, y_emu, width_emu, height_emu, 
                 is_rtl=False, highlight_color=None):
    # Add floating text box at absolute position
    # Set RTL if Arabic-dominant
    # Apply highlight if flagged
```

### 8.4 Shape Insertion

**Simple geometric shapes** via DrawingML:
```python
def add_word_shape(doc, shape_type, x_emu, y_emu, width_emu, height_emu):
    # shape_type: "ellipse", "triangle", "rect", "line"
    # Uses docx XML manipulation to insert <wps:wsp> element
```

**Complex shapes** as embedded images:
```python
def add_image_at_position(doc, image_path, x_emu, y_emu, width_emu, height_emu):
    # Insert PIL image crop as floating picture
    # Uses docx.Document.add_picture with positioning XML
```

### 8.5 Highlight Colors in Word

Word highlight colors are limited to a fixed palette (Yellow, Cyan, Magenta, Red, DarkRed, DarkBlue, DarkCyan, DarkMagenta, DarkGreen, DarkYellow, DarkGray, LightGray, Black).

- **Yellow** (`WD_COLOR_INDEX.YELLOW`): LLM-corrected with high certainty
- **Red** (`WD_COLOR_INDEX.RED`): LLM-corrected with low certainty / needs human review

### 8.6 RTL Text Handling

```python
def set_rtl_paragraph(paragraph):
    pPr = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement('w:bidi')
    bidi.set(qn('w:val'), '1')
    pPr.append(bidi)
```

Arabic text runs also need the `rtl` property set at the run level for correct rendering in Word.

---

## 9. Feedback Loop and Fine-Tuning

### 9.1 Data Storage Schema

```sql
CREATE TABLE correction_events (
    id          UUID PRIMARY KEY,
    created_at  TIMESTAMP,
    image_hash  VARCHAR(64),      -- SHA256 of source image
    word_crop   BYTEA,            -- cropped image of the word region
    ocr_text    VARCHAR(500),     -- what PaddleOCR returned
    llm_text    VARCHAR(500),     -- what Claude Haiku suggested
    final_text  VARCHAR(500),     -- what the user confirmed/corrected
    language    VARCHAR(10),      -- 'ar', 'en', 'mixed'
    confidence_ocr   FLOAT,
    confidence_llm   FLOAT,
    context_words    TEXT         -- JSON array of surrounding words
);
```

### 9.2 Three-Layer Improvement

#### Layer 1: Immediate (After first correction)
Every user correction is stored. No model changes yet.

#### Layer 2: Few-Shot Prompting (After ~50 corrections)
When calling Claude Haiku for correction, prepend the top-N most similar historical corrections as examples:

```
Here are examples of past corrections in this domain:
- "entrnce" → "entrance" (certainty: 0.99)
- "غرفه النون" → "غرفة النوم" (certainty: 0.97)
- "3.5n" → "3.5m" (certainty: 0.99)
...
Now correct the following:
```

Similarity: fuzzy string match (Levenshtein distance) between current flagged word and historical originals.

#### Layer 3: PaddleOCR Fine-Tuning (After ~500 corrections)
PaddleOCR supports custom training via its training pipeline:

1. Prepare training data:
   ```
   /data/train/
     images/
       word_001.png  # cropped word image
       word_002.png
     labels.txt      # "word_001.png\tالنوم\n"
   ```

2. Fine-tune the recognition model:
   ```bash
   python tools/train.py \
     -c configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml \
     -o Global.pretrained_model=./pretrain_models/PP-OCRv4_rec \
        Train.dataset.data_dir=./data/train \
        Train.dataset.label_file_list=./data/train/labels.txt
   ```

3. The fine-tuned model replaces the base model in the pipeline.

Expected improvement: ~30–40% reduction in WER for domain-specific handwriting after 500+ training pairs.

### 9.3 Feedback API Endpoint

```
POST /feedback
Content-Type: application/json

{
  "document_id": "uuid",
  "corrections": [
    {
      "element_id": "text_042",
      "original_ocr": "entrnce",
      "llm_suggestion": "entrance",
      "user_final": "entrance",
      "accepted_llm": true
    }
  ]
}
```

The .NET system collects user corrections through its review UI and POSTs them to this endpoint.

---

## 10. Cost Analysis

### 10.1 Per-Image Cost Breakdown

| Component | Unit Cost | Per Image | Notes |
|---|---|---|---|
| PaddleOCR | $0.00 | $0.00 | Runs locally |
| OpenCV | $0.00 | $0.00 | Runs locally |
| Claude Haiku (text correction) | $0.25/M input tokens | ~$0.00025 | ~500 tokens avg per image |
| Claude Haiku Vision (shape labeling) | ~$0.0004/image | ~$0.0003 | 1–2 shape crops avg |
| Cloud hosting (compute) | Varies | ~$0.001 | 2 vCPU, 4GB RAM instance |
| **Total** | | **~$0.001–$0.002** | |

### 10.2 Scale Projections

| Monthly Volume | LLM Cost | Compute Cost | Total/Month |
|---|---|---|---|
| 1,000 images | $0.25–$0.50 | $10–20 | $10–21 |
| 10,000 images | $2.50–$5.00 | $20–40 | $23–45 |
| 100,000 images | $25–$50 | $50–100 | $75–150 |

### 10.3 Cost Optimization Strategies

1. **Skip LLM for high-confidence images**: If overall OCR confidence > 0.90, skip LLM correction entirely → saves ~$0.00025 per image
2. **Cache shape crops**: If the same symbol appears multiple times (common in drawings), label once and cache → saves Haiku Vision calls
3. **Batch LLM calls**: Group multiple images' corrections into a single API call with higher token limit → reduces per-call overhead
4. **Use Claude Haiku without vision for text-only corrections**: Vision tokens cost more — only send the image when there are many low-confidence words that need visual context

---

## 11. Architecture Decision

### 11.1 Final System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    .NET Application                          │
│  ┌──────────────┐    POST /process      ┌────────────────┐  │
│  │  Image Input │─────────────────────▶│  OCR API       │  │
│  └──────────────┘                       │  (FastAPI)     │  │
│  ┌──────────────┐    ◀── .docx + JSON ──│                │  │
│  │  Review UI   │                       └────────────────┘  │
│  │  (Correction)│    POST /feedback           │             │
│  └──────┬───────┘─────────────────────────────▶            │
└─────────┼───────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                   OCR Microservice (Docker)                   │
│                                                               │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ Preprocess │→ │Layout Segment│→ │Shape Detection     │   │
│  │ (OpenCV)   │  │(PaddleOCR)   │  │(OpenCV contours)   │   │
│  └────────────┘  └──────────────┘  └─────────┬──────────┘   │
│                         │                     │               │
│                         ▼                     ▼               │
│                  ┌────────────┐    ┌──────────────────┐      │
│                  │  OCR       │    │ Simple: Word shape│      │
│                  │ (PaddleOCR)│    │ Complex: crop img │      │
│                  └─────┬──────┘    └──────────────────┘      │
│                        │                                      │
│                        ▼                                      │
│                 ┌─────────────┐                              │
│                 │ LLM Correct │ ← Claude Haiku (Vision)      │
│                 │ (low conf)  │                               │
│                 └──────┬──────┘                              │
│                        │                                      │
│                        ▼                                      │
│              ┌──────────────────┐                            │
│              │  Word Assembly   │                             │
│              │  (python-docx)   │                             │
│              └──────────────────┘                            │
│                        │                                      │
│              ┌──────────────────┐                            │
│              │  JSON Sidecar    │                             │
│              │  Generator       │                             │
│              └──────────────────┘                            │
│                                                               │
│  ┌─────────────────────────────────────────────┐            │
│  │  Correction DB (PostgreSQL)                  │            │
│  │  + Few-shot cache + Training data store      │            │
│  └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### 11.2 API Specification

#### POST /process
```
Request:
  Content-Type: multipart/form-data
  Fields:
    image: file (PNG, JPG, TIFF, PDF single page)
    language_hint: string (default: "ar+en")
    confidence_threshold: float (default: 0.75)
    label_shapes: bool (default: true)

Response:
  Content-Type: application/json
  {
    "document_id": "uuid",
    "docx": "<binary base64 or stream>",
    "sidecar": {
      "page_dimensions": {"width_px": 2480, "height_px": 3508},
      "elements": [
        {
          "id": "text_001",
          "type": "text",
          "bbox": {"x": 0.10, "y": 0.05, "w": 0.35, "h": 0.03},
          "content": "غرفة النوم",
          "language": "arabic",
          "confidence": 0.91,
          "llm_correction": null,
          "highlight": null
        },
        {
          "id": "shape_001",
          "type": "simple_shape",
          "shape": "circle",
          "bbox": {"x": 0.30, "y": 0.20, "w": 0.08, "h": 0.08},
          "confidence": 0.97
        },
        {
          "id": "shape_002",
          "type": "complex_shape",
          "shape": "door swing",
          "bbox": {"x": 0.50, "y": 0.30, "w": 0.10, "h": 0.08},
          "embedded_as": "image",
          "llm_label": "door swing",
          "llm_label_certainty": 0.94
        },
        {
          "id": "text_042",
          "type": "text",
          "content": "entrance",
          "confidence_original": 0.48,
          "llm_correction": {
            "original": "entrnce",
            "corrected": "entrance",
            "certainty": 0.97
          },
          "highlight": "yellow"
        }
      ],
      "stats": {
        "total_elements": 47,
        "text_elements": 32,
        "simple_shapes": 8,
        "complex_shapes": 7,
        "llm_corrections": 4,
        "processing_time_ms": 2340
      }
    }
  }
```

#### POST /feedback
```
Request:
  Content-Type: application/json
  {
    "document_id": "uuid",
    "corrections": [
      {
        "element_id": "text_042",
        "user_final": "entrance"
      }
    ]
  }

Response: 204 No Content
```

#### GET /health
```
Response: {"status": "ok", "models_loaded": true}
```

---

## 12. Challenges and Mitigations

### 12.1 Low Image Quality

**Problem:** Scanned drawings at 75 DPI, crumpled paper, coffee stains, faded ink  
**Mitigation:**
- DPI upscaling: if detected DPI < 150, apply super-resolution (ESRGAN or Real-ESRGAN, free open-source)
- Adaptive binarization (Sauvola's method) works better than global thresholding on uneven illumination
- Morphological operations to remove noise spots
- Return a quality score in the sidecar so the .NET system can warn the user

### 12.2 Heavily Overlapping Text and Shapes

**Problem:** On complex floor plans, dimension lines pass through text labels  
**Mitigation:**
- Run OCR before shape detection (get text bounding boxes)
- Mask text regions when doing shape detection
- For remaining ambiguous regions, flag as "uncertain" in the sidecar

### 12.3 Very Small Text (Labels, Dimensions)

**Problem:** Dimension values like "1200" written in 6pt size on a full A1 drawing scanned at 150 DPI  
**Mitigation:**
- Detect text region size before OCR
- For regions where text height < 10px, run a super-resolution pass on just that crop before OCR
- Flag small text in sidecar for human review

### 12.4 Rotated Annotations

**Problem:** Side notes written at 90° or 45° angles  
**Mitigation:**
- PaddleOCR's direction classifier handles 0°, 90°, 180° rotations automatically
- For arbitrary angles: detect line skew angle via Hough transform, rotate crop, OCR, then rotate back for position
- Annotations at angles other than 0/90/180 are rare in professional drawings — flag as complex and return as image crop

### 12.5 Arabic Diacritics (Tashkeel)

**Problem:** Tashkeel (ً ٌ ٍ َ ُ ِ ّ ْ) appear very close to base letters and are often confused with noise or merged  
**Mitigation:**
- PaddleOCR v4 handles tashkeel better than v3
- Post-process: if a word has low confidence and contains suspected tashkeel, try OCR with and without diacritic normalization (strip tashkeel, re-run, compare)
- LLM correction is effective here because it knows the word's semantic meaning

### 12.6 Memory Usage for Large Images

**Problem:** Full A0/A1 architectural drawings at 300 DPI can be 200MB+ images  
**Mitigation:**
- Tile processing: split into overlapping tiles, process independently, merge results
- Tile size: 2000×2000px with 200px overlap
- Maximum input resolution: cap at 4000×4000px (downsample if larger, preserving OCR-relevant regions)

### 12.7 Word Document Positioning Accuracy

**Problem:** Word's absolute positioning model differs from image coordinates — elements may drift  
**Mitigation:**
- Use text boxes with fixed anchor points rather than inline content
- Round positions to the nearest 5mm grid (sufficient for "approximately correct" requirement)
- Add a canvas-style background rectangle at page level to constrain floating elements

---

## 13. Implementation Roadmap

### Phase 1 — Core OCR Pipeline (Weeks 1–3)
- FastAPI skeleton with `/process` and `/health` endpoints
- OpenCV preprocessing (deskew, denoise, binarize)
- PaddleOCR integration (Arabic + English, bounding boxes)
- Basic Word document output (text only, no shapes, no absolute positioning)
- Docker containerization

**Deliverable:** API returns a Word doc with all text, in reading order, no positioning

### Phase 2 — Layout and Positioning (Weeks 4–5)
- PaddleOCR layout analysis integration
- RTL reading order reconstruction (`python-bidi`, `arabic-reshaper`)
- Absolute text box positioning in Word document
- Coordinate system translation (pixel → EMU)
- Side note detection and attachment

**Deliverable:** API returns a Word doc with text approximately positioned

### Phase 3 — Shape Detection (Weeks 6–7)
- OpenCV contour detection and filtering
- Geometric classification (circle, triangle, rectangle)
- Word vector shape insertion (DrawingML)
- Complex shape detection and image cropping
- Image embedding in Word document
- Claude Haiku Vision integration for shape labeling

**Deliverable:** API returns a Word doc with text + shapes + embedded images

### Phase 4 — LLM Correction (Week 8)
- Claude Haiku integration
- Confidence threshold logic
- Full image context in API call
- Yellow/red highlighting in Word
- JSON sidecar generation

**Deliverable:** Full pipeline with LLM correction and complete JSON sidecar

### Phase 5 — Feedback Loop (Week 9)
- PostgreSQL database setup
- `/feedback` endpoint
- Few-shot example retrieval and injection into LLM prompt
- Correction storage and deduplication

**Deliverable:** Feedback API operational, few-shot improving over time

### Phase 6 — Hardening (Week 10)
- Error handling (corrupted images, timeouts, oversized files)
- Input validation and sanitization
- Rate limiting
- Quality score in sidecar
- Load testing
- PaddleOCR fine-tuning pipeline (for when correction data accumulates)

**Deliverable:** Production-ready microservice

---

## 14. References and Tools

### Libraries
| Library | Version | Purpose |
|---|---|---|
| `paddlepaddle` | 2.6+ | PaddleOCR backend |
| `paddleocr` | 2.7+ | OCR engine + layout analysis |
| `opencv-python` | 4.9+ | Image preprocessing + contour detection |
| `Pillow` | 10+ | Image I/O and manipulation |
| `python-docx` | 1.1+ | Word document generation |
| `fastapi` | 0.110+ | REST API framework |
| `uvicorn` | 0.29+ | ASGI server |
| `anthropic` | 0.25+ | Claude Haiku SDK |
| `python-bidi` | 0.4+ | Bidirectional text reordering |
| `arabic-reshaper` | 3.0+ | Arabic glyph reshaping |
| `langdetect` | 1.0+ | Per-word language detection |
| `psycopg2` | 2.9+ | PostgreSQL driver for feedback DB |

### Research Papers
- **PP-OCRv3**: "PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System" (Baidu, 2022)
- **PaddleOCR Layout**: "PP-Structure: An End-to-End Document Understanding System" (Baidu, 2022)
- **DBNet++**: "Real-Time Scene Text Detection with Differentiable Binarization" (2022)
- **Arabic OCR Survey**: "A Survey on Arabic Optical Character Recognition: Challenges and Opportunities" (2021)
- **Bidirectional Algorithm**: Unicode Standard Annex #9 — Unicode Bidirectional Algorithm

### Benchmark Datasets
- **APTI** (Arabic Printed Text Image): 113,000 Arabic word images, for printed Arabic OCR evaluation
- **AHDB** (Arabic Handwriting Database): Handwritten Arabic text benchmark
- **COCO-Text**: Mixed printed/handwritten, useful for layout model evaluation

### Arabic Architectural Terminology Reference
Common terms that should be included in the LLM system prompt:
- غرفة النوم (bedroom), الصالة (living room), المطبخ (kitchen)
- الحمام (bathroom), دورة المياه (WC), المدخل (entrance)
- الفناء (courtyard), الرواق (corridor), الدرج (stairs)
- الطابق (floor/level), السقف (ceiling/roof), الجدار (wall)
- النافذة (window), الباب (door), الفتحة (opening)
- الحديقة (garden), الموقف (parking), المستودع (storage)
- الواجهة (facade), المسقط الأفقي (floor plan), القطاع (section)
