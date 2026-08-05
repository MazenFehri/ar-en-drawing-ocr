# Arabic Architectural OCR API

FastAPI microservice that accepts architectural drawing images (Arabic/English, handwritten or printed) and returns a positioned Word `.docx` document plus a JSON sidecar.

Every word and every shape lands where it sat on the original page. Text boxes are real
Word text boxes, and recognised geometry is real DrawingML — you can select, move and
resize a wall or a circle in Word, not just look at a picture of one. Only ink that
genuinely cannot be named is embedded as a raster, so nothing on the page is dropped.

Built as a server-to-server component for a .NET application. Runs OCR locally: no
per-page cost, and drawings never leave the machine. The vision LLM is an optional
reviewer for low-confidence words only — if the provider is down or rate-limited, the
pipeline still returns the document from raw OCR and says so in the sidecar.

```bash
docker compose up --build
python try_it.py sample_drawing.png
```

---

## What it produces

### A mixed Arabic/English floor plan

Input on the left, the generated `.docx` opened in Word on the right. Nothing was
hand-corrected — this is one `try_it.py` run.

| Input image | Generated Word document |
|---|---|
| ![input](docs/images/floor-plan-input.png) | ![output](docs/images/floor-plan-output.png) |

**82 elements in 5.6s** — 44 text, 37 shapes, 1 raster. Reproduce it with
`python try_it.py docs/images/floor-plan-input.png`.

Worth noting in the output:

- **The Arabic notes paragraph came through whole**, at 98% and 94%, in logical order —
  not as reversed fragments.
- **The area schedule survives as a table**, because each cell border is detected as its
  own rectangle and positioned independently. The pipeline has no table model; the grid
  is an emergent result of accurate rectangle placement.
- **Every shape is a real Word shape.** Click a room, a `WC` circle or the north-arrow
  triangle in Word and you get resize handles, not a picture.
- **The freehand revision mark** (right of the plan, labelled `rev.`) is the one thing
  that couldn't be named. It's embedded as a cropped raster at its original position
  rather than dropped — that's the designed fallback, working.

Honest about the misses: the dimension **tick marks** on the `14.80 m` line were not
recovered as line shapes, and two Arabic sentences lost the space at a full stop
(`التنفيذ. سماكة` → `التنفيذسماكة`). Both are visible in the screenshot above.

### The simpler bundled sample

| Input image | Generated Word document |
|---|---|
| ![input](docs/images/sample-input.png) | ![output](docs/images/sample-output.png) |

**19 elements in 5.0s**, all 12 text elements at 92–100% confidence. The irregular
pentagon has no preset geometry to map to, so it becomes an embedded raster.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/process` | Upload image → get `.docx` + JSON sidecar |
| `POST` | `/feedback` | Submit user corrections for future improvement |

---

## Option A — Docker (recommended)

No local Python or PostgreSQL setup required.

```bash
cp .env.example .env      # then put your OpenRouter key in it
docker compose up --build
```

The service runs without a key — LLM correction is optional and its absence is reported
as `llm_status.state: "not_attempted"`, `reason: "not_configured"`.

The build bakes in the three model weights (~31 MB total). Subsequent starts are fast.

```bash
# Verify it's up
curl http://localhost:8000/health
```

---

## Option B — Local Python

**Requirements:** Python 3.11, PostgreSQL.

### 1. Create virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# or
source .venv/bin/activate     # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> PaddleOCR downloads the three model weights (~31 MB total) automatically on first use,
> into `~/.paddlex/official_models`.

### 3. Configure environment

Copy `.env.example` to `.env` and fill it in:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free
OPENROUTER_FALLBACK_MODELS=google/gemma-4-31b-it:free,nvidia/nemotron-nano-12b-v2-vl:free
CONFIDENCE_THRESHOLD=0.75
LABEL_SHAPES=true
DATABASE_URL=postgresql://ocr:ocr@localhost:5432/ocr_db
LOG_LEVEL=INFO
```

> **Note on the LLM model:** `OPENROUTER_MODEL` must be a currently-valid,
> vision-capable model ID. OpenRouter rotates its free models over time — if you
> get a `400 not a valid model ID` error, list the current free vision models at
> <https://openrouter.ai/models?modality=text%2Bimage&max_price=0> and update the
> value. LLM correction is an **optional enhancement**: if the provider is
> unavailable or rate-limited (HTTP 429 is common on free tiers), the pipeline
> logs a warning and still returns the document using the raw OCR text.

### 4. Create PostgreSQL database

```sql
CREATE USER ocr WITH PASSWORD 'ocr';
CREATE DATABASE ocr_db OWNER ocr;
```

### 5. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

The `correction_events` table is created automatically on startup.

### 6. Run tests

```bash
pytest
```

---

## Usage

### Easiest: interactive Swagger UI

Open <http://localhost:8000/docs> in a browser → **POST /process** → **Try it
out** → choose an image → **Execute**. No command line needed — recommended on
Windows, where PowerShell's `curl` is an alias for `Invoke-WebRequest` and does
not accept `-X`/`-F` flags.

**Set `response_format` to `docx`** unless you specifically want the JSON. The
default `json` returns the document as a base64 string you then have to decode by
hand; `docx` gives you a file you can open in Word straight from the browser.

### Easier still: one command

```bash
python try_it.py your_drawing.jpg      # omit the filename to use sample_drawing.png
python try_it.py your_drawing.jpg --ar     # Arabic only, skips the Latin reader
python try_it.py your_drawing.jpg --en     # English only
python try_it.py your_drawing.jpg --label  # also ask the LLM to name unrecognised shapes
```

Writes `your_drawing.docx` and `your_drawing.sidecar.json` next to the image — named
after the input, so testing a second image doesn't overwrite the first — and prints
every element found with its confidence. To watch the LLM reviewer actually run, raise
the threshold; at the `0.75` default a clean image flags almost nothing:

```bash
# in try_it.py, or as a -F field on the curl call below
confidence_threshold=0.99
```

### Scan resolution is the biggest lever on accuracy

`try_it.py` prints a loud warning, and the sidecar sets `low_resolution: true`, when the
median text line is under 20px tall. Arabic letters are distinguished by dots 1–2px
across at that size — they are not sampled at all, and the reading is partly guesswork
however confident the scores look. **Scan at 200–300 DPI.** No OCR model recovers what
the scan never captured.

### Process an image (curl)

```bash
curl -X POST http://localhost:8000/process \
  -F "image=@your_drawing.jpg" \
  -F "confidence_threshold=0.75" \
  -F "language_hint=ar+en" \
  -F "label_shapes=true"
```

| Field | Default | Meaning |
|---|---|---|
| `image` | required | The drawing. jpeg/png/tiff/bmp, max 25 MB. **WebP is refused** — see below. |
| `confidence_threshold` | `0.75` | Words below this go to the LLM for correction. Must be 0.0–1.0. |
| `language_hint` | `ar+en` | One of `ar+en`, `ar`, `en`. Picks the OCR model; `ar+en` also reads Latin and digits. |
| `label_shapes` | from env | Ask the vision model to name complex shape crops (`llm_label` in the sidecar). |
| `response_format` | `json` | `docx` returns the Word file itself as a download, with the sidecar stats in the `X-Sidecar-Stats` header. `json` returns the base64 payload below. |

Raising `confidence_threshold` sends more words to the LLM: slower, and on a free
model it will hit rate limits. `0.75` is the useful default.

Response:
```json
{
  "document_id": "uuid",
  "docx_base64": "<base64-encoded .docx>",
  "sidecar": {
    "document_id": "uuid",
    "page_dimensions": { "width_px": 1920, "height_px": 1080 },
    "elements": [...],
    "stats": {
      "total_elements": 42,
      "text_elements": 30,
      "simple_shapes": 8,
      "complex_shapes": 4,
      "llm_corrections": 5,
      "processing_time_ms": 1200,
      "quality_score": 0.87,
      "review_queue": ["text_014", "text_003", "text_021"],
      "llm_status": { "state": "success", "reason": null, "model": "google/gemma-4-26b-a4b-it:free" },
      "shape_label_status": { "state": "failed", "reason": "rate_limited", "model": null }
    }
  }
}
```

**The LLM can never rewrite a number.** A correction whose digit runs differ from what OCR read
is refused outright and the line ships flagged for review instead. Measured against a live
model on an Arabic worksheet, it returned `45` → `43` at certainty **1.00**, and separately
dropped a `27250` that the two-recogniser splice had just recovered — so certainty cannot be
what gates this. Corrections that change only letters are applied normally. Likewise a
correction at certainty `0.0` is treated as the model declining to read the crop, not as an
answer, whatever string it returned.

**`review_queue` is the list to work through**, most urgent first — element ids whose text
nothing has verified. Numeric disagreements lead it (a line where two recognisers disagreed
about a number can still carry a *high* confidence, so ordering by confidence alone buries the
one error that makes a maths worksheet wrong); edge marking-column cells sink to the back,
since they are empty printed score boxes that no model can read; everything else is ordered
least-confident first. Per element, `digit_disagreement` and `margin_column` say which case
applies, and `digits_recovered` lists numbers the second recogniser put back.

**Always check `llm_status` before trusting `llm_corrections: 0`.** Zero corrections
means either "the model confirmed every word" (`state: "success"`) or "the model never
ran" (`state: "failed"` / `"not_attempted"`) — the counts alone cannot tell them apart.

| `state` | `reason` |
|---|---|
| `success` | `null`; `model` names the model that answered |
| `not_attempted` | `no_flagged_words`, `no_shapes_to_label`, `not_configured` |
| `failed` | `rate_limited`, `invalid_model`, `unauthorized`, `network`, `server_error`, `parse_error`, `empty_response`, `timed_out`, `unknown` |

`empty_response` means the provider returned HTTP 200 with no usable completion —
OpenRouter's gateway does this when an upstream provider fails, putting the real reason
in an `error` field rather than a non-200 status. The provider's message is extracted
and logged; check the container log for it. Distinct from `parse_error`, which means we
got content and could not parse it.

LLM work is bounded to `TOTAL_LLM_BUDGET_SECONDS` (60s) — a real wall-clock ceiling,
enforced by running each model call under a `thread.join(timeout=...)`, not just a
check between attempts. That distinction matters: a plain per-request timeout can't
guarantee this, because httpx (which the OpenAI SDK sits on) only exposes phase
timeouts — connect/read/write/pool — never a total one, and its read timeout resets on
every byte received. A provider that trickles keep-alive bytes while a request sits
queued can hold a read timeout open indefinitely; only a wall-clock join actually cuts
a stuck call off. When the budget expires the pipeline returns the document built from
raw OCR text and reports `timed_out` — it never fails the request.

The 60s applies per call — one shared deadline across every chunk, model, and retry
within a single call to word correction, and separately within a single call to shape
labelling. The two are independent optional stages, so a request that exercises both
and has both time out end to end can take up to 2x this budget (~120s), not a hard
120s cap on the whole request. A stuck call's worker thread is abandoned rather than
killed (Python can't forcibly stop a thread) — it's a daemon thread, so it can't block
the process from shutting down, but it does keep holding its socket in the background
until the call eventually errors out on its own.

Decode the `.docx`:
```python
import base64, pathlib
data = response.json()
pathlib.Path("output.docx").write_bytes(base64.b64decode(data["docx_base64"]))
```

### Submit corrections

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "uuid-from-process-response",
    "corrections": [
      { "element_id": "el_001", "user_final": "corrected text" }
    ]
  }'
```

---

## Architecture

```
image → preprocess → OCR (one detection pass, per-crop recognition) → shape detection
      → LLM correction (vision model via OpenRouter)
      → layout reconstruction → Word assembly → JSON sidecar
```

- **OCR:** PaddleOCR 3.7, running three models directly rather than a bundled pipeline.
  `PP-OCRv6_tiny_det` segments the page **once** (it is language-agnostic); each detected
  polygon is then perspective-cropped and read by `PP-OCRv6_small_rec`, with
  `arabic_PP-OCRv5_mobile_rec` re-reading only the crops whose Latin confidence fell
  below 0.90 (~2/89 on class-diagram, 5/12 on sample_drawing). The winner per crop is
  chosen by script, not by confidence — the Arabic model mangles Latin while still
  scoring it highly, so it only wins when it actually returned Arabic.
  This replaced two full detect+rec passes whose disagreeing segmentations had to be
  reconciled by a bbox-overlap union-find; with one segmentation there is nothing to
  merge, and that code is gone.
- **LLM correction:** OpenRouter free vision model (configurable, optional)
- **Shape detection:** OpenCV contour analysis with `RETR_CCOMP` so nested content
  (interior walls, fixtures, furniture inside a room outline) survives — `RETR_EXTERNAL`
  keeps only the outermost contour of each blob and discards everything within it.
  Whatever is left unclassified is captured as **residual ink**: dilated, grouped into
  connected components and embedded as positioned images, so drawing we cannot name
  still appears in the output.
- **Page geometry:** element bboxes are relative fractions, so the page is **letterboxed**
  to the source aspect ratio (and flipped to landscape for wide drawings). Mapping `x` by
  page width and `y` by page height independently stretches any non-A4-shaped input.
- **Word output:** Absolutely positioned DrawingML objects (text boxes, shapes, embedded images).
  Simple shapes render outline-only — Word's default shape style is a solid blue fill, which
  would hide the text underneath.
- **Arabic text:** `arabic_PP-OCRv5_mobile_rec` returns whole phrases already in
  *logical* (Unicode storage) order, which is what the sidecar and Word both want, so the
  pipeline does **no** bidi reordering. Under PP-OCRv4 it did: that model emitted visual
  order (the order glyphs sit on the page) and `utils.bidi.to_logical_order` reversed it.
  That function is deleted — applying it now would be a second correction on a correct
  string and would silently ship reversed Arabic. Reading order still groups words into
  lines and sorts each line in its own direction, right-to-left for Arabic.
- **Storage:** PostgreSQL via asyncpg for correction feedback

---

## Further reading

| Document | What's in it |
|---|---|
| [`md_files/PROJECT_REPORT.md`](md_files/PROJECT_REPORT.md) | Stage-by-stage design, every measurement behind a tuning constant, and the known limits |
| [`md_files/research.md`](md_files/research.md) | The prior-art survey the architecture was chosen from — Arabic OCR options, layout analysis, cost analysis |
| [`docs/superpowers/specs/`](docs/superpowers/specs/) | Original design spec |
