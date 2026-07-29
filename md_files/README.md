# Arabic Architectural OCR API

FastAPI microservice that accepts architectural drawing images (Arabic/English, handwritten or printed) and returns a positioned Word `.docx` document plus a JSON sidecar.

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
docker compose up --build
```

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

The `.env` file in the project root is already populated. If starting fresh, set these variables:

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
```

Writes `output.docx` and `sidecar.json` next to the image and prints every element
found with its confidence. To watch the LLM reviewer actually run, raise the
threshold — at the `0.75` default a clean image flags almost nothing:

```bash
# in try_it.py, or as a -F field on the curl call below
confidence_threshold=0.99
```

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
      "llm_status": { "state": "success", "reason": null, "model": "google/gemma-4-26b-a4b-it:free" },
      "shape_label_status": { "state": "failed", "reason": "rate_limited", "model": null }
    }
  }
}
```

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

## Troubleshooting

**`NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support ...` at
`onednn_instruction.cc:116` when a model runs.**
Stock paddle 3.3.1 is dead on CPUs without AVX-512 (Intel 12th/13th/14th-gen consumer
parts — Alder Lake / Raptor Lake — have it fused off). The **only** fix that works is
passing `enable_mkldnn=False` to every `TextDetection` / `TextRecognition` constructor,
which `pipeline/ocr.py` does. Verified not to help: `ir_optim=False`, the
`FLAGS_enable_pir_api=0` env var, and the `FLAGS_use_mkldnn=0` env var (silently ignored
by paddle 3.x, which is why it is no longer set in the Dockerfile). Miss one constructor
and it is an immediate hard crash, not a slow path.

The older `SIGILL` / `Illegal instruction` crash on this hardware was a paddle 2.6.x
problem (Paddle#76111) handled by a `pipeline/_paddle_patch.py` monkeypatch. It does not
reproduce on 3.3.1, and that module is deleted.

**First request is slow (~6s), later ones are faster.**
Model weights (~31 MB for the three models) are baked into the image and cached in the
`paddlex_models` Docker volume at `~/.paddlex/official_models`, so they are not
re-downloaded. The first request still pays predictor construction. Note the cache path
moved from 2.x's `~/.paddleocr`; the volume was renamed alongside it, because a volume
holding the old v4 weights would have been mounted over the new ones rather than
re-seeded. (Never run `docker compose down -v` to clean up — it also drops `pgdata`.)

**opencv.** There is exactly one opencv distribution now: paddlex requires
`opencv-contrib-python==4.10.0.84` and nothing else in the graph wants an opencv, so
`requirements.txt` pins none. Under 2.7.3 three distributions fought over
`site-packages/cv2` and install order decided which won. `cv2.ximgproc.createEdgeDrawing`
(needed by `pipeline/shape_detector.py`) is therefore present by construction; the
Dockerfile still asserts it at build time.

**Why WebP uploads are still rejected (HTTP 400).**
No longer a live mitigation: opencv 4.10.0.84 carries a libwebp with CVE-2023-4863 fixed
(the fix landed in 4.8.1.78). The guard is kept as defence-in-depth — WebP is not a
format architectural drawings arrive in, so refusing it costs nothing, and it is checked
by `RIFF....WEBP` magic bytes rather than `Content-Type`, because `cv2.imdecode` detects
format from content.

**Known advisories.** The two `paddlepaddle 2.6.2` command-injection advisories
(PYSEC-2026-1754/-1756) are resolved by this migration — they were fixed on the 3.x line,
and the tree is now on `paddlepaddle 3.3.1`.
