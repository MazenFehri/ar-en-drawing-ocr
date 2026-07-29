# Arabic + English Drawing OCR — Project Report

An HTTP microservice that turns a scanned architectural or technical drawing into an
editable Word document that keeps the original layout, plus a JSON sidecar describing
every element it found. Called server-to-server from a .NET application.

Input: one image (JPEG / PNG / TIFF / BMP).
Output: a `.docx` where every word and every shape sits where it sat on the page, and a
JSON sidecar with coordinates, confidence, and per-stage status.

---

## The pipeline

Seven stages, run in order by `pipeline/__init__.py::process_image`.

| # | Stage | File | Tech | What it does, and why |
|---|-------|------|------|----------------------|
| 1 | Preprocess | `pipeline/preprocessor.py` | OpenCV | Downscale to 2000px longest side, deskew, denoise, CLAHE contrast. The 2000px cap is a measured sweet spot — see *Measurements*. Deskew refuses to rotate past 15°, because a diagonal section line makes detection report ~45° and rotating would wreck the page. |
| 2 | OCR | `pipeline/ocr.py` | PaddleOCR 3.7: **PP-OCRv6** det + rec, **PP-OCRv5** Arabic rec | One language-agnostic detection pass, then per-crop recognition. The Latin recogniser runs on every crop; the Arabic one only where Latin confidence is below 0.90 (measured: 2/89 crops on the class diagram, 5/12 on the floor plan). Runs *before* shape detection so its boxes can mask text. |
| 3 | Shape detection | `pipeline/shape_detector.py` | OpenCV contours + **EdgeDrawing** | Closed outlines and filled blobs go to the contour classifier; everything else goes to a line-segment pass that merges collinear strokes and assembles rectangles. Only genuinely un-nameable ink is cropped to a raster. |
| 4 | Layout reconstruction | `pipeline/layout_reconstructor.py` | — | Pixels → relative coordinates, reading order, and word↔shape association. No bidi reordering: the Arabic recogniser already returns logical order. |
| 5 | LLM correction | `pipeline/llm_corrector.py` | OpenRouter (OpenAI SDK) | Re-reads low-confidence words from cropped images. Words it never covers stay highlighted rather than shipping as certain. |
| 5b | Shape labelling | `pipeline/llm_corrector.py` | Same | Names unclassifiable shapes ("door swing", "north arrow"). |
| 6 | Word assembly | `pipeline/word_assembler.py` | python-docx + raw DrawingML | Builds the `.docx` with absolutely-positioned elements. |
| 7 | Sidecar | `pipeline/sidecar.py` | — | JSON describing every element, plus stats and per-stage status. |

---

## Features in detail

### Shape and drawing capture

Everything that isn't text is captured and placed. Recognisable geometry becomes a real
Word shape; anything else is embedded as a cropped image at its original position.

Ink is routed by what it actually is, in three passes:

- **Region pass.** Contours with solidity ≥ 0.60 — closed outlines and filled blobs — are
  classified by vertex count, extent, circularity and solidity →
  `circle · ellipse · triangle · rect · square`. Otsu thresholding rather than a fixed
  cutoff, since drawings are bimodal. `RETR_CCOMP` rather than `RETR_EXTERNAL`, which used
  to discard everything inside an outer outline; hole rejection drops the inner contour of
  a stroke drawn as two parallel lines.
- **Stroke pass.** `cv2.ximgproc.createEdgeDrawing` finds line segments, collinear ones are
  merged, and rectangles are assembled from parallel-pair 4-cycles. This exists because
  `findContours` traces *region boundaries* and so structurally cannot represent a stroke —
  a box fused to the connectors touching it becomes one snake-shaped contour with solidity
  ~0.001 and no rectangle to find. Before this pass the class diagram yielded **zero**
  rectangles; it now yields 11 of 11. Leftover strokes become open polylines, emitted as
  real DrawingML freeforms (`a:custGeom`) rather than pictures.
- **Residual catch-all.** Whatever is left gets dilated, grouped into connected components,
  and emitted as cropped images — so hand drawings and odd symbols don't silently vanish.
  Raster embedding is reserved for ink that genuinely cannot be named: on the class diagram
  the split went from 54 rasters / 6 vectors to 15 / 74.

### Text is never duplicated into a shape

Text regions are masked out of the image that crops are cut from, not just out of the
image contours are found in. Without this, a box containing a label carried that label
into the embedded picture and the document showed the words twice.

The erased area is filled with the **median colour of a thin ring just outside it**, so a
colour-filled box stays that colour instead of getting a white punch-out.

Masking also punches holes in colour-filled boxes, which `RETR_CCOMP` reports as nested
contours — those are recognised as self-inflicted and skipped, otherwise every word inside
a filled box came back as a phantom rectangle.

### Layout fidelity

- **Relative coordinates.** Every bbox is a fraction of page size, so it survives the
  preprocessing downscale.
- **Letterboxing.** The source aspect ratio is fitted inside the page and centred. Mapping
  x by page width and y by page height independently stretched the drawing — circles came
  out as ellipses.
- **Z-order bands.** Text is anchored in a higher `relativeHeight` band than graphics, so a
  label inside a box renders *on top of* it. Previously all anchors shared one value and
  Word fell back to document order, painting shapes over text and hiding it.
- **Font sized to its box.** Font size is derived from box height so text fills the space
  OCR measured rather than being clipped by it.

### Arabic and bidirectional text

- **No bidi reordering.** `arabic_PP-OCRv5_mobile_rec` returns logical (Unicode storage)
  order already, and as whole phrases. PP-OCRv4 did not — it emitted visual order, and
  `utils.bidi.to_logical_order` reversed it to compensate. That function is deleted:
  applying it now would be a second correction, silently shipping reversed Arabic in every
  document. The failure mode is invisible — reversed Arabic still renders as Arabic and
  still passes `is_arabic()` — so it is pinned by a codepoint-level test, not a visual one.
- Word paragraphs get `<w:bidi/>`; Arabic lines are sorted right-to-left in reading order.
- Language per element is detected by character ratio (`arabic` / `english` / `mixed`).

### LLM correction

Only words below the confidence threshold are sent, as **padded, upscaled crops** — one per
word. Sending the whole page failed completely: the model could not locate a small word in
a large image. Crops changed that outright.

Reliability machinery, all of it added in response to a measured failure:

| Mechanism | Reason |
|---|---|
| Model fallback chain | Free-tier models rate-limit, retire, and return malformed JSON |
| Hard wall-clock ceiling per attempt | Calls return HTTP 200 then never send a body; `httpx` has no total timeout |
| One budget shared by both LLM stages | Each stage previously had its own, doubling worst-case wait |
| Permanent-failure model skipping | A 404 "no endpoints support image input" cannot recover within a request |
| Perceptual-hash dedup (dHash) | A drawing repeats the same symbol many times; label it once |
| Index-keyed responses | Two identical labels on a page must correct independently |
| `llm_status` in the sidecar | Degradation used to be silent — callers could not tell correction never ran |

**Status is always reported.** `success` · `not_attempted / no_flagged_words` ·
`failed / rate_limited` · `failed / timed_out`. A failure returns the document anyway.

### Word↔shape association *(shipped, disabled)*

Each word carries the id of the smallest shape whose bbox contains it, so a flagged word's
confident neighbours inside the same box can be used as correction context.

**Default off (`SHAPE_CONTEXT_ENABLED=false`).** Measured on the test document: only 1 of 4
flagged words had a confident sibling in its box, and enabling it made the configured 12B
model return *no corrections at all* on 2 of 3 runs. This matches published work finding
that added context helps 27B/70B models and makes smaller ones drop content. Kept for a
larger model.

### API hardening

| Guard | File | Protects against |
|---|---|---|
| `asyncio.Semaphore(1)` + threadpool offload | `app/main.py` | One upload freezing the service; PaddleOCR predictors are not thread-safe |
| WebP magic-byte rejection | `app/main.py` | CVE-2023-4863 — fixed upstream in opencv 4.8.1.78, so now defence-in-depth rather than the mitigation |
| 200 MP decoded-pixel cap | `app/main.py` | Decompression bombs — the 25 MB cap bounds the *encoded* file only |
| Generic 500 + server-side logging | `app/main.py` | Tracebacks leaking to callers |
| DB failure tolerated at startup | `app/main.py` | Postgres is only needed by `/feedback`, never by OCR |
| Log level reasserted after each run | `app/main.py` | PaddleOCR's import silently sets the root logger to WARNING, killing all logging after the first request |

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness + whether models are loaded |
| `/process` | POST | Image → `.docx` + sidecar |
| `/feedback` | POST | Store user corrections |
| `/docs` | GET | Interactive Swagger UI |

**`/process` form fields**

| Field | Values | Default |
|---|---|---|
| `image` | JPEG / PNG / TIFF / BMP, ≤ 25 MB | required |
| `language_hint` | `ar+en` · `ar` · `en` | `ar+en` |
| `confidence_threshold` | `0.0`–`1.0` | `0.75` |
| `label_shapes` | `true` / `false` | `true` |
| `response_format` | `json` · `docx` | `json` |

`json` returns the docx base64-encoded with the sidecar. `docx` returns the file itself,
with stats in the `X-Sidecar-Stats` header.

> `language_hint=ar+en` uses the **Arabic** model, which also reads Latin. On an
> all-English page the dedicated `en` model measured better (see below), but it cannot read
> Arabic at all, so the default stays.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Uvicorn | Async, typed, generates the Swagger UI used for manual testing |
| OCR | PaddleOCR 3.7.0 (PP-OCRv6 + PP-OCRv5 Arabic) | One of the few engines with real Arabic support that runs on CPU. v6 has no Arabic, so Arabic stays on the v5 recogniser — per-script routing is still mandatory upstream in 2026. |
| Vision / CV | OpenCV 4.10.0.84 (contrib) | Set by paddlex, and the only opencv installed. contrib is required for `cv2.ximgproc` (shape detection). |
| Document | python-docx + hand-written DrawingML | python-docx cannot do floating anchors; the XML is written directly |
| LLM | OpenRouter via the OpenAI SDK | One API across many models, with a free tier |
| Database | PostgreSQL 15 + asyncpg | Correction feedback only |
| Config | pydantic-settings | `.env` → typed settings |
| Container | Docker Compose, non-root uid 1000 | |

### Known constraints

- **`enable_mkldnn=False` is load-bearing.** Stock paddle 3.3.1 dies on a CPU without
  AVX-512 with a PIR/oneDNN `NotImplementedError`. Neither `ir_optim=False` nor
  `FLAGS_enable_pir_api=0` fixes it, and the `FLAGS_use_mkldnn=0` env var is silently
  ignored by paddle 3.x — the setting must be a constructor kwarg. It is passed at the
  single predictor construction site so it cannot be missed.
- **The WebP guard is no longer load-bearing, but stays.** opencv 4.10.0.84 postdates the
  CVE-2023-4863 fix (4.8.1.78), so byte-level WebP rejection is now defence-in-depth
  rather than the mitigation it was under the old 4.6.0.66 pin.
- **Handwritten Arabic is out of scope for automatic transcription.** Nothing
  CPU-deployable handles it as of 2026; the systems that do are GPU or hosted API. Those
  regions come back low-confidence and highlighted for review rather than guessed at.
- **Correction quality is capped by the free LLM tier.** Rate limits are a daily quota,
  and the free models return high certainty on wrong answers. Low-confidence words are
  highlighted whether or not the model answers, so nothing ships silently unverified.

---

## File structure

```
app/
  main.py                     FastAPI app, endpoints, input validation, hardening
  config.py                   pydantic-settings; all tunables live here
pipeline/
  __init__.py                 process_image() — orchestrates all 7 stages
  preprocessor.py             downscale, deskew, denoise, contrast, quality score
  ocr.py                      one detection pass + per-crop script routing
  shape_detector.py           contour + EdgeDrawing segment analysis, residual ink
  layout_reconstructor.py     reading order, relative coords, word↔shape association
  llm_corrector.py            word correction + shape labelling, budgets, fallback
  word_assembler.py           .docx generation via DrawingML
  sidecar.py                  JSON output
models/
  elements.py                 Text / SimpleShape / PolylineShape / ComplexShape elements
db/
  connection.py               asyncpg pool
  corrections.py              correction_events table
utils/
  bidi.py                     Arabic detection, visual→logical order, reshaping
  image_utils.py              ndarray ↔ PNG bytes
tests/                        157 tests
md_files/                     README, research notes, this report
Dockerfile  docker-compose.yml  requirements.txt  .env.example  try_it.py
```

---

## How to run it

### 1. Configure

```bash
cp .env.example .env
```

Set `OPENROUTER_API_KEY`. **`OPENROUTER_MODEL` must be a vision-capable model** — a
text-only model returns `404 No endpoints found that support image input` on every call.
Free models rotate; check <https://openrouter.ai/models?modality=text%2Bimage&max_price=0>.

Without a key the service still runs — correction and labelling report
`not_attempted / not_configured`, and it is logged loudly once at boot.

### 2. Start

```bash
docker compose up -d --build
docker compose ps            # wait for (healthy)
```

First request after a restart is slower — models load on demand.

### 3. Use it

**Browser (easiest):** open <http://localhost:8000/docs> → `POST /process` → *Try it out* →
choose your image → set `response_format` to `docx` → *Execute* → click the download link.

**Command line:**
```bash
curl -X POST http://localhost:8000/process \
  -F "image=@your-drawing.png;type=image/png" \
  -F "response_format=docx" \
  -o result.docx
```

The content type must match the file. WebP is rejected deliberately.

**Script:** `python try_it.py path/to/image.png` writes `output.docx` + `sidecar.json`.

### 4. Operate

```bash
docker compose logs -f ocr-api    # live logs, incl. the LLM fallback chain
docker compose up -d              # apply .env changes (recreates; restart won't)
docker compose down               # stop
```

> **Never `docker compose down -v`** — that deletes the Postgres volume.

### Tests

Run against the pinned dependencies, not a local venv:

```bash
docker run --rm -v "$PWD:/src" -w /src -e PYTHONPATH=/src ocrtask-ocr-api \
  sh -c "pip install -q pytest && python -m pytest tests/ -q"
```

### Upgrading an existing deployment

If the PaddleOCR model volume predates the non-root container change, it is root-owned and
model downloads fail with `PermissionError`. Once:

```bash
docker compose rm -sf ocr-api
docker volume rm ocrtask_paddle_models
docker compose up -d --build ocr-api
```

---

## Measurements

All from the running container, CPU-only.

**Latency** — 3897×4752 page, 167 elements

| | Before | Now |
|---|---|---|
| Large page, LLM failing | 137s | **~72s** |
| Small page | 14.9s | **~5s** |
| `/health` during a busy `/process` | 2.72s | **6–11ms** |

Worst case is now bounded by the LLM budget rather than by how long a hung socket takes.

**Where non-LLM time goes** (large page, ~12.1s total): OCR 6.3s · layout segmentation
5.1s · preprocess 0.5s · everything else < 0.2s.

**Resolution does not trade off the way you'd expect.** Layout+OCR time is flat from
1000–2000px, because PaddleOCR runs at a largely fixed internal resolution. Downscaling
further only saves time at 800px, where 45% of words are lost. 2000px stays.

**Arabic model vs English model on an all-English page**

| | `ar+en` (Arabic model) | `en` (English model) |
|---|---|---|
| Mean confidence | 0.936 | **0.991** |
| Words below threshold | 4 | **1** |
| Tokens with Arabic-Indic digits | 1 | **0** |
| OCR runtime | 8.1s | **5.4s** |

Recovered readings: `Icreated` → `+created_at`, `window._+tolerance` →
`+tolerance_window_days`, `er١` → `user`.

The same English model on an *Arabic* drawing found only 8 of 21 words — it cannot read
Arabic. Hence the default is unchanged, and script-aware routing is the open follow-up.

---

## Known gaps

| Item | Status |
|---|---|
| Feedback loop | Corrections are stored; `get_few_shot_examples()` exists but is never called |
| `/feedback` validation | Accepts any `document_id` — no foreign-key check |
| `import paddle` flakiness | Known open upstream issue; server unaffected once started |
| Word↔shape context | Shipped, off — needs a model above the 8–13B tier |
| Handwriting recognition | Not implemented; deferred |
| Mixed Arabic+English drawing | No test file exists — the case that would decide script-aware routing |
| Free LLM tier | Rate limits and hangs dominate correction reliability; a paid key is the fix |
