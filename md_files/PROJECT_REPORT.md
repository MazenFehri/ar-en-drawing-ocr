# Arabic + English Drawing OCR — Project Report

An HTTP microservice that turns a scanned architectural or technical drawing into an
editable Word document that keeps the original layout, plus a JSON sidecar describing
every element it found. Called server-to-server from a .NET application.

Input: one image (JPEG / PNG / TIFF / BMP).
Output: a `.docx` where every word and every shape sits where it sat on the page, and a
JSON sidecar with coordinates, confidence, and per-stage status.

---

## The pipeline

Eight stages, run in order by `pipeline/__init__.py::process_image`.

| # | Stage | File | Tech | What it does, and why |
|---|-------|------|------|----------------------|
| 1 | Preprocess | `pipeline/preprocessor.py` | OpenCV | Downscale to 2000px longest side, deskew, denoise, CLAHE contrast. The 2000px cap is a measured sweet spot — see *Measurements*. Deskew refuses to rotate past 15°, because a diagonal section line makes detection report ~45° and rotating would wreck the page. |
| 2 | Layout segmentation | `pipeline/layout.py` | PaddleOCR **PP-Structure** | Classifies page regions as text / figure / table / title. Used to mask text areas before shape detection. |
| 3 | OCR | `pipeline/ocr.py` | PaddleOCR **PP-OCRv4** | Recognises words with bounding boxes and confidence. Runs *before* shape detection so its word boxes can mask text. |
| 4 | Shape detection | `pipeline/shape_detector.py` | OpenCV contours | Finds shapes and drawings, classifies them, and crops anything it can't name. |
| 5 | Layout reconstruction | `pipeline/layout_reconstructor.py` | — | Pixels → relative coordinates, reading order, and word↔shape association. |
| 6 | LLM correction | `pipeline/llm_corrector.py` | OpenRouter (OpenAI SDK) | Re-reads low-confidence words from cropped images. |
| 6b | Shape labelling | `pipeline/llm_corrector.py` | Same | Names unclassifiable shapes ("door swing", "north arrow"). |
| 7 | Word assembly | `pipeline/word_assembler.py` | python-docx + raw DrawingML | Builds the `.docx` with absolutely-positioned elements. |
| 8 | Sidecar | `pipeline/sidecar.py` | — | JSON describing every element, plus stats and per-stage status. |

---

## Features in detail

### Shape and drawing capture

Everything that isn't text is captured and placed. Recognisable geometry becomes a real
Word shape; anything else is embedded as a cropped image at its original position.

- **Otsu thresholding** rather than a fixed cutoff — drawings are bimodal (dark ink, light
  page) and Otsu finds the split itself.
- **`RETR_CCOMP`, not `RETR_EXTERNAL`.** External-only retrieval discarded everything
  inside an outer outline — interior walls, doors, furniture. This was the single biggest
  content-loss bug in the project.
- **Hole rejection.** A stroke drawn as two parallel lines produces an inner contour that
  duplicates the outer one. Contours whose bbox nearly matches their parent are dropped.
- **Classification** by vertex count, extent, circularity and solidity →
  `circle · ellipse · triangle · rect · square · line · complex`.
- **Residual ink catch-all.** Whatever is neither text nor a classified contour gets
  dilated, grouped into connected components, and emitted as cropped images — so hand
  drawings and odd symbols don't silently vanish.

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

- `utils/bidi.py` converts OCR's visual order to logical order — PaddleOCR reports Arabic
  in the order glyphs sit on the page, which is reversed from how the string is stored.
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
| WebP magic-byte rejection | `app/main.py` | CVE-2023-4863 in the pinned opencv's libwebp |
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
| OCR | PaddleOCR 2.7.3 (PP-OCRv4) | One of the few engines with real Arabic support that runs on CPU |
| Layout | PP-Structure | Ships with PaddleOCR; no extra dependency |
| Vision / CV | OpenCV 4.6.0.66 | Pinned — paddleocr 2.7.3 requires `<=4.6.0.66` |
| Document | python-docx + hand-written DrawingML | python-docx cannot do floating anchors; the XML is written directly |
| LLM | OpenRouter via the OpenAI SDK | One API across many models, with a free tier |
| Database | PostgreSQL 15 + asyncpg | Correction feedback only |
| Config | pydantic-settings | `.env` → typed settings |
| Container | Docker Compose, non-root uid 1000 | |

### Known constraints

- **The opencv pin is load-bearing.** paddleocr 2.7.3 requires `opencv-python<=4.6.0.66`,
  which bundles a libwebp with CVE-2023-4863. WebP uploads are refused at the byte level as
  the mitigation. PaddleOCR 3.x would free this — at the cost of an API rewrite
  (`.ocr()`→`.predict()`, changed return shape, `PPStructure` removed).
- **`import paddle` is flaky** — aborts in short-lived processes on some hosts. A known
  open upstream issue, not fixed by migrating. The long-running server is unaffected.

---

## File structure

```
app/
  main.py                     FastAPI app, endpoints, input validation, hardening
  config.py                   pydantic-settings; all tunables live here
pipeline/
  __init__.py                 process_image() — orchestrates all 8 stages
  preprocessor.py             downscale, deskew, denoise, contrast, quality score
  layout.py                   PP-Structure region segmentation
  ocr.py                      PaddleOCR wrapper, per-language model cache
  shape_detector.py           contour analysis, classification, residual ink
  layout_reconstructor.py     reading order, relative coords, word↔shape association
  llm_corrector.py            word correction + shape labelling, budgets, fallback
  word_assembler.py           .docx generation via DrawingML
  sidecar.py                  JSON output
  _paddle_patch.py            disables Paddle IR optimisation (AVX-512 SIGILL workaround)
models/
  elements.py                 TextElement / SimpleShapeElement / ComplexShapeElement
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
