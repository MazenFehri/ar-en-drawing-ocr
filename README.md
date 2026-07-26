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

First build downloads PaddleOCR Arabic model weights (~200 MB). Subsequent starts are fast.

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

> PaddleOCR downloads Arabic model weights (~200 MB) automatically on first use.

### 3. Configure environment

The `.env` file in the project root is already populated. If starting fresh, set these variables:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=google/gemma-4-31b-it:free
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
out** → choose an image → **Execute**. The response (with `docx_base64`) appears
below. No command line needed — recommended on Windows, where PowerShell's `curl`
is an alias for `Invoke-WebRequest` and does not accept `-X`/`-F` flags.

### Process an image (curl)

```bash
curl -X POST http://localhost:8000/process \
  -F "image=@your_drawing.jpg"
```

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
      "processing_time_ms": 1200
    }
  }
}
```

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
image → preprocess → layout segmentation → shape detection
      → OCR (PaddleOCR PP-OCRv4) → LLM correction (vision model via OpenRouter)
      → layout reconstruction → Word assembly → JSON sidecar
```

- **OCR:** PaddleOCR PP-OCRv4, Arabic + English, runs locally
- **Layout:** PaddleOCR PP-Structure region segmentation
- **LLM correction:** OpenRouter free vision model (configurable, optional)
- **Word output:** Absolutely positioned DrawingML objects (text boxes, shapes, embedded images)
- **Arabic text:** `arabic_reshaper` + `python-bidi` + `<w:bidi/>` RTL tags
- **Storage:** PostgreSQL via asyncpg for correction feedback

## Troubleshooting

**`SIGILL` / `Illegal instruction` crash when a model loads.**
PaddlePaddle 2.6.x's inference optimizer emits AVX-512 instructions, which Intel
12th/13th/14th-gen consumer CPUs (Alder Lake / Raptor Lake) do not have. This is
handled in `pipeline/_paddle_patch.py`, which disables Paddle IR optimization so
inference falls back to AVX2 kernels. Keep that import in the model loaders.

**First request is very slow (minutes), later ones are fast (~8–12s).**
On first use PaddleOCR/PP-Structure download model weights (~35 MB) from a slow
CDN. They are cached in the `paddle_models` Docker volume, so this only happens
once — even across `docker compose up --build`. (Running `docker compose down -v`
wipes the volume and forces a re-download.)

**`opencv-python` dependency conflict on build.**
`paddleocr 2.7.3` requires `opencv-python<=4.6.0.66`; the pin in
`requirements.txt` matches this.
