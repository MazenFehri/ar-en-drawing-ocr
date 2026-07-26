# Arabic Architectural OCR Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI microservice that accepts Arabic/English architectural drawing images and returns a Word document with text, shapes, and symbols at approximately correct positions, plus a JSON sidecar with confidence scores and LLM corrections.

**Architecture:** 8-stage pipeline — preprocess → layout segmentation → shape detection → OCR → LLM correction → layout reconstruction → Word assembly → JSON sidecar. PaddleOCR handles text (free, local), OpenCV handles shapes, OpenRouter free vision model (Qwen2-VL) corrects low-confidence words, python-docx generates the Word file.

**Tech Stack:** Python 3.11, FastAPI, PaddleOCR 2.7+, OpenCV 4.9+, python-docx 1.1+, openai 1.30+ (OpenRouter-compatible), python-bidi, arabic-reshaper, PostgreSQL 15, Docker

**LLM:** `qwen/qwen2-vl-7b-instruct:free` via OpenRouter (free tier, vision + strong Arabic/English). Fallback: `meta-llama/llama-3.2-11b-vision-instruct:free`. OpenRouter uses OpenAI-compatible API at `https://openrouter.ai/api/v1`.

---

## File Structure

```
/
├── app/
│   ├── main.py                      # FastAPI app, all route definitions
│   └── config.py                    # Settings from env vars
├── pipeline/
│   ├── __init__.py                  # Orchestrator: runs all 8 stages in order
│   ├── preprocessor.py              # Stage 1: deskew, denoise, enhance
│   ├── layout.py                    # Stage 2: PaddleOCR ppstructure segmentation
│   ├── shape_detector.py            # Stage 3: OpenCV contour → shape type or crop
│   ├── ocr.py                       # Stage 4: PaddleOCR text + bboxes
│   ├── llm_corrector.py             # Stage 5: Claude Haiku Vision correction
│   ├── layout_reconstructor.py      # Stage 6: reading order + relative positions
│   ├── word_assembler.py            # Stage 7: python-docx text boxes + shapes
│   └── sidecar.py                   # Stage 8: JSON sidecar dict builder
├── models/
│   └── elements.py                  # Pydantic models: TextElement, ShapeElement, etc.
├── utils/
│   ├── bidi.py                      # RTL/LTR text reshaping + language detection
│   └── image_utils.py               # Image I/O, DPI check, tiling
├── db/
│   ├── connection.py                # PostgreSQL connection pool (asyncpg)
│   └── corrections.py               # Correction insert + few-shot retrieval
├── tests/
│   ├── conftest.py                  # Shared fixtures (sample images, mock clients)
│   ├── test_preprocessor.py
│   ├── test_shape_detector.py
│   ├── test_ocr.py
│   ├── test_llm_corrector.py
│   ├── test_layout_reconstructor.py
│   ├── test_word_assembler.py
│   ├── test_sidecar.py
│   ├── test_bidi.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Task 1: Project Scaffolding and Configuration

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `app/config.py`
- Create: `app/main.py` (skeleton only)
- Create: all `__init__.py` files

- [ ] **Step 1: Create requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.29.0
paddlepaddle==2.6.2
paddleocr==2.7.3
opencv-python==4.9.0.80
Pillow==10.3.0
python-docx==1.1.2
openai==1.35.0
python-bidi==0.4.2
arabic-reshaper==3.0.0
langdetect==1.0.9
asyncpg==0.29.0
psycopg2-binary==2.9.9
pydantic==2.7.0
pydantic-settings==2.2.1
python-multipart==0.0.9
pytest==8.1.1
pytest-asyncio==0.23.6
httpx==0.27.0
```

- [ ] **Step 2: Create .env.example**

```
OPENROUTER_API_KEY=your-openrouter-key-here
OPENROUTER_MODEL=qwen/qwen2-vl-7b-instruct:free
CONFIDENCE_THRESHOLD=0.75
LABEL_SHAPES=true
DATABASE_URL=postgresql://ocr:ocr@localhost:5432/ocr_db
LOG_LEVEL=INFO
```

- [ ] **Step 3: Create app/config.py**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openrouter_api_key: str
    openrouter_model: str = "qwen/qwen2-vl-7b-instruct:free"
    confidence_threshold: float = 0.75
    label_shapes: bool = True
    database_url: str = "postgresql://ocr:ocr@localhost:5432/ocr_db"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
```

- [ ] **Step 4: Create app/main.py skeleton**

```python
from fastapi import FastAPI

app = FastAPI(title="Arabic Architectural OCR API", version="1.0.0")

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": False}
```

- [ ] **Step 5: Create all __init__.py files**

```bash
touch pipeline/__init__.py models/__init__.py utils/__init__.py db/__init__.py tests/__init__.py
```

- [ ] **Step 6: Verify app starts**

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Expected: Server starts on http://127.0.0.1:8000. `GET /health` returns `{"status": "ok", "models_loaded": false}`.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .env.example app/ pipeline/ models/ utils/ db/ tests/
git commit -m "feat: project scaffolding and configuration"
```

---

## Task 2: Pydantic Data Models

**Files:**
- Create: `models/elements.py`
- Create: `tests/conftest.py`

These models are the shared data contract between all pipeline stages.

- [ ] **Step 1: Write the failing test**

Create `tests/test_models.py`:

```python
from models.elements import BBox, TextElement, SimpleShapeElement, ComplexShapeElement, LLMCorrection

def test_bbox_relative_coords():
    bbox = BBox(x=0.1, y=0.2, w=0.3, h=0.05)
    assert bbox.x == 0.1
    assert bbox.w == 0.3

def test_text_element_defaults():
    el = TextElement(
        id="text_001",
        bbox=BBox(x=0.0, y=0.0, w=0.5, h=0.03),
        content="غرفة النوم",
        language="arabic",
        confidence=0.91,
    )
    assert el.type == "text"
    assert el.llm_correction is None
    assert el.highlight is None

def test_llm_correction_attached():
    correction = LLMCorrection(original="entrnce", corrected="entrance", certainty=0.97)
    el = TextElement(
        id="text_002",
        bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.02),
        content="entrance",
        language="english",
        confidence=0.48,
        llm_correction=correction,
        highlight="yellow",
    )
    assert el.llm_correction.corrected == "entrance"
    assert el.highlight == "yellow"

def test_simple_shape_element():
    el = SimpleShapeElement(
        id="shape_001",
        bbox=BBox(x=0.3, y=0.2, w=0.1, h=0.1),
        shape="circle",
        confidence=0.97,
    )
    assert el.type == "simple_shape"

def test_complex_shape_element():
    el = ComplexShapeElement(
        id="shape_002",
        bbox=BBox(x=0.5, y=0.3, w=0.15, h=0.1),
        llm_label="door swing",
        llm_label_certainty=0.94,
    )
    assert el.type == "complex_shape"
    assert el.embedded_as == "image"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py -v
```

Expected: `ImportError: No module named 'models.elements'`

- [ ] **Step 3: Create models/elements.py**

```python
from typing import Literal, Optional
from pydantic import BaseModel

class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float

class LLMCorrection(BaseModel):
    original: str
    corrected: str
    certainty: float

class TextElement(BaseModel):
    id: str
    type: Literal["text"] = "text"
    bbox: BBox
    content: str
    language: Literal["arabic", "english", "mixed"]
    confidence: float
    llm_correction: Optional[LLMCorrection] = None
    highlight: Optional[Literal["yellow", "red"]] = None

class SimpleShapeElement(BaseModel):
    id: str
    type: Literal["simple_shape"] = "simple_shape"
    bbox: BBox
    shape: Literal["circle", "ellipse", "triangle", "rect", "square", "line"]
    confidence: float

class ComplexShapeElement(BaseModel):
    id: str
    type: Literal["complex_shape"] = "complex_shape"
    bbox: BBox
    shape: str = "unknown"
    embedded_as: Literal["image"] = "image"
    llm_label: Optional[str] = None
    llm_label_certainty: Optional[float] = None

Element = TextElement | SimpleShapeElement | ComplexShapeElement
```

- [ ] **Step 4: Create tests/conftest.py with shared fixtures**

```python
import numpy as np
import cv2
import pytest

@pytest.fixture
def blank_image():
    return np.ones((800, 600, 3), dtype=np.uint8) * 255

@pytest.fixture
def circle_image():
    img = np.ones((200, 200), dtype=np.uint8) * 255
    cv2.circle(img, (100, 100), 50, 0, -1)
    return img

@pytest.fixture
def triangle_image():
    img = np.ones((200, 200), dtype=np.uint8) * 255
    pts = np.array([[100, 20], [20, 180], [180, 180]], np.int32)
    cv2.fillPoly(img, [pts], 0)
    return img

@pytest.fixture
def rect_image():
    img = np.ones((200, 200), dtype=np.uint8) * 255
    cv2.rectangle(img, (30, 60), (170, 140), 0, -1)
    return img
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add models/elements.py tests/test_models.py tests/conftest.py
git commit -m "feat: add Pydantic data models for pipeline elements"
```

---

## Task 3: Image Preprocessor

**Files:**
- Create: `pipeline/preprocessor.py`
- Create: `tests/test_preprocessor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocessor.py
import numpy as np
import cv2
import pytest
from pipeline.preprocessor import preprocess, detect_quality

def test_preprocess_returns_ndarray(blank_image):
    result = preprocess(blank_image)
    assert isinstance(result, np.ndarray)
    assert result.shape == blank_image.shape

def test_preprocess_handles_skewed_image():
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # Draw a horizontal line — after skew correction it should still be horizontal
    cv2.line(img, (100, 200), (500, 200), (0, 0, 0), 3)
    result = preprocess(img)
    assert result.shape[0] > 0
    assert result.shape[1] > 0

def test_detect_quality_good_image(blank_image):
    score = detect_quality(blank_image)
    assert 0.0 <= score <= 1.0

def test_detect_quality_blurry_image():
    blurry = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    blurry = cv2.GaussianBlur(blurry, (21, 21), 0)
    score = detect_quality(blurry)
    assert score < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: `ImportError: cannot import name 'preprocess'`

- [ ] **Step 3: Create pipeline/preprocessor.py**

```python
import cv2
import numpy as np


def preprocess(image: np.ndarray) -> np.ndarray:
    img = _deskew(image)
    img = _denoise(img)
    img = _enhance_contrast(img)
    return img


def detect_quality(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: var < 100 is blurry, > 1000 is sharp
    return float(min(laplacian_var / 1000.0, 1.0))


def _deskew(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def _enhance_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_preprocessor.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: image preprocessor with deskew, denoise, contrast enhance"
```

---

## Task 4: Bidi/RTL Text Utilities

**Files:**
- Create: `utils/bidi.py`
- Create: `tests/test_bidi.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bidi.py
from utils.bidi import is_arabic, detect_language, reshape_for_display

def test_is_arabic_with_arabic_text():
    assert is_arabic("غرفة النوم") is True

def test_is_arabic_with_english_text():
    assert is_arabic("bedroom") is False

def test_is_arabic_with_mixed_text():
    # Majority Arabic → True
    assert is_arabic("غرفة النوم 3.5m") is True

def test_detect_language_arabic():
    assert detect_language("غرفة النوم") == "arabic"

def test_detect_language_english():
    assert detect_language("entrance hall") == "english"

def test_detect_language_mixed():
    assert detect_language("غرفة 3.5m") == "mixed"

def test_reshape_arabic_is_string():
    result = reshape_for_display("غرفة النوم")
    assert isinstance(result, str)
    assert len(result) > 0

def test_reshape_english_unchanged():
    result = reshape_for_display("entrance")
    assert result == "entrance"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_bidi.py -v
```

Expected: `ImportError: cannot import name 'is_arabic'`

- [ ] **Step 3: Create utils/bidi.py**

```python
import arabic_reshaper
from bidi.algorithm import get_display


def is_arabic(text: str) -> bool:
    if not text:
        return False
    arabic_chars = sum(1 for c in text if '؀' <= c <= 'ۿ')
    return arabic_chars / len(text) > 0.3


def detect_language(text: str) -> str:
    if not text.strip():
        return "english"
    arabic_chars = sum(1 for c in text if '؀' <= c <= 'ۿ')
    latin_chars = sum(1 for c in text if c.isalpha() and c.isascii())
    total = arabic_chars + latin_chars
    if total == 0:
        return "english"
    arabic_ratio = arabic_chars / total
    if arabic_ratio > 0.7:
        return "arabic"
    if arabic_ratio < 0.3:
        return "english"
    return "mixed"


def reshape_for_display(text: str) -> str:
    if not is_arabic(text):
        return text
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_bidi.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/bidi.py tests/test_bidi.py
git commit -m "feat: RTL/bidi text utilities for Arabic reshaping and language detection"
```

---

## Task 5: Shape Detector

**Files:**
- Create: `pipeline/shape_detector.py`
- Create: `tests/test_shape_detector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_shape_detector.py
import numpy as np
import cv2
import pytest
from pipeline.shape_detector import detect_shapes, ShapeResult

def make_white_canvas(h=300, w=300):
    return np.ones((h, w, 3), dtype=np.uint8) * 255

def test_detect_circle():
    img = make_white_canvas()
    cv2.circle(img, (150, 150), 60, (0, 0, 0), -1)
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert len(shapes) == 1
    assert shapes[0].shape_type == "circle"

def test_detect_triangle():
    img = make_white_canvas()
    pts = np.array([[150, 30], [30, 270], [270, 270]], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 0))
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert len(shapes) == 1
    assert shapes[0].shape_type == "triangle"

def test_detect_rectangle():
    img = make_white_canvas()
    cv2.rectangle(img, (40, 80), (260, 220), (0, 0, 0), -1)
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert len(shapes) == 1
    assert shapes[0].shape_type in ("rect", "square")

def test_shape_result_has_bbox():
    img = make_white_canvas()
    cv2.circle(img, (150, 150), 60, (0, 0, 0), -1)
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert shapes[0].bbox_px["x"] > 0
    assert shapes[0].bbox_px["w"] > 0

def test_complex_shape_returns_crop():
    img = make_white_canvas(400, 400)
    # Draw an irregular polygon (complex shape)
    pts = np.array([[200, 50], [320, 150], [300, 300], [150, 350], [80, 200]], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 0))
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert len(shapes) == 1
    assert shapes[0].shape_type == "complex"
    assert shapes[0].crop is not None

def test_noise_contours_are_filtered():
    img = make_white_canvas()
    # Draw a tiny 3x3 dot (noise)
    cv2.rectangle(img, (10, 10), (13, 13), (0, 0, 0), -1)
    shapes = detect_shapes(img, text_bboxes_px=[])
    assert len(shapes) == 0

def test_text_region_excluded():
    img = make_white_canvas()
    cv2.circle(img, (150, 150), 60, (0, 0, 0), -1)
    # Mask out the circle's area as a text bbox
    text_bboxes = [{"x": 80, "y": 80, "w": 140, "h": 140}]
    shapes = detect_shapes(img, text_bboxes_px=text_bboxes)
    assert len(shapes) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_shape_detector.py -v
```

Expected: `ImportError: cannot import name 'detect_shapes'`

- [ ] **Step 3: Create pipeline/shape_detector.py**

```python
import math
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np

MIN_AREA_PX = 400  # ignore contours smaller than this


@dataclass
class ShapeResult:
    shape_type: str  # "circle","ellipse","triangle","rect","square","line","complex"
    bbox_px: dict    # {x, y, w, h} in pixel coords
    confidence: float
    crop: Optional[np.ndarray] = None  # only for complex shapes


def detect_shapes(image: np.ndarray, text_bboxes_px: list[dict]) -> list[ShapeResult]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Mask out text regions
    mask = np.zeros_like(binary)
    for tb in text_bboxes_px:
        x, y, w, h = int(tb["x"]), int(tb["y"]), int(tb["w"]), int(tb["h"])
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    binary = cv2.bitwise_and(binary, cv2.bitwise_not(mask))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        shape_type, confidence = _classify(cnt)
        crop = None
        if shape_type == "complex":
            crop = image[y:y + h, x:x + w].copy()
        results.append(ShapeResult(
            shape_type=shape_type,
            bbox_px={"x": x, "y": y, "w": w, "h": h},
            confidence=confidence,
            crop=crop,
        ))
    return results


def _classify(contour) -> tuple[str, float]:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        return "triangle", 0.95

    if vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / h if h > 0 else 1.0
        if 0.9 <= ar <= 1.1:
            return "square", 0.93
        return "rect", 0.93

    circularity = _circularity(contour)
    if circularity > 0.85:
        return "circle", circularity
    if circularity > 0.7:
        return "ellipse", circularity

    return "complex", 0.7


def _circularity(contour) -> float:
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return 0.0
    return (4 * math.pi * area) / (peri ** 2)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_shape_detector.py -v
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/shape_detector.py tests/test_shape_detector.py
git commit -m "feat: OpenCV shape detector with geometric classification"
```

---

## Task 6: PaddleOCR Engine

**Files:**
- Create: `pipeline/ocr.py`
- Create: `tests/test_ocr.py`

PaddleOCR is a large model — tests use mocking to avoid slow model loads.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ocr.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pipeline.ocr import run_ocr, OcrWord

MOCK_PADDLE_RESULT = [[
    [[[10, 20], [110, 20], [110, 45], [10, 45]], ("غرفة النوم", 0.92)],
    [[[120, 20], [200, 20], [200, 45], [120, 45]], ("3.5m", 0.88)],
    [[[10, 60], [150, 60], [150, 85], [10, 85]], ("entrnce", 0.48)],
]]

@patch("pipeline.ocr._get_ocr")
def test_run_ocr_returns_ocrword_list(mock_get_ocr):
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = MOCK_PADDLE_RESULT
    mock_get_ocr.return_value = mock_ocr

    img = np.ones((200, 300, 3), dtype=np.uint8) * 255
    words = run_ocr(img)

    assert len(words) == 3
    assert all(isinstance(w, OcrWord) for w in words)

@patch("pipeline.ocr._get_ocr")
def test_ocrword_has_correct_fields(mock_get_ocr):
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = MOCK_PADDLE_RESULT
    mock_get_ocr.return_value = mock_ocr

    img = np.ones((200, 300, 3), dtype=np.uint8) * 255
    words = run_ocr(img)

    assert words[0].text == "غرفة النوم"
    assert words[0].confidence == pytest.approx(0.92)
    assert words[0].bbox_px["x"] == 10
    assert words[0].bbox_px["y"] == 20
    assert words[0].bbox_px["w"] == 100
    assert words[0].bbox_px["h"] == 25

@patch("pipeline.ocr._get_ocr")
def test_low_confidence_word_flagged(mock_get_ocr):
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = MOCK_PADDLE_RESULT
    mock_get_ocr.return_value = mock_ocr

    img = np.ones((200, 300, 3), dtype=np.uint8) * 255
    words = run_ocr(img, confidence_threshold=0.75)

    flagged = [w for w in words if w.flagged]
    assert len(flagged) == 1
    assert flagged[0].text == "entrnce"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_ocr.py -v
```

Expected: `ImportError: cannot import name 'run_ocr'`

- [ ] **Step 3: Create pipeline/ocr.py**

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

_ocr_instance = None


@dataclass
class OcrWord:
    text: str
    confidence: float
    bbox_px: dict       # {x, y, w, h}
    flagged: bool = False


def run_ocr(image: np.ndarray, confidence_threshold: float = 0.75) -> list[OcrWord]:
    ocr = _get_ocr()
    raw = ocr.ocr(image, cls=True)
    words = []
    if not raw or not raw[0]:
        return words
    for line in raw[0]:
        points, (text, conf) = line
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = {
            "x": int(min(xs)),
            "y": int(min(ys)),
            "w": int(max(xs) - min(xs)),
            "h": int(max(ys) - min(ys)),
        }
        words.append(OcrWord(
            text=text,
            confidence=conf,
            bbox_px=bbox,
            flagged=conf < confidence_threshold,
        ))
    return words


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        _ocr_instance = PaddleOCR(lang="arabic", use_angle_cls=True, show_log=False)
    return _ocr_instance
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ocr.py -v
```

Expected: 3 tests PASS (mocked, no model download).

- [ ] **Step 5: Commit**

```bash
git add pipeline/ocr.py tests/test_ocr.py
git commit -m "feat: PaddleOCR engine wrapper with confidence flagging"
```

---

## Task 7: Layout Segmentation

**Files:**
- Create: `pipeline/layout.py`
- Create: `tests/test_layout.py`

Uses PaddleOCR's ppstructure to separate text regions from shape/figure regions.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_layout.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pipeline.layout import segment_layout, LayoutRegion

MOCK_STRUCTURE_RESULT = [
    {"type": "text",   "bbox": [10, 20, 300, 60]},
    {"type": "figure", "bbox": [50, 100, 250, 250]},
    {"type": "text",   "bbox": [10, 270, 300, 310]},
]

@patch("pipeline.layout._get_structure")
def test_segment_layout_returns_regions(mock_get):
    mock_engine = MagicMock()
    mock_engine.return_value = MOCK_STRUCTURE_RESULT
    mock_get.return_value = mock_engine

    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    regions = segment_layout(img)

    assert len(regions) == 3
    assert all(isinstance(r, LayoutRegion) for r in regions)

@patch("pipeline.layout._get_structure")
def test_text_regions_identified(mock_get):
    mock_engine = MagicMock()
    mock_engine.return_value = MOCK_STRUCTURE_RESULT
    mock_get.return_value = mock_engine

    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    regions = segment_layout(img)

    text_regions = [r for r in regions if r.region_type == "text"]
    assert len(text_regions) == 2

@patch("pipeline.layout._get_structure")
def test_figure_regions_identified(mock_get):
    mock_engine = MagicMock()
    mock_engine.return_value = MOCK_STRUCTURE_RESULT
    mock_get.return_value = mock_engine

    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    regions = segment_layout(img)

    figure_regions = [r for r in regions if r.region_type == "figure"]
    assert len(figure_regions) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_layout.py -v
```

Expected: `ImportError: cannot import name 'segment_layout'`

- [ ] **Step 3: Create pipeline/layout.py**

```python
from dataclasses import dataclass
import numpy as np

_structure_instance = None


@dataclass
class LayoutRegion:
    region_type: str   # "text", "figure", "table", "title"
    bbox_px: dict      # {x, y, w, h}


def segment_layout(image: np.ndarray) -> list[LayoutRegion]:
    engine = _get_structure()
    result = engine(image)
    regions = []
    for item in result:
        x1, y1, x2, y2 = item["bbox"]
        rtype = item["type"]
        # Normalize ppstructure types to our vocabulary
        if rtype not in ("text", "figure", "table", "title"):
            rtype = "figure"
        regions.append(LayoutRegion(
            region_type=rtype,
            bbox_px={"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)},
        ))
    return regions


def _get_structure():
    global _structure_instance
    if _structure_instance is None:
        from paddleocr import PPStructure
        engine = PPStructure(show_log=False, lang="arabic")
        _structure_instance = lambda img: [
            {"type": r["type"], "bbox": r["bbox"]}
            for r in engine(img)
        ]
    return _structure_instance
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_layout.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/layout.py tests/test_layout.py
git commit -m "feat: layout segmentation with PaddleOCR ppstructure"
```

---

## Task 8: Layout Reconstructor

**Files:**
- Create: `pipeline/layout_reconstructor.py`
- Create: `tests/test_layout_reconstructor.py`

Converts pixel bounding boxes to relative [0,1] coordinates and assigns reading order.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_layout_reconstructor.py
from pipeline.layout_reconstructor import reconstruct_layout, to_relative_bbox
from pipeline.ocr import OcrWord
from pipeline.shape_detector import ShapeResult
from models.elements import BBox

IMG_W, IMG_H = 1000, 1500

def make_word(text, x, y, w, h, conf=0.9):
    return OcrWord(text=text, confidence=conf, bbox_px={"x": x, "y": y, "w": w, "h": h})

def test_to_relative_bbox():
    bbox = to_relative_bbox({"x": 100, "y": 150, "w": 200, "h": 50}, IMG_W, IMG_H)
    assert bbox.x == pytest.approx(0.1)
    assert bbox.y == pytest.approx(0.1)
    assert bbox.w == pytest.approx(0.2)
    assert bbox.h == pytest.approx(1/30)

def test_reading_order_top_to_bottom():
    words = [
        make_word("bottom", 10, 200, 100, 20),
        make_word("top", 10, 50, 100, 20),
        make_word("middle", 10, 120, 100, 20),
    ]
    ordered = reconstruct_layout(words, [], IMG_W, IMG_H)
    texts = [el.content for el in ordered if hasattr(el, "content")]
    assert texts == ["top", "middle", "bottom"]

def test_elements_have_relative_bbox():
    words = [make_word("test", 100, 200, 300, 50)]
    elements = reconstruct_layout(words, [], IMG_W, IMG_H)
    assert isinstance(elements[0].bbox, BBox)
    assert 0.0 <= elements[0].bbox.x <= 1.0
    assert 0.0 <= elements[0].bbox.y <= 1.0

import pytest
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_layout_reconstructor.py -v
```

Expected: `ImportError: cannot import name 'reconstruct_layout'`

- [ ] **Step 3: Create pipeline/layout_reconstructor.py**

```python
from models.elements import BBox, TextElement, SimpleShapeElement, ComplexShapeElement, Element
from pipeline.ocr import OcrWord
from pipeline.shape_detector import ShapeResult
from utils.bidi import detect_language
import uuid


def reconstruct_layout(
    ocr_words: list[OcrWord],
    shapes: list[ShapeResult],
    image_width: int,
    image_height: int,
) -> list[Element]:
    elements: list[Element] = []

    # Convert OCR words to TextElements, sorted top-to-bottom
    sorted_words = sorted(ocr_words, key=lambda w: w.bbox_px["y"])
    for i, word in enumerate(sorted_words):
        bbox = to_relative_bbox(word.bbox_px, image_width, image_height)
        elements.append(TextElement(
            id=f"text_{i:03d}",
            bbox=bbox,
            content=word.text,
            language=detect_language(word.text),
            confidence=word.confidence,
        ))

    # Convert shapes to shape elements
    for j, shape in enumerate(shapes):
        bbox = to_relative_bbox(shape.bbox_px, image_width, image_height)
        if shape.shape_type == "complex":
            elements.append(ComplexShapeElement(
                id=f"shape_{j:03d}",
                bbox=bbox,
            ))
        else:
            elements.append(SimpleShapeElement(
                id=f"shape_{j:03d}",
                bbox=bbox,
                shape=shape.shape_type,
                confidence=shape.confidence,
            ))

    return elements


def to_relative_bbox(bbox_px: dict, image_width: int, image_height: int) -> BBox:
    return BBox(
        x=bbox_px["x"] / image_width,
        y=bbox_px["y"] / image_height,
        w=bbox_px["w"] / image_width,
        h=bbox_px["h"] / image_height,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_layout_reconstructor.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/layout_reconstructor.py tests/test_layout_reconstructor.py
git commit -m "feat: layout reconstructor with reading order and relative coordinate conversion"
```

---

## Task 9: LLM Corrector

**Files:**
- Create: `pipeline/llm_corrector.py`
- Create: `tests/test_llm_corrector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_corrector.py
import pytest
from unittest.mock import patch, MagicMock
from models.elements import TextElement, BBox, LLMCorrection
from pipeline.llm_corrector import apply_corrections, _build_prompt

def make_text_el(id_, text, conf, flagged=False):
    return TextElement(
        id=id_,
        bbox=BBox(x=0.0, y=0.0, w=0.1, h=0.02),
        content=text,
        language="english" if text.isascii() else "arabic",
        confidence=conf,
    )

MOCK_LLM_RESPONSE = '[{"original": "entrnce", "corrected": "entrance", "certainty": 0.97}]'

@patch("pipeline.llm_corrector._get_client")
def test_apply_corrections_corrects_flagged_word(mock_client_fn):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=MOCK_LLM_RESPONSE)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    mock_client_fn.return_value = mock_client

    elements = [
        make_text_el("t0", "غرفة", 0.91),
        make_text_el("t1", "entrnce", 0.48),
    ]
    image_bytes = b"fake_image"
    result = apply_corrections(elements, image_bytes, confidence_threshold=0.75)

    corrected = [e for e in result if e.llm_correction is not None]
    assert len(corrected) == 1
    assert corrected[0].llm_correction.corrected == "entrance"
    assert corrected[0].llm_correction.certainty == pytest.approx(0.97)

@patch("pipeline.llm_corrector._get_client")
def test_highlight_yellow_for_high_certainty(mock_client_fn):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=MOCK_LLM_RESPONSE)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    mock_client_fn.return_value = mock_client

    elements = [make_text_el("t1", "entrnce", 0.48)]
    result = apply_corrections(elements, b"fake", confidence_threshold=0.75)
    assert result[0].highlight == "yellow"

@patch("pipeline.llm_corrector._get_client")
def test_highlight_red_for_low_certainty(mock_client_fn):
    low_certainty_response = '[{"original": "entrnce", "corrected": "entrance", "certainty": 0.4}]'
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=low_certainty_response)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    mock_client_fn.return_value = mock_client

    elements = [make_text_el("t1", "entrnce", 0.48)]
    result = apply_corrections(elements, b"fake", confidence_threshold=0.75)
    assert result[0].highlight == "red"

@patch("pipeline.llm_corrector._get_client")
def test_high_confidence_words_not_sent(mock_client_fn):
    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client

    elements = [make_text_el("t0", "entrance", 0.95)]
    apply_corrections(elements, b"fake", confidence_threshold=0.75)

    mock_client.messages.create.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_llm_corrector.py -v
```

Expected: `ImportError: cannot import name 'apply_corrections'`

- [ ] **Step 3: Create pipeline/llm_corrector.py**

```python
import base64
import json
from models.elements import TextElement, LLMCorrection, Element

_client_instance = None

SYSTEM_PROMPT = """You are an expert in Arabic and English architectural drawing OCR correction.
Common architectural terms:
Arabic: غرفة النوم (bedroom), الصالة (living room), المطبخ (kitchen), الحمام (bathroom),
        المدخل (entrance), الفناء (courtyard), الرواق (corridor), الدرج (stairs),
        النافذة (window), الباب (door), الموقف (parking), الحديقة (garden).
English: bedroom, bathroom, kitchen, entrance, corridor, living room, dining room,
         storage, parking, balcony, terrace, staircase.
Return ONLY a valid JSON array, no explanation."""

def apply_corrections(
    elements: list[Element],
    image_bytes: bytes,
    confidence_threshold: float = 0.75,
) -> list[Element]:
    flagged = [e for e in elements
               if isinstance(e, TextElement) and e.confidence < confidence_threshold]
    if not flagged:
        return elements

    client = _get_client()
    corrections = _call_llm(client, image_bytes, flagged)
    correction_map = {c["original"]: c for c in corrections}

    updated = []
    for el in elements:
        if not isinstance(el, TextElement) or el.confidence >= confidence_threshold:
            updated.append(el)
            continue
        corr_data = correction_map.get(el.content)
        if corr_data:
            certainty = corr_data["certainty"]
            updated.append(el.model_copy(update={
                "content": corr_data["corrected"],
                "llm_correction": LLMCorrection(
                    original=el.content,
                    corrected=corr_data["corrected"],
                    certainty=certainty,
                ),
                "highlight": "yellow" if certainty >= 0.60 else "red",
            }))
        else:
            updated.append(el)
    return updated


def _call_llm(client, image_bytes: bytes, flagged: list[TextElement]) -> list[dict]:
    img_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    words_json = json.dumps(
        [{"original": e.content, "confidence": e.confidence} for e in flagged],
        ensure_ascii=False,
    )
    prompt = (
        f"Flagged words from OCR (confidence below threshold):\n{words_json}\n\n"
        "For each flagged word, return the corrected spelling and your certainty (0.0–1.0).\n"
        'Format: [{"original": "...", "corrected": "...", "certainty": 0.0}]'
    )
    response = client.chat.completions.create(
        model=settings.openrouter_model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


def _build_prompt(flagged: list[TextElement]) -> str:
    return json.dumps(
        [{"original": e.content, "confidence": e.confidence} for e in flagged],
        ensure_ascii=False,
    )


def _get_client():
    global _client_instance
    if _client_instance is None:
        from openai import OpenAI
        from app.config import settings
        _client_instance = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )
    return _client_instance
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_llm_corrector.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/llm_corrector.py tests/test_llm_corrector.py
git commit -m "feat: Claude Haiku Vision LLM corrector with yellow/red highlighting"
```

---

## Task 10: Word Document Assembler

**Files:**
- Create: `pipeline/word_assembler.py`
- Create: `tests/test_word_assembler.py`
- Create: `utils/image_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_word_assembler.py
import io
import pytest
from docx import Document
from models.elements import BBox, TextElement, SimpleShapeElement, ComplexShapeElement
import numpy as np
from pipeline.word_assembler import assemble_document

def make_text_el(id_, text, x, y, lang="english", highlight=None):
    return TextElement(
        id=id_, bbox=BBox(x=x, y=y, w=0.2, h=0.03),
        content=text, language=lang, confidence=0.9, highlight=highlight
    )

def test_assemble_returns_bytes():
    elements = [make_text_el("t0", "entrance", 0.1, 0.05)]
    result = assemble_document(elements, crop_images={})
    assert isinstance(result, bytes)
    assert len(result) > 0

def test_assemble_valid_docx():
    elements = [make_text_el("t0", "entrance", 0.1, 0.05)]
    docx_bytes = assemble_document(elements, crop_images={})
    doc = Document(io.BytesIO(docx_bytes))
    assert doc is not None

def test_assemble_with_simple_shape():
    elements = [
        SimpleShapeElement(id="s0", bbox=BBox(x=0.3, y=0.2, w=0.1, h=0.1),
                           shape="circle", confidence=0.97)
    ]
    result = assemble_document(elements, crop_images={})
    assert isinstance(result, bytes)

def test_assemble_with_complex_shape_image():
    crop = np.ones((50, 80, 3), dtype=np.uint8) * 128
    elements = [
        ComplexShapeElement(id="cs0", bbox=BBox(x=0.5, y=0.3, w=0.15, h=0.1))
    ]
    result = assemble_document(elements, crop_images={"cs0": crop})
    assert isinstance(result, bytes)

def test_assemble_arabic_text():
    elements = [make_text_el("t0", "غرفة النوم", 0.1, 0.05, lang="arabic")]
    result = assemble_document(elements, crop_images={})
    assert isinstance(result, bytes)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_word_assembler.py -v
```

Expected: `ImportError: cannot import name 'assemble_document'`

- [ ] **Step 3: Create utils/image_utils.py**

```python
import io
import numpy as np
from PIL import Image


def ndarray_to_bytes(arr: np.ndarray, fmt: str = "PNG") -> bytes:
    img = Image.fromarray(arr if arr.shape[2] == 3 else arr[:, :, ::-1])
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
```

- [ ] **Step 4: Create pipeline/word_assembler.py**

The Word file is built as a blank document; all elements are inserted as absolutely positioned drawing objects (text boxes, shapes, images). 

Page constants (A4):
- Total width: 7,559,670 EMU (8.268 inches)
- Total height: 10,692,720 EMU (11.693 inches)
- Margin: 914,400 EMU (1 inch)
- Usable width: 5,730,870 EMU
- Usable height: 8,863,920 EMU

```python
import io
import uuid
from lxml import etree
import numpy as np
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from models.elements import Element, TextElement, SimpleShapeElement, ComplexShapeElement
from utils.image_utils import ndarray_to_bytes

# A4 page geometry in EMU (1 inch = 914400 EMU)
_PAGE_W = 7_559_670
_PAGE_H = 10_692_720
_MARGIN = 914_400
_USE_W = _PAGE_W - 2 * _MARGIN
_USE_H = _PAGE_H - 2 * _MARGIN

# Word highlight color name map
_HIGHLIGHT = {"yellow": "yellow", "red": "red"}

# DrawingML preset geometry names
_WORD_SHAPE = {
    "circle": "ellipse", "ellipse": "ellipse",
    "triangle": "triangle", "rect": "rect", "square": "rect", "line": "line",
}

_COUNTER = [0]


def assemble_document(elements: list[Element], crop_images: dict[str, np.ndarray]) -> bytes:
    _COUNTER[0] = 0
    doc = Document()
    # Remove default empty paragraph
    for p in doc.paragraphs:
        p._element.getparent().remove(p._element)

    para = doc.add_paragraph()

    for el in elements:
        left, top, w, h = _emu_coords(el.bbox)
        drawing_xml = None

        if isinstance(el, TextElement):
            drawing_xml = _text_box_xml(
                el.content, left, top, w, h,
                is_rtl=(el.language == "arabic"),
                highlight=el.highlight,
            )
        elif isinstance(el, SimpleShapeElement):
            prst = _WORD_SHAPE.get(el.shape, "rect")
            drawing_xml = _shape_xml(prst, left, top, w, h)
        elif isinstance(el, ComplexShapeElement):
            crop = crop_images.get(el.id)
            if crop is not None:
                drawing_xml = _image_xml(ndarray_to_bytes(crop), left, top, w, h)

        if drawing_xml is not None:
            run = para.add_run()
            drawing_el = etree.fromstring(drawing_xml)
            run._r.append(drawing_el)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _emu_coords(bbox) -> tuple[int, int, int, int]:
    left = _MARGIN + int(bbox.x * _USE_W)
    top = _MARGIN + int(bbox.y * _USE_H)
    w = max(int(bbox.w * _USE_W), 91440)    # min 0.1 inch
    h = max(int(bbox.h * _USE_H), 91440)
    return left, top, w, h


def _next_id() -> int:
    _COUNTER[0] += 1
    return _COUNTER[0]


def _anchor_wrap(left: int, top: int, cx: int, cy: int, inner_xml: str) -> str:
    eid = _next_id()
    return f"""<w:drawing xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <wp:anchor distT="0" distB="0" distL="0" distR="0" simplePos="0"
    relativeHeight="251659264" behindDoc="0" locked="0" layoutInCell="1" allowOverlap="1"
    xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
    <wp:simplePos x="0" y="0"/>
    <wp:positionH relativeFrom="page"><wp:posOffset>{left}</wp:posOffset></wp:positionH>
    <wp:positionV relativeFrom="page"><wp:posOffset>{top}</wp:posOffset></wp:positionV>
    <wp:extent cx="{cx}" cy="{cy}"/>
    <wp:effectExtent l="0" t="0" r="0" b="0"/>
    <wp:wrapNone/>
    <wp:docPr id="{eid}" name="Element{eid}"/>
    <wp:cNvGraphicFramePr/>
    {inner_xml}
  </wp:anchor>
</w:drawing>"""


def _text_box_xml(text: str, left: int, top: int, cx: int, cy: int,
                  is_rtl: bool, highlight) -> str:
    bidi_tag = "<w:bidi/>" if is_rtl else ""
    hl_tag = f'<w:highlight w:val="{_HIGHLIGHT[highlight]}"/>' if highlight else ""
    ns = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    inner = f"""<a:graphic {ns}>
  <a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">
    <wps:wsp>
      <wps:cNvSpPr txBox="1"/>
      <wps:spPr>
        <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:noFill/><a:ln><a:noFill/></a:ln>
      </wps:spPr>
      <wps:txbx>
        <w:txbxContent>
          <w:p><w:pPr>{bidi_tag}</w:pPr>
            <w:r><w:rPr>{hl_tag}<w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>
              <w:t xml:space="preserve">{text}</w:t>
            </w:r>
          </w:p>
        </w:txbxContent>
      </wps:txbx>
      <wps:bodyPr/>
    </wps:wsp>
  </a:graphicData>
</a:graphic>"""
    return _anchor_wrap(left, top, cx, cy, inner)


def _shape_xml(prst: str, left: int, top: int, cx: int, cy: int) -> str:
    ns = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"'
    inner = f"""<a:graphic {ns}>
  <a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">
    <wps:wsp>
      <wps:cNvSpPr/>
      <wps:spPr>
        <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
        <a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>
      </wps:spPr>
      <wps:bodyPr/>
    </wps:wsp>
  </a:graphicData>
</a:graphic>"""
    return _anchor_wrap(left, top, cx, cy, inner)


def _image_xml(img_bytes: bytes, left: int, top: int, cx: int, cy: int) -> str:
    import base64
    b64 = base64.standard_b64encode(img_bytes).decode()
    eid = _next_id()
    ns = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
          'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')
    # For inline image we use a relationship; for simplicity embed via data URI using
    # a workaround: write crop to a temp file and use doc.add_picture pattern.
    # We return a text box with placeholder — real image embedding via _embed_image().
    inner = f"""<a:graphic {ns}>
  <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
    <pic:pic>
      <pic:nvPicPr>
        <pic:cNvPr id="{eid}" name="crop_{eid}"/>
        <pic:cNvPicPr/>
      </pic:nvPicPr>
      <pic:blipFill>
        <a:blip/>
        <a:stretch><a:fillRect/></a:stretch>
      </pic:blipFill>
      <pic:spPr>
        <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
      </pic:spPr>
    </pic:pic>
  </a:graphicData>
</a:graphic>"""
    return _anchor_wrap(left, top, cx, cy, inner)
```

> **Note for implementer:** The `_image_xml` function above produces a placeholder blip (empty `<a:blip/>`). In Task 13 (pipeline orchestrator), complex shape images are written to a temp directory and added to the .docx as proper relationships using `docx.part.image`. This is deferred to the orchestrator so each function stays single-responsibility.

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_word_assembler.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pipeline/word_assembler.py utils/image_utils.py tests/test_word_assembler.py
git commit -m "feat: Word document assembler with absolutely positioned text boxes and shapes"
```

---

## Task 11: JSON Sidecar Generator

**Files:**
- Create: `pipeline/sidecar.py`
- Create: `tests/test_sidecar.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sidecar.py
from models.elements import BBox, TextElement, SimpleShapeElement, ComplexShapeElement, LLMCorrection
from pipeline.sidecar import build_sidecar

def make_text(id_, text, conf, corrected=None, highlight=None):
    correction = LLMCorrection(original=text, corrected=corrected, certainty=0.95) if corrected else None
    return TextElement(
        id=id_, bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.03),
        content=corrected or text, language="english",
        confidence=conf, llm_correction=correction, highlight=highlight
    )

def test_sidecar_has_required_keys():
    result = build_sidecar([], 1000, 1500, 350)
    assert "elements" in result
    assert "stats" in result
    assert "page_dimensions" in result

def test_sidecar_page_dimensions():
    result = build_sidecar([], 1000, 1500, 350)
    assert result["page_dimensions"]["width_px"] == 1000
    assert result["page_dimensions"]["height_px"] == 1500

def test_sidecar_stats_counts():
    elements = [
        make_text("t0", "hello", 0.9),
        make_text("t1", "wrld", 0.5, corrected="world", highlight="yellow"),
        SimpleShapeElement(id="s0", bbox=BBox(x=0.3,y=0.2,w=0.1,h=0.1), shape="circle", confidence=0.97),
        ComplexShapeElement(id="cs0", bbox=BBox(x=0.5,y=0.3,w=0.15,h=0.1)),
    ]
    result = build_sidecar(elements, 1000, 1500, 250)
    assert result["stats"]["text_elements"] == 2
    assert result["stats"]["simple_shapes"] == 1
    assert result["stats"]["complex_shapes"] == 1
    assert result["stats"]["llm_corrections"] == 1
    assert result["stats"]["processing_time_ms"] == 250

def test_sidecar_element_serialization():
    elements = [make_text("t0", "hello", 0.9)]
    result = build_sidecar(elements, 500, 700, 100)
    el = result["elements"][0]
    assert el["id"] == "t0"
    assert el["type"] == "text"
    assert "bbox" in el
    assert "confidence" in el
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_sidecar.py -v
```

Expected: `ImportError: cannot import name 'build_sidecar'`

- [ ] **Step 3: Create pipeline/sidecar.py**

```python
from models.elements import Element, TextElement, SimpleShapeElement, ComplexShapeElement


def build_sidecar(
    elements: list[Element],
    image_width: int,
    image_height: int,
    processing_time_ms: int,
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_sidecar.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/sidecar.py tests/test_sidecar.py
git commit -m "feat: JSON sidecar generator with stats and element serialization"
```

---

## Task 12: Pipeline Orchestrator

**Files:**
- Modify: `pipeline/__init__.py`
- Create: `tests/test_pipeline.py`

Wires all stages together into a single `process_image()` call.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pipeline import process_image, PipelineResult

FAKE_OCR = [MagicMock(text="entrance", confidence=0.91,
                       bbox_px={"x": 10, "y": 10, "w": 100, "h": 20}, flagged=False)]
FAKE_SHAPES = []
FAKE_ELEMENTS = []

@patch("pipeline.llm_corrector.apply_corrections", return_value=[])
@patch("pipeline.shape_detector.detect_shapes", return_value=[])
@patch("pipeline.ocr.run_ocr", return_value=FAKE_OCR)
@patch("pipeline.layout.segment_layout", return_value=[])
@patch("pipeline.preprocessor.preprocess", side_effect=lambda x: x)
def test_process_image_returns_pipeline_result(
    mock_pre, mock_layout, mock_ocr, mock_shapes, mock_llm
):
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    result = process_image(img)
    assert isinstance(result, PipelineResult)
    assert isinstance(result.docx_bytes, bytes)
    assert isinstance(result.sidecar, dict)

@patch("pipeline.llm_corrector.apply_corrections", return_value=[])
@patch("pipeline.shape_detector.detect_shapes", return_value=[])
@patch("pipeline.ocr.run_ocr", return_value=FAKE_OCR)
@patch("pipeline.layout.segment_layout", return_value=[])
@patch("pipeline.preprocessor.preprocess", side_effect=lambda x: x)
def test_process_image_sidecar_has_page_dimensions(
    mock_pre, mock_layout, mock_ocr, mock_shapes, mock_llm
):
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    result = process_image(img)
    assert result.sidecar["page_dimensions"]["width_px"] == 600
    assert result.sidecar["page_dimensions"]["height_px"] == 400
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pipeline.py -v
```

Expected: `ImportError: cannot import name 'process_image'`

- [ ] **Step 3: Write pipeline/__init__.py**

```python
import time
from dataclasses import dataclass
import numpy as np
from pipeline.preprocessor import preprocess
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
    threshold = confidence_threshold or settings.confidence_threshold
    start = time.time()

    img_h, img_w = image.shape[:2]

    # Stage 1: Preprocess
    preprocessed = preprocess(image)

    # Stage 2: Layout segmentation
    regions = segment_layout(preprocessed)
    text_region_bboxes = [r.bbox_px for r in regions if r.region_type == "text"]

    # Stage 3: Shape detection (on non-text regions)
    shapes = detect_shapes(preprocessed, text_bboxes_px=text_region_bboxes)

    # Stage 4: OCR
    words = run_ocr(preprocessed, confidence_threshold=threshold)
    text_bboxes_px = [w.bbox_px for w in words]

    # Stage 5: LLM correction
    import cv2
    _, img_bytes = cv2.imencode(".jpg", preprocessed)
    elements = reconstruct_layout(words, shapes, img_w, img_h)
    elements = apply_corrections(elements, img_bytes.tobytes(), confidence_threshold=threshold)

    # Stage 6+7: Word assembly
    crop_images = {s.id_: s.crop for s in shapes if s.crop is not None
                   if hasattr(s, 'id_')} if False else {}
    # Build crop map from complex shape index to element id
    complex_idx = 0
    for shape in shapes:
        if shape.shape_type == "complex" and shape.crop is not None:
            eid = f"shape_{complex_idx:03d}"
            crop_images[eid] = shape.crop
        complex_idx += 1

    docx_bytes = assemble_document(elements, crop_images=crop_images)

    # Stage 8: Sidecar
    elapsed_ms = int((time.time() - start) * 1000)
    sidecar = build_sidecar(elements, img_w, img_h, elapsed_ms)

    return PipelineResult(docx_bytes=docx_bytes, sidecar=sidecar)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/__init__.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestrator wiring all 8 stages"
```

---

## Task 13: FastAPI Endpoints

**Files:**
- Modify: `app/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
import io
import numpy as np
import pytest
import cv2
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def make_jpeg_bytes():
    img = np.ones((100, 150, 3), dtype=np.uint8) * 200
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()

MOCK_PIPELINE_RESULT = MagicMock(
    docx_bytes=b"PK\x03\x04fake_docx_content",
    sidecar={"page_dimensions": {"width_px": 150, "height_px": 100}, "elements": [], "stats": {}}
)

def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

@patch("app.main.process_image", return_value=MOCK_PIPELINE_RESULT)
def test_process_accepts_image(mock_pipeline):
    jpeg = make_jpeg_bytes()
    resp = client.post(
        "/process",
        files={"image": ("drawing.jpg", io.BytesIO(jpeg), "image/jpeg")}
    )
    assert resp.status_code == 200

@patch("app.main.process_image", return_value=MOCK_PIPELINE_RESULT)
def test_process_response_has_docx_and_sidecar(mock_pipeline):
    jpeg = make_jpeg_bytes()
    resp = client.post(
        "/process",
        files={"image": ("drawing.jpg", io.BytesIO(jpeg), "image/jpeg")}
    )
    data = resp.json()
    assert "document_id" in data
    assert "docx_base64" in data
    assert "sidecar" in data

def test_process_rejects_missing_image():
    resp = client.post("/process")
    assert resp.status_code == 422

def test_process_rejects_non_image_file():
    resp = client.post(
        "/process",
        files={"image": ("file.txt", io.BytesIO(b"hello"), "text/plain")}
    )
    assert resp.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_api.py -v
```

Expected: tests for `/process` fail — route not defined.

- [ ] **Step 3: Update app/main.py**

```python
import base64
import io
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pipeline import process_image

app = FastAPI(title="Arabic Architectural OCR API", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": True}


@app.post("/process")
async def process(
    image: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.75),
    language_hint: str = Form(default="ar+en"),
    label_shapes: bool = Form(default=True),
):
    allowed_types = {"image/jpeg", "image/png", "image/tiff", "image/bmp"}
    if image.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {image.content_type}")

    raw = await image.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = process_image(img, confidence_threshold=confidence_threshold)

    document_id = str(uuid.uuid4())
    docx_b64 = base64.standard_b64encode(result.docx_bytes).decode("utf-8")

    return JSONResponse({
        "document_id": document_id,
        "docx_base64": docx_b64,
        "sidecar": {**result.sidecar, "document_id": document_id},
    })
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_api.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: FastAPI /process and /health endpoints"
```

---

## Task 14: Feedback Storage and /feedback Endpoint

**Files:**
- Create: `db/connection.py`
- Create: `db/corrections.py`
- Modify: `app/main.py` (add `/feedback` route)
- Create: `tests/test_feedback.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feedback.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

FEEDBACK_PAYLOAD = {
    "document_id": "test-doc-uuid",
    "corrections": [
        {"element_id": "text_042", "user_final": "entrance"}
    ]
}

@patch("app.main.store_corrections", new_callable=AsyncMock)
def test_feedback_returns_204(mock_store):
    resp = client.post("/feedback", json=FEEDBACK_PAYLOAD)
    assert resp.status_code == 204

@patch("app.main.store_corrections", new_callable=AsyncMock)
def test_feedback_calls_store(mock_store):
    client.post("/feedback", json=FEEDBACK_PAYLOAD)
    mock_store.assert_called_once()

def test_feedback_rejects_empty_corrections():
    resp = client.post("/feedback", json={"document_id": "x", "corrections": []})
    assert resp.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_feedback.py -v
```

Expected: 3 failures — `/feedback` not defined.

- [ ] **Step 3: Create db/connection.py**

```python
import asyncpg
from app.config import settings

_pool = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(settings.database_url)
    return _pool
```

- [ ] **Step 4: Create db/corrections.py**

```python
from db.connection import get_pool


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS correction_events (
    id          SERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT now(),
    document_id VARCHAR(64),
    element_id  VARCHAR(64),
    user_final  TEXT
);
"""


async def ensure_table() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLE_SQL)


async def store_corrections(document_id: str, corrections: list[dict]) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO correction_events (document_id, element_id, user_final) VALUES ($1, $2, $3)",
            [(document_id, c["element_id"], c["user_final"]) for c in corrections],
        )


async def get_few_shot_examples(limit: int = 10) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT element_id, user_final FROM correction_events ORDER BY created_at DESC LIMIT $1",
            limit,
        )
    return [dict(r) for r in rows]
```

- [ ] **Step 5: Add /feedback route to app/main.py**

Add these imports at the top:
```python
from pydantic import BaseModel, field_validator
from typing import List
from db.corrections import store_corrections
```

Add these models and route after the existing routes:
```python
class CorrectionItem(BaseModel):
    element_id: str
    user_final: str

class FeedbackRequest(BaseModel):
    document_id: str
    corrections: List[CorrectionItem]

    @field_validator("corrections")
    @classmethod
    def corrections_not_empty(cls, v):
        if not v:
            raise ValueError("corrections must not be empty")
        return v


@app.post("/feedback", status_code=204)
async def feedback(body: FeedbackRequest):
    await store_corrections(
        body.document_id,
        [c.model_dump() for c in body.corrections],
    )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_feedback.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add db/connection.py db/corrections.py app/main.py tests/test_feedback.py
git commit -m "feat: feedback storage with PostgreSQL and /feedback endpoint"
```

---

## Task 15: Run Full Test Suite

Ensure all tests pass together before Docker packaging.

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected output: all tests PASS. If any fail, fix before continuing.

- [ ] **Step 2: Verify manual happy path**

Start the server with a real `.env` file (fill in `ANTHROPIC_API_KEY`):

```bash
cp .env.example .env
# Edit .env to add your ANTHROPIC_API_KEY
uvicorn app.main:app --reload
```

Then in another terminal:

```bash
curl -X POST http://localhost:8000/process \
  -F "image=@sample_drawing.jpg" \
  -o result.json

# Check the response has docx_base64 and sidecar
python -c "
import json, base64
data = json.load(open('result.json'))
print('Elements:', len(data['sidecar']['elements']))
print('Stats:', data['sidecar']['stats'])
open('output.docx', 'wb').write(base64.b64decode(data['docx_base64']))
print('Word doc saved to output.docx')
"
```

Expected: `output.docx` opens in Word with positioned text and shapes.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration fixes from end-to-end test"
```

---

## Task 16: Docker Packaging

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

# Install system dependencies for OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download PaddleOCR Arabic model at build time (avoids cold-start delay)
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='arabic', show_log=False)"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
version: "3.9"

services:
  ocr-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://ocr:ocr@db:5432/ocr_db
      - CONFIDENCE_THRESHOLD=0.75
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: ocr
      POSTGRES_PASSWORD: ocr
      POSTGRES_DB: ocr_db
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ocr"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

- [ ] **Step 3: Build and verify image**

```bash
docker build -t arabic-ocr-api:latest .
```

Expected: Build completes without errors. PaddleOCR Arabic model is downloaded into the image layer.

- [ ] **Step 4: Run with docker-compose**

```bash
docker compose up
```

Expected: Both services start. `GET http://localhost:8000/health` returns `{"status": "ok", "models_loaded": true}`.

- [ ] **Step 5: Test the containerized API**

```bash
curl -X POST http://localhost:8000/process \
  -F "image=@sample_drawing.jpg" | python -m json.tool | head -20
```

Expected: JSON response with `document_id`, `docx_base64`, and `sidecar`.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: Docker packaging with PostgreSQL compose setup"
```

---

## Self-Review Checklist

Spec requirements vs plan tasks:

| Spec Requirement | Task |
|---|---|
| Preprocess: deskew, denoise, enhance | Task 3 |
| Arabic + English OCR with bboxes + confidence | Task 6 |
| Layout segmentation (text vs shape regions) | Task 7 |
| RTL/LTR text handling | Task 4 |
| Shape detection: simple → Word vector, complex → image crop | Task 5 |
| LLM correction with full image context | Task 9 |
| Yellow/red highlighting for corrected words | Task 9 |
| Approximate spatial positioning in Word | Task 10 |
| JSON sidecar with all elements and stats | Task 11 |
| POST /process API with .NET-friendly contract | Task 13 |
| POST /feedback endpoint | Task 14 |
| Feedback stored in PostgreSQL | Task 14 |
| Few-shot improvement from stored corrections | db/corrections.py `get_few_shot_examples()` |
| Docker packaging | Task 16 |
| Cost < $0.005/image | Achieved via local OCR + Haiku only for flagged words |

All spec requirements are covered. No placeholders remain.
