import base64
import io
import json
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, field_validator
from typing import List
from pipeline import process_image
from db.corrections import store_corrections, ensure_table
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app):
    await ensure_table()
    yield


app = FastAPI(title="Arabic Architectural OCR API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    import pipeline.ocr as _ocr
    return {"status": "ok", "models_loaded": bool(_ocr._ocr_instances)}


_ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/tiff", "image/bmp"
}
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _is_webp(raw: bytes) -> bool:
    """WebP is refused because opencv-python is pinned to 4.6.0.66, which bundles a
    libwebp carrying CVE-2023-4863 (heap overflow, exploited in the wild). The pin
    can't move: paddleocr 2.7.3 requires opencv-python<=4.6.0.66.

    Sniffing the bytes rather than trusting content_type is the whole point —
    cv2.imdecode detects format from content, so a crafted WebP sent as image/png
    would still reach the vulnerable decoder.
    ponytail: drop this once paddleocr 3.x frees the opencv pin.
    """
    return len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"


_LANGUAGE_HINTS = {"ar+en", "ar", "en"}


@app.post("/process")
async def process(
    image: UploadFile = File(...),
    confidence_threshold: float = Form(None),
    language_hint: str = Form("ar+en"),
    label_shapes: bool = Form(None),
    response_format: str = Form("json"),
):
    if image.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {image.content_type}")
    if confidence_threshold is not None and not 0.0 <= confidence_threshold <= 1.0:
        raise HTTPException(status_code=400, detail="confidence_threshold must be between 0.0 and 1.0")
    if language_hint not in _LANGUAGE_HINTS:
        raise HTTPException(
            status_code=400,
            detail=f"language_hint must be one of {sorted(_LANGUAGE_HINTS)}",
        )
    if response_format not in ("json", "docx"):
        raise HTTPException(status_code=400, detail="response_format must be 'json' or 'docx'")

    raw = await image.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )
    if _is_webp(raw):
        raise HTTPException(status_code=400, detail="WebP images are not supported")
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = process_image(
        img,
        confidence_threshold=confidence_threshold,
        language_hint=language_hint,
        label_shapes=label_shapes,
    )

    document_id = str(uuid.uuid4())

    if response_format == "docx":
        # The .docx itself, so a browser (or Swagger's "Download file" link) saves a
        # file you can open instead of a base64 blob you have to decode by hand.
        # The sidecar rides along in headers.
        return Response(
            content=result.docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{document_id}.docx"',
                "X-Document-Id": document_id,
                "X-Sidecar-Stats": json.dumps(result.sidecar["stats"]),
            },
        )

    docx_b64 = base64.standard_b64encode(result.docx_bytes).decode("utf-8")

    return JSONResponse({
        "document_id": document_id,
        "docx_base64": docx_b64,
        "sidecar": {**result.sidecar, "document_id": document_id},
    })


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
