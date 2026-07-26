import base64
import io
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
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
    "image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp"
}
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024


_LANGUAGE_HINTS = {"ar+en", "ar", "en"}


@app.post("/process")
async def process(
    image: UploadFile = File(...),
    confidence_threshold: float = Form(None),
    language_hint: str = Form("ar+en"),
    label_shapes: bool = Form(None),
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

    raw = await image.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )
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
