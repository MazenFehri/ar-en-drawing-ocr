import base64
import io
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pipeline import process_image

app = FastAPI(title="Arabic Architectural OCR API", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": False}


_ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp"
}


@app.post("/process")
async def process(image: UploadFile = File(...)):
    if image.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {image.content_type}")

    raw = await image.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = process_image(img)

    document_id = str(uuid.uuid4())
    docx_b64 = base64.standard_b64encode(result.docx_bytes).decode("utf-8")

    return JSONResponse({
        "document_id": document_id,
        "docx_base64": docx_b64,
        "sidecar": {**result.sidecar, "document_id": document_id},
    })
