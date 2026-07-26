import base64
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


MOCK_SIDECAR = {
    "page_dimensions": {"width_px": 150, "height_px": 100},
    "elements": [],
    "stats": {
        "total_elements": 0, "text_elements": 0, "simple_shapes": 0,
        "complex_shapes": 0, "llm_corrections": 0, "processing_time_ms": 10
    }
}

MOCK_PIPELINE_RESULT = MagicMock(
    docx_bytes=b"PK\x03\x04fake_docx",
    sidecar=MOCK_SIDECAR,
)


def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@patch("app.main.process_image", return_value=MOCK_PIPELINE_RESULT)
def test_process_accepts_jpeg(mock_pipeline):
    jpeg = make_jpeg_bytes()
    resp = client.post(
        "/process",
        files={"image": ("drawing.jpg", io.BytesIO(jpeg), "image/jpeg")}
    )
    assert resp.status_code == 200


@patch("app.main.process_image", return_value=MOCK_PIPELINE_RESULT)
def test_process_response_shape(mock_pipeline):
    jpeg = make_jpeg_bytes()
    resp = client.post(
        "/process",
        files={"image": ("drawing.jpg", io.BytesIO(jpeg), "image/jpeg")}
    )
    data = resp.json()
    assert "document_id" in data
    assert "docx_base64" in data
    assert "sidecar" in data


@patch("app.main.process_image", return_value=MOCK_PIPELINE_RESULT)
def test_process_docx_base64_is_valid(mock_pipeline):
    jpeg = make_jpeg_bytes()
    resp = client.post(
        "/process",
        files={"image": ("drawing.jpg", io.BytesIO(jpeg), "image/jpeg")}
    )
    docx_b64 = resp.json()["docx_base64"]
    decoded = base64.b64decode(docx_b64)
    assert decoded == b"PK\x03\x04fake_docx"


def test_process_rejects_missing_image():
    resp = client.post("/process")
    assert resp.status_code == 422


def test_process_rejects_text_file():
    resp = client.post(
        "/process",
        files={"image": ("file.txt", io.BytesIO(b"hello world"), "text/plain")}
    )
    assert resp.status_code == 400


@patch("app.main.process_image", return_value=MOCK_PIPELINE_RESULT)
def test_process_can_return_raw_docx(mock_pipeline):
    resp = client.post(
        "/process",
        files={"image": ("drawing.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")},
        data={"response_format": "docx"},
    )
    assert resp.status_code == 200
    assert resp.content == b"PK\x03\x04fake_docx"
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.headers["x-document-id"]


def test_process_rejects_unknown_response_format():
    resp = client.post(
        "/process",
        files={"image": ("drawing.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")},
        data={"response_format": "pdf"},
    )
    assert resp.status_code == 400


def test_process_rejects_oversized_upload():
    from app.main import _MAX_UPLOAD_BYTES
    huge = b"\xff" * (_MAX_UPLOAD_BYTES + 1)
    resp = client.post(
        "/process",
        files={"image": ("huge.jpg", io.BytesIO(huge), "image/jpeg")}
    )
    assert resp.status_code == 413
