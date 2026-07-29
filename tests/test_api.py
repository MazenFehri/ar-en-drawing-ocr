import base64
import io
import logging
import threading
import time
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


def test_process_rejects_webp_disguised_as_png():
    """Content-type is attacker-controlled and cv2.imdecode sniffs the bytes, so a
    content_type check alone would let a crafted WebP through to the decoder. Under
    opencv 4.10.0.84 that decoder no longer carries CVE-2023-4863 (fixed in 4.8.1.78),
    but the guard is kept as defence-in-depth — see _is_webp in app/main.py — and this
    test pins the byte-sniffing behaviour that makes it worth keeping."""
    webp = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"VP8 " + b"\x00" * 32
    resp = client.post(
        "/process",
        files={"image": ("evil.png", io.BytesIO(webp), "image/png")},
    )
    assert resp.status_code == 400
    assert "WebP" in resp.json()["detail"]


def test_process_rejects_webp_content_type():
    resp = client.post(
        "/process",
        files={"image": ("x.webp", io.BytesIO(make_jpeg_bytes()), "image/webp")},
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


@patch("app.main.cv2.imdecode")
def test_process_rejects_decompression_bomb(mock_imdecode):
    """A small encoded file can still decode into a huge in-memory bitmap
    (PNG/TIFF dimensions aren't bounded by the encoded byte count). Fake a
    decode that reports an enormous shape without actually allocating one."""
    mock_imdecode.return_value = MagicMock(shape=(20000, 20000, 3))
    resp = client.post(
        "/process",
        files={"image": ("huge.png", io.BytesIO(make_jpeg_bytes()), "image/png")}
    )
    assert resp.status_code == 400
    assert "pixel" in resp.json()["detail"].lower()


@patch("app.main.process_image", side_effect=RuntimeError("boom: /secret/internal/path"))
def test_process_pipeline_failure_returns_generic_500(mock_pipeline):
    """An unhandled exception anywhere in the pipeline must not reach the
    caller as a bare 500 with a traceback/exception message in the body."""
    resp = client.post(
        "/process",
        files={"image": ("drawing.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")}
    )
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Image processing failed"}
    assert "boom" not in resp.text
    assert "secret" not in resp.text


def test_process_restores_log_level_after_pipeline_clobbers_it(caplog):
    """Regression proxy for a real bug: PaddleOCR's first model load imports
    paddle.distributed submodules as a side effect, two of which call a
    paddle-internal get_logger(level, name="root") helper at import time —
    name="root" resolves to the actual root logger, so this silently
    overwrites our configured level (INFO) with WARNING, permanently, and
    every "process done" log after that point simply never fires (confirmed
    live in the running container: logging.getLogger("app.main").disabled
    stays False the whole time, only the *level* changes).

    This suite never imports paddleocr, so it cannot reproduce the real
    trigger. It simulates the observed effect instead: process_image mutates
    the root logger's level as a side effect, standing in for what the real
    import does, and asserts app.main's _restore_log_level fix undoes it in
    time for "process done" to still be logged.
    """
    def clobber_then_return(*args, **kwargs):
        logging.getLogger().setLevel(logging.WARNING)
        return MOCK_PIPELINE_RESULT
    with patch("app.main.process_image", side_effect=clobber_then_return):
        with caplog.at_level(logging.INFO):
            resp = client.post(
                "/process",
                files={"image": ("drawing.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")}
            )
    assert resp.status_code == 200
    messages = [r.message for r in caplog.records if r.name == "app.main"]
    assert any(m.startswith("process start:") for m in messages)
    assert any(m.startswith("process done:") for m in messages), (
        "process done was not logged — the root logger level clobber was not "
        "undone before the log call"
    )


def test_health_not_blocked_by_slow_process():
    """process_image is sync/CPU-bound and must run off the event loop, so
    /health (and any other request) stays responsive while /process is busy.
    Regression test for the event-loop-blocking bug: before the fix, /health
    had to wait for the full duration of the concurrent /process call."""
    SLEEP_S = 0.5

    def slow_process(*a, **kw):
        time.sleep(SLEEP_S)
        return MOCK_PIPELINE_RESULT

    outcome = {}

    def call_process():
        with patch("app.main.process_image", side_effect=slow_process):
            outcome["process_resp"] = client.post(
                "/process",
                files={"image": ("drawing.jpg", io.BytesIO(make_jpeg_bytes()), "image/jpeg")}
            )

    def call_health():
        time.sleep(SLEEP_S / 5)  # let /process start first
        t0 = time.time()
        outcome["health_resp"] = client.get("/health")
        outcome["health_elapsed"] = time.time() - t0

    t1 = threading.Thread(target=call_process)
    t2 = threading.Thread(target=call_health)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert outcome["process_resp"].status_code == 200
    assert outcome["health_resp"].status_code == 200
    # Generous bound: /health should return almost immediately, well short of
    # the remaining ~0.4s of the concurrent /process sleep.
    assert outcome["health_elapsed"] < SLEEP_S * 0.6


@patch("app.main.ensure_table", side_effect=RuntimeError("db unreachable"))
def test_startup_survives_db_outage(mock_ensure_table):
    """The DB is only needed for /feedback, not OCR — a Postgres outage at
    boot must not prevent the app (and therefore /process) from starting."""
    with TestClient(app) as c:
        resp = c.get("/health")
        assert resp.status_code == 200


@patch("app.main.store_corrections", side_effect=OSError("connection refused"))
def test_feedback_returns_503_when_db_unavailable(mock_store):
    resp = client.post(
        "/feedback",
        json={"document_id": "doc-1", "corrections": [{"element_id": "e1", "user_final": "x"}]},
    )
    assert resp.status_code == 503
