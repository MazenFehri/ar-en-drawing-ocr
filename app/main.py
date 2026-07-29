import asyncio
import base64
import json
import logging
import time
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, field_validator
from starlette.concurrency import run_in_threadpool
from typing import List
from pipeline import process_image
from db.corrections import store_corrections, ensure_table
from db.connection import close_pool
from app.config import settings
from contextlib import asynccontextmanager

# settings.log_level existed but nothing ever applied it — INFO-level logs
# (including the retry/backoff logging already in pipeline/llm_corrector.py)
# were silently dropped by the logging module's WARNING-level default. This is
# the one place the whole process's logging gets configured.
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _restore_log_level():
    """Undo a logging clobber that PaddleOCR's first model load causes as a side effect.

    Confirmed live (traced logging.Logger.setLevel calls in the running container,
    since a bare `import paddle` reliably SIGABRTs/SIGSEGVs in a throwaway process
    on this box): loading a model via pipeline/ocr.py imports paddle, which transitively
    imports paddle.distributed submodules. Two of those
    (paddle/distributed/utils/launch_utils.py and
    paddle/distributed/fleet/meta_parallel/sharding/group_sharded_stage2.py) call
    a paddle-internal `get_logger(level, name="root")` helper at module import
    time. `name="root"` is the special string logging.getLogger() resolves to the
    actual root logger, and that helper unconditionally calls .setLevel() on
    whatever it gets — so this one-time import silently overwrites our
    basicConfig level (INFO) with WARNING, permanently, the first time any
    /process request loads a model. logger.disabled is never touched — only the
    level is — which is why "process done" (INFO) vanishes forever after the
    first request while "process failed" (logged at ERROR) would not have.
    ponytail: reasserting after every pipeline call rather than patching paddle's
    import is the smallest fix that's still correct — the clobber only happens
    once (Python caches the import), so this only ever has to win the race once,
    and it's cheap enough (one setLevel call) to just always do it. Upgrade path
    if paddle ever also flips .disabled: reset that here too.
    """
    logging.getLogger().setLevel(settings.log_level)


@asynccontextmanager
async def lifespan(app):
    # The DB is only needed for the /feedback correction loop, not for OCR. If
    # Postgres is down or misconfigured, ensure_table() raises and (proven:
    # asyncpg.exceptions.InvalidPasswordError against this repo's own
    # docker-compose credentials) that used to take startup down with it,
    # taking /process down too. Log and continue instead — /feedback will
    # raise its own error per-request (via get_pool()) until the DB recovers.
    try:
        await ensure_table()
    except Exception:
        logger.exception("DB unavailable at startup; /process will still work, /feedback will not until this is fixed")
    if not settings.openrouter_api_key:
        # LLM correction degrades to "not_attempted" per-request already (see
        # pipeline/llm_corrector.py), but that's easy to miss buried in a
        # sidecar. Say it once, loudly, at boot.
        logger.warning("OPENROUTER_API_KEY is not set; LLM correction and shape labelling are disabled")
    yield
    await close_pool()


app = FastAPI(title="Arabic Architectural OCR API", version="1.0.0", lifespan=lifespan)

# process_image is synchronous, CPU-bound (~3s) and can call out to an LLM for
# up to another ~60s (see TOTAL_LLM_BUDGET_SECONDS in llm_corrector.py). Run it
# in FastAPI's threadpool (run_in_threadpool below) so it doesn't block the
# event loop — otherwise nothing else, including /health, gets served while an
# image is processing (proven: see scratchpad proof — /health took 2.72s to
# answer during a 3.02s /process call before this fix).
#
# Predictors are cached per model name in pipeline.ocr._predictors
# and reused across calls; Paddle Inference predictors are documented as not
# thread-safe for concurrent inference on one shared instance. Offloading to
# the threadpool alone would let concurrent /process requests hit the same
# predictor from different threads. This semaphore caps that at 1 concurrent
# pipeline run, trading away parallelism the shared model can't safely give
# us anyway.
# ponytail: bound is 1 concurrent /process at a time; extra requests queue
# rather than run in parallel. Upgrade path if throughput becomes the
# bottleneck: one PaddleOCR instance per worker thread (~100MB extra per
# slot) or predictor.clone(), then raise this to match.
_process_semaphore = asyncio.Semaphore(1)


@app.get("/health")
async def health():
    import pipeline.ocr as _ocr
    return {"status": "ok", "models_loaded": bool(_ocr._predictors)}


_ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/tiff", "image/bmp"
}
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _is_webp(raw: bytes) -> bool:
    """WebP uploads are refused. This is now belt-and-braces, not a live mitigation.

    It was originally load-bearing: paddleocr 2.7.3 capped opencv at <=4.6.0.66, which
    bundles a libwebp carrying CVE-2023-4863 (heap overflow, exploited in the wild), and
    the cap could not move while paddleocr was pinned. Under paddleocr 3.7.0 the only
    opencv in the graph is opencv-contrib-python 4.10.0.84 (verified in the built image,
    not inferred from the dependency graph), and the libwebp fix landed upstream in
    4.8.1.78 — so the vulnerable decoder is gone.

    The guard stays anyway. WebP is not a format architectural drawings arrive in, so
    refusing it costs nothing, and image-decoder CVEs are a recurring genre rather than a
    one-off: keeping the format out of cv2.imdecode entirely is cheaper than tracking the
    next one. Sniffing the bytes rather than trusting content_type is still the whole
    point — cv2.imdecode detects format from content, so a crafted WebP sent as image/png
    would otherwise reach the decoder regardless of what the request claimed.
    ponytail: a denylist of one format, which only helps for the format we thought of.
    Upgrade path if decoder exposure ever matters more than it does here: allowlist by
    sniffed magic bytes (PNG/JPEG/TIFF/BMP) instead of denying WebP specifically.
    """
    return len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"


# The 25 MB cap above bounds the *encoded* upload, not the decoded array —
# PNG/TIFF can expand a small file into a huge in-memory bitmap ("decompression
# bomb"), e.g. a few KB of PNG declaring 65535x65535 decodes to ~12GB as a BGR
# array. Reject decoded images bigger than this many pixels before they reach
# the (much heavier) pipeline. 200 MP clears a real large-format scan (A0 at
# 300 DPI is ~139 MP) with headroom, while still bounding a hostile decode to
# ~600 MB.
# ponytail: the number is a guess at "clearly hostile ceiling", not a measured
# one. Raise it if legitimate large-format scans start getting rejected.
_MAX_DECODED_PIXELS = 200_000_000

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
    if img.shape[0] * img.shape[1] > _MAX_DECODED_PIXELS:
        raise HTTPException(status_code=400, detail="Decoded image exceeds the maximum allowed pixel count")

    document_id = str(uuid.uuid4())
    logger.info(
        "process start: document_id=%s filename=%s content_type=%s size_bytes=%d dims=%dx%d",
        document_id, image.filename, image.content_type, len(raw), img.shape[1], img.shape[0],
    )
    t0 = time.monotonic()
    try:
        # See _process_semaphore above for why this is both threadpool-offloaded
        # (don't block the event loop) and serialized (PaddleOCR instances
        # aren't safe for concurrent use).
        async with _process_semaphore:
            try:
                result = await run_in_threadpool(
                    process_image,
                    img,
                    confidence_threshold=confidence_threshold,
                    language_hint=language_hint,
                    label_shapes=label_shapes,
                )
            finally:
                # Must run before any log call below — see _restore_log_level's
                # docstring for why the first model load needs this every time.
                _restore_log_level()
    except Exception:
        # The pipeline is a lot of moving parts (preprocessing, layout, OCR,
        # shape detection, docx assembly) and previously any exception in any
        # of them propagated straight to the client as a bare 500 with a
        # traceback in the body. Log the real thing server-side, tell the
        # caller only that it failed.
        logger.exception(
            "process failed: document_id=%s after %.1fs", document_id, time.monotonic() - t0,
        )
        raise HTTPException(status_code=500, detail="Image processing failed") from None
    logger.info(
        "process done: document_id=%s elapsed=%.1fs llm_status=%s",
        document_id, time.monotonic() - t0, result.sidecar["stats"].get("llm_status"),
    )

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
    try:
        await store_corrections(
            body.document_id,
            [c.model_dump() for c in body.corrections],
        )
    except Exception:
        # Same DB-outage scenario as the lifespan startup check above, just
        # per-request instead of at boot: /feedback is the only thing that
        # needs Postgres, so this is the one place an outage should surface —
        # as a clear 503, not an unhandled exception (Starlette's default
        # handler would already return a generic 500 without leaking
        # anything, but 503 + a logged reason is more useful to the caller
        # and to whoever is on call).
        logger.exception("feedback failed: document_id=%s", body.document_id)
        raise HTTPException(status_code=503, detail="Could not store corrections") from None
