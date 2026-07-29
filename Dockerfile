FROM python:3.11-slim

# System dependencies for OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# There used to be an `ENV FLAGS_use_mkldnn=0` here, to keep paddle off AVX-512 kernels
# this CPU (Alder/Raptor Lake, AVX-512 fused off) does not have. It was a no-op lie:
# paddle 3.x ignores that flag entirely. Verified on 3.3.1 — with the env var set,
# stock inference still dies with
#   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
#   [pir::ArrayAttribute<pir::DoubleAttribute>]  (onednn_instruction.cc:116)
# Neither `ir_optim=False` nor `FLAGS_enable_pir_api=0` fixes it either. The only thing
# that does is `enable_mkldnn=False` passed as a constructor kwarg, so the setting now
# lives in pipeline/ocr.py on every TextDetection/TextRecognition construction. Do not
# re-add an env var here and assume it is doing anything.

# Non-root runtime user. paddleocr 3.x caches model weights under ~/.paddlex/official_models
# (paddlex's model registry), NOT the ~/.paddleocr of 2.x — so the pre-download step below
# and the docker-compose `paddle_models` volume must both point at this user's home under
# the *new* path, or the weights re-download from the CDN on every container start.
RUN useradd --create-home --uid 1000 --shell /bin/bash appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# paddlex requires opencv-contrib-python==4.10.0.84 and is now the only thing in the
# graph that wants an opencv, so there is exactly one distribution in site-packages/cv2
# and no install-order race to force (see requirements.txt). The assert stays: it is the
# actual contract pipeline/shape_detector.py depends on (cv2.ximgproc.createEdgeDrawing),
# and a silent regression here would quietly downgrade every drawing to the LSD fallback.
RUN python -c "import cv2; assert hasattr(cv2, 'ximgproc'), cv2.__version__; assert hasattr(cv2.ximgproc, 'createEdgeDrawing'), 'no createEdgeDrawing'; print('cv2', cv2.__version__, 'with ximgproc.createEdgeDrawing')"

USER appuser

# Create the model cache dir as appuser BEFORE anything tries to populate it,
# independent of whether the downloads below succeed. This is the part that
# actually prevents the PermissionError class of bug: if this path doesn't
# exist in the image at all, Docker auto-vivifies the paddle_models volume
# mountpoint (docker-compose.yml) as root on first use, and a from-scratch
# runtime model download hits Permission denied instead of self-healing.
RUN mkdir -p /home/appuser/.paddlex/official_models

# Pre-download the three models the one-detection-pass pipeline uses (see
# pipeline/ocr.py): one detector, plus a Latin and an Arabic recogniser. All three are
# needed for the default ar+en hint, and /process?language_hint=en selects the Latin
# recogniser on its own, so none of them can be skipped here or that request
# downloads-on-demand the first time anyone picks it. ~37 MB total.
#
# enable_mkldnn=False on every construction — see the FLAGS_use_mkldnn note above; a
# construction without it is an instant hard crash on this CPU, including here.
#
# Retried a few times and `|| true` either way: constructing a paddle predictor has been
# observed to SIGABRT/SIGSEGV intermittently in a short-lived one-shot process on some
# hosts even though it is solid inside the long-running uvicorn server. A failed bake
# isn't fatal, it just means the first real request downloads instead, which works
# because of the mkdir above.
# ponytail: retry-and-shrug is still the ceiling here, not a real fix for the native-init
# flakiness. If bakes keep failing in prod, the upgrade path is fetching+extracting the
# model archives directly (bypassing predictor construction for the download) instead of
# going through paddlex's own downloader.
RUN for i in 1 2 3; do python -c "from paddleocr import TextDetection; TextDetection(model_name='PP-OCRv6_tiny_det', enable_mkldnn=False)" && break; done || true
RUN for i in 1 2 3; do python -c "from paddleocr import TextRecognition; TextRecognition(model_name='PP-OCRv6_small_rec', enable_mkldnn=False)" && break; done || true
RUN for i in 1 2 3; do python -c "from paddleocr import TextRecognition; TextRecognition(model_name='arabic_PP-OCRv5_mobile_rec', enable_mkldnn=False)" && break; done || true

USER root
COPY . .
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# python:3.11-slim has no curl; use the stdlib instead of installing one just
# for this.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
