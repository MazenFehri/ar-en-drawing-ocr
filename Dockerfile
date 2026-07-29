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

# Disable MKL-DNN to avoid AVX-512 kernels on CPUs without AVX-512 (Alder/Raptor Lake).
# IR optimization is disabled in code via pipeline/_paddle_patch.py.
ENV FLAGS_use_mkldnn=0

# Non-root runtime user. PaddleOCR caches its model weights under the *home*
# directory it sees at runtime (~/.paddleocr), so the pre-download step below
# and the docker-compose `paddle_models` volume must both point at this user's
# home, not root's, or the ~200MB model weights re-download from PaddleOCR's
# CDN on every container start.
RUN useradd --create-home --uid 1000 --shell /bin/bash appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Three distributions (opencv-python, opencv-contrib-python, opencv-python-headless via
# pdf2docx) all unpack into the same site-packages/cv2, and none of them can be dropped
# — see the comment block in requirements.txt. Only the contrib build carries
# cv2.ximgproc, which pipeline/shape_detector.py detects line segments with, so reinstall
# it last and let it win the directory outright instead of leaving it to pip's resolution
# order. --no-deps because everything it needs is already resolved above. The assert is
# the point of the step: a silent regression here would quietly downgrade every drawing
# to the LSD fallback path.
RUN pip install --no-cache-dir --force-reinstall --no-deps opencv-contrib-python==4.6.0.66 \
    && python -c "import cv2; assert hasattr(cv2, 'ximgproc'), cv2.__version__; print('cv2', cv2.__version__, 'with ximgproc')"

USER appuser

# Create the model cache dir as appuser BEFORE anything tries to populate it,
# independent of whether the downloads below succeed. This is the part that
# actually prevents the PermissionError class of bug: if this path doesn't
# exist in the image at all, Docker auto-vivifies the paddle_models volume
# mountpoint (docker-compose.yml) as root on first use, and a from-scratch
# runtime model download hits Permission denied instead of self-healing.
RUN mkdir -p /home/appuser/.paddleocr

# Pre-download PaddleOCR models to avoid cold-start delay AND to avoid a
# runtime download entirely: /process?language_hint=en selects the separate
# "en" model (see pipeline/ocr.py _LANG_MODELS), so it must be baked in here
# too or that request downloads-on-demand the first time anyone picks it.
# Retried a few times: constructing PaddleOCR() has been observed to
# SIGABRT/SIGSEGV intermittently in a short-lived one-shot process on some
# hosts (paddle's native init appears to dislike a fresh interpreter that
# exits right after) even though it's solid inside the long-running uvicorn
# server. `|| true` either way — a failed bake isn't fatal, it just means the
# first real request downloads instead, which now works because of the mkdir
# above.
# ponytail: retry-and-shrug is the ceiling here, not a real fix for the
# native-init flakiness. If bakes keep failing in prod, the upgrade path is
# fetching+extracting the model tarballs directly (bypassing `import paddle`
# for the download) instead of going through PaddleOCR()'s own downloader.
RUN for i in 1 2 3; do python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='arabic', show_log=False)" && break; done || true
RUN for i in 1 2 3; do python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en', show_log=False)" && break; done || true

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
