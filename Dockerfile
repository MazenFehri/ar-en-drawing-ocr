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

# Pre-download PaddleOCR Arabic model to avoid cold-start delay. Run as
# appuser so the cache lands in /home/appuser/.paddleocr — the same place the
# runtime user will read it from.
USER appuser
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='arabic', show_log=False)" || true

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
