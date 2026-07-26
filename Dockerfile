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

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download PaddleOCR Arabic model to avoid cold-start delay
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='arabic', show_log=False)" || true

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
