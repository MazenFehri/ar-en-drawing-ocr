from fastapi import FastAPI

app = FastAPI(title="Arabic Architectural OCR API", version="1.0.0")

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": False}
