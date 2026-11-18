"""FastAPI entry point for the NutriVision AI prototype."""

import io
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .ai_service import NutriVisionService
from .feedback_store import init_db, register_prediction, router


MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
service = None
inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    init_db()
    service = NutriVisionService()
    yield
    service = None


app = FastAPI(
    title="NutriVision AI",
    description="Food-101 classification, nutrition estimation, and optional open-vocabulary suggestions.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/health")
def health():
    return {
        "status": "ready" if service is not None else "loading",
        "device": service.device.type if service is not None else None,
        "open_vocabulary_enabled": (
            service.recognizer is not None if service is not None else False
        ),
    }


@app.post("/predict")
def predict(
    image: UploadFile = File(...),
    analyze_ingredients: bool = Form(True),
):
    if service is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    content = image.file.read(MAX_FILE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Empty image")
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB")

    try:
        with Image.open(io.BytesIO(content)) as raw:
            width, height = raw.size
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions too large")
            raw.load()
            uploaded = raw.copy()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Serialize inference in this development server to avoid memory spikes.
    with inference_lock:
        result = service.predict(uploaded, include_ingredients=analyze_ingredients)
    return register_prediction(result)
