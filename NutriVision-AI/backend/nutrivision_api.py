"""FastAPI entry point for NutriVision AI.

V2 Food/No-Food detection is OFF by default; enable explicitly for testing.
"""

import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .ai_service import NutriVisionService
from .feedback_store import init_db, register_prediction, router
from .food_presence import FoodPresenceDetector


MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
service = None
food_detector = None
inference_lock = threading.Lock()


def _mode():
    mode = os.environ.get("NUTRIVISION_FOOD_PRESENCE_MODE", "off").strip().lower()
    if mode not in {"off", "v2"}:
        raise ValueError("NUTRIVISION_FOOD_PRESENCE_MODE must be 'off' or 'v2'")
    return mode


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service, food_detector
    init_db()
    service = NutriVisionService()
    if _mode() == "v2":
        base = Path(os.environ.get(
            "NUTRIVISION_MODEL_DIR",
            Path(__file__).resolve().parents[1] / "models",
        ))
        cnn_path = Path(os.environ.get(
            "NUTRIVISION_CONVNEXT_CHECKPOINT",
            str(base / "convnext_stage3_best.pth"),
        ))
        head_path = Path(os.environ.get(
            "NUTRIVISION_FOOD_PRESENCE_CHECKPOINT",
            str(base / "food_presence_head_v2.pth"),
        ))
        # An explicit V2 request must NOT fall back silently to the old model.
        food_detector = FoodPresenceDetector(cnn_path, head_path, service.device)
    try:
        yield
    finally:
        food_detector = None
        service = None


app = FastAPI(
    title="NutriVision AI",
    description="Food-101 classification, nutrition/portion estimation and optional CNN food detection",
    version="0.2.0-preview",
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
        "food_presence_mode": "v2" if food_detector is not None else "off",
    }


def _no_food_response(detection: dict, device: str):
    """Return a full, backward-compatible prediction shape with NO calories."""
    return {
        "suggested_food": None,
        "food_candidates": [],
        "food101_top3": [],
        "cnn_food101_top3": [],
        "open_vocab_suggestions": [],
        "possible_ingredients": [],
        "estimated_calories": None,
        "estimated_portion_mass_g": None,
        "calorie_unit": "kcal",
        "requires_confirmation": False,
        "user_can_correct_food": False,
        "message": "No food detected. No calories or portion mass were estimated.",
        "device": device,
        "inference_seconds": detection["inference_seconds"],
        "food_present": False,
        "food_presence_mode": "v2",
        "food_presence_score_percent": detection["food_presence_score_percent"],
        "food_presence_threshold": detection["food_presence_threshold"],
    }


@app.post("/predict")
def predict(image: UploadFile = File(...), analyze_ingredients: bool = Form(True)):
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

    # Serialize inference to limit CPU/GPU memory spikes in the prototype.
    with inference_lock:
        detection = food_detector.predict(uploaded) if food_detector is not None else None
        if detection is not None and not detection["food_present"]:
            return register_prediction(_no_food_response(detection, service.device.type))

        # Original food-choice and nutrition path remains unchanged for food.
        result = service.predict(uploaded, include_ingredients=analyze_ingredients)
        if detection is not None:
            result.update({
                "food_present": True,
                "food_presence_mode": "v2",
                "food_presence_score_percent": detection["food_presence_score_percent"],
                "food_presence_threshold": detection["food_presence_threshold"],
                "cnn_food101_top3": detection["cnn_food101_top3"],
                "inference_seconds": round(
                    result["inference_seconds"] + detection["inference_seconds"], 3
                ),
            })
        else:
            result.update({
                "food_present": None,
                "food_presence_mode": "off",
                "food_presence_score_percent": None,
                "food_presence_threshold": None,
                "cnn_food101_top3": [],
            })
        return register_prediction(result)
