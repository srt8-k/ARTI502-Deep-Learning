"""Minimal local prototype feedback storage. No images are persisted."""

import json
import os
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get(
    "NUTRIVISION_FEEDBACK_DB",
    PROJECT_ROOT / ".local" / "feedback.sqlite3",
))
router = APIRouter()


def connect():
    return sqlite3.connect(DB_PATH, timeout=30)


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(connect()) as db, db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                corrected_food TEXT,
                confirmed_ingredients TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                corrected_at TEXT
            )
            """
        )


def register_prediction(result: dict):
    prediction_id = str(uuid.uuid4())
    with closing(connect()) as db, db:
        db.execute(
            "INSERT INTO predictions (prediction_id, result_json) VALUES (?, ?)",
            (prediction_id, json.dumps(result, ensure_ascii=False)),
        )
    return {**result, "prediction_id": prediction_id}


class FoodCorrection(BaseModel):
    prediction_id: str = Field(min_length=1, max_length=128)
    corrected_food: str = Field(min_length=1, max_length=150)
    confirmed_ingredients: list[str] = Field(default_factory=list, max_length=30)


@router.post("/feedback")
def save_feedback(feedback: FoodCorrection):
    with closing(connect()) as db, db:
        row = db.execute(
            "SELECT prediction_id FROM predictions WHERE prediction_id = ?",
            (feedback.prediction_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Prediction not found")
        db.execute(
            """
            UPDATE predictions
            SET corrected_food = ?, confirmed_ingredients = ?,
                corrected_at = CURRENT_TIMESTAMP
            WHERE prediction_id = ?
            """,
            (
                feedback.corrected_food.strip(),
                json.dumps(feedback.confirmed_ingredients, ensure_ascii=False),
                feedback.prediction_id,
            ),
        )
    return {
        "saved": True,
        "prediction_id": feedback.prediction_id,
        "corrected_food": feedback.corrected_food.strip(),
    }
