"""Feedback tests; do not load model weights or download datasets."""

import tempfile
from pathlib import Path

import pytest
from fastapi import HTTPException

from backend import feedback_store as store


@pytest.fixture()
def isolated_db():
    original = store.DB_PATH
    with tempfile.TemporaryDirectory() as tmp:
        store.DB_PATH = Path(tmp) / "feedback.sqlite3"
        store.init_db()
        try:
            yield
        finally:
            store.DB_PATH = original


def test_prediction_and_correction_round_trip(isolated_db):
    result = store.register_prediction(
        {"suggested_food": "chicken_wings", "estimated_calories": 507.2}
    )
    assert "prediction_id" in result
    feedback = store.FoodCorrection(
        prediction_id=result["prediction_id"],
        corrected_food="Chicken Kabsa",
        confirmed_ingredients=["chicken", "rice"],
    )
    saved = store.save_feedback(feedback)
    assert saved["saved"] is True

    with store.connect() as db:
        row = db.execute(
            "SELECT corrected_food, confirmed_ingredients FROM predictions "
            "WHERE prediction_id = ?",
            (result["prediction_id"],),
        ).fetchone()
    assert row == ("Chicken Kabsa", '["chicken", "rice"]')


def test_unknown_prediction_id_returns_404(isolated_db):
    feedback = store.FoodCorrection(
        prediction_id="does-not-exist",
        corrected_food="Kabsa",
    )
    with pytest.raises(HTTPException) as exc:
        store.save_feedback(feedback)
    assert exc.value.status_code == 404
