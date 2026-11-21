"""Exercise the real feedback router and SQLite without loading ML models."""

import asyncio
import gc
import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI

from backend import feedback_store as store


async def post_feedback(app, payload):
    """Send one JSON request through ASGI without opening a network socket."""
    body = json.dumps(payload).encode()
    scope = {
        "type": "http", "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": "/feedback", "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }
    messages = []

    async def receive():
        return {"type": "http.request", "body": body}

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)
    return messages[0]["status"], json.loads(messages[-1]["body"])


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]))
        # Release SQLite handles left to garbage collection by the production store
        # before Windows attempts to remove the isolated test directory.
        self.addCleanup(gc.collect)
        self.enterContext(patch.object(store, "DB_PATH", Path(tmp) / "feedback.sqlite3"))
        store.init_db()
        self.app = FastAPI()
        self.app.include_router(store.router)
        self.prediction = store.register_prediction({"suggested_food": "hamburger"})
        self.payload = {
            "prediction_id": self.prediction["prediction_id"],
            "corrected_food": "Chicken Kabsa",
            "confirmed_ingredients": ["chicken", "rice"],
        }

    def test_submission_and_repeat_update_same_record(self):
        for _ in range(2):
            status, reply = asyncio.run(post_feedback(self.app, self.payload))
            self.assertEqual(status, 200)
            self.assertEqual(reply, {
                "saved": True, "prediction_id": self.payload["prediction_id"],
                "corrected_food": "Chicken Kabsa",
            })
        with closing(store.connect()) as db:
            rows = db.execute(
                "SELECT prediction_id, corrected_food, confirmed_ingredients FROM predictions"
            ).fetchall()
        self.assertEqual(rows, [(self.payload["prediction_id"], "Chicken Kabsa", '["chicken", "rice"]')])

    def test_unknown_and_invalid_ids_do_not_change_record(self):
        for prediction_id, expected in [("unknown", 404), ("", 422)]:
            with self.subTest(prediction_id=prediction_id):
                status, reply = asyncio.run(post_feedback(
                    self.app, {**self.payload, "prediction_id": prediction_id}
                ))
                self.assertEqual(status, expected)
                self.assertIn("detail", reply)
        with closing(store.connect()) as db:
            self.assertEqual(db.execute("SELECT corrected_at FROM predictions").fetchall(), [(None,)])
