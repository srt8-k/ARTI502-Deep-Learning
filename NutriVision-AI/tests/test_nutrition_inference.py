import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from fastapi import UploadFile
from PIL import Image

from backend import nutrivision_api as api
from backend.ai_service import MODEL_DIR, NutriVisionService
from backend.food_presence import FoodPresenceDetector


class NutritionTests(unittest.TestCase):
    def test_repeat_automatic_pizza_prediction_through_route(self):
        photo = Path(__file__).resolve().parents[1] / ".local/evaluation_real/food101/01_pizza.jpg"
        checkpoints = ["nutrition_mlp_best.pth", "dinov2_last4_best.pth",
                       "convnext_stage3_best.pth", "food_presence_head_v2.pth"]
        if not photo.is_file() or not all((MODEL_DIR / name).is_file() for name in checkpoints):
            self.skipTest("Private release checkpoints and pilot pizza required")
        with patch("backend.ai_service.ENABLE_OPEN_VOCAB", False):
            service = NutriVisionService()
        detector = FoodPresenceDetector(MODEL_DIR / checkpoints[2], MODEL_DIR / checkpoints[3], service.device)
        with patch.object(api, "service", service), patch.object(api, "food_detector", detector), \
             patch.object(api, "register_prediction", side_effect=lambda r: dict(r, prediction_id="repeat-test")):
            replies = []
            for _ in range(2):
                with photo.open("rb") as stream:
                    replies.append(api.predict(UploadFile(file=stream), analyze_ingredients=False))
        for reply in replies:
            self.assertTrue(reply["food_present"])
            self.assertEqual(reply["suggested_food"], "pizza")
            self.assertEqual(reply["calorie_unit"], "kcal")
            for key in ("estimated_calories", "estimated_portion_mass_g"):
                self.assertIsInstance(reply[key], float)
                self.assertGreater(reply[key], 0)
                self.assertEqual(reply[key], replies[0][key])

    def test_deployed_decoder_uses_standardized_features_and_first_two_log_targets(self):
        service = NutriVisionService.__new__(NutriVisionService)
        service.device = torch.device("cpu")
        service.transform = Mock(return_value=torch.zeros(3, 280, 280))
        service.classes = ["dish", "b", "c"]
        service.recognizer = None
        service.food_model = Mock()
        service.food_model.forward_head.side_effect = [torch.full((1, 768), 7.), torch.tensor([[5., 2., 1.]])]
        service.feature_mean = torch.ones(768)
        service.feature_std = torch.full((768,), 3.)
        service.target_mean = torch.tensor([3., 4., 0., 0., 0., 0.])
        service.target_std = torch.tensor([2., 0.5, 1., 1., 1., 1.])
        targets = torch.tensor([[408., 254., 1., 2., 3., 4.]])
        service.nutrition_model = Mock(return_value=(targets.log1p() - service.target_mean) / service.target_std)
        result = service.predict(Image.new("RGB", (32, 32)))
        torch.testing.assert_close(service.nutrition_model.call_args.args[0], torch.full((1, 768), 2.))
        self.assertEqual(result["estimated_calories"], 408.)
        self.assertEqual(result["estimated_portion_mass_g"], 254.)

    def test_predict_route_skips_nutrition_when_v2_rejects_image(self):
        image = io.BytesIO()
        Image.new("RGB", (32, 32)).save(image, format="PNG")
        image.seek(0)
        service = Mock(device=SimpleNamespace(type="cpu"))
        detector = Mock()
        detector.predict.return_value = dict(food_present=False, food_presence_score_percent=1.,
                                            food_presence_threshold=0.5, inference_seconds=0.)
        with patch.object(api, "service", service), patch.object(api, "food_detector", detector), \
             patch.object(api, "register_prediction", side_effect=lambda result: result):
            result = api.predict(UploadFile(file=image), analyze_ingredients=False)
        service.predict.assert_not_called()
        self.assertFalse(result["food_present"])
        self.assertIsNone(result["estimated_calories"])
        self.assertIsNone(result["estimated_portion_mass_g"])
