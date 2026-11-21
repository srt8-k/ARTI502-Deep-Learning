"""Inference service: DINOv2 food recognition, nutrition MLP, optional SigLIP2.

Weights are NOT distributed with this repository. See ../models/README.md.
"""
import os
import time
from contextlib import nullcontext
from pathlib import Path

import timm
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T
from .food_choice import food_choice


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("NUTRIVISION_MODEL_DIR", PROJECT_ROOT / "models"))
ENABLE_OPEN_VOCAB = os.environ.get("NUTRIVISION_OPEN_VOCAB", "1") == "1"

EXTRA_DISHES = [
    "Saudi Arabian chicken kabsa with spiced rice",
    "Saudi Arabian lamb kabsa with rice",
    "chicken mandi with rice",
    "lamb mandi with rice",
    "chicken biryani",
    "grilled chicken with rice",
    "mixed vegetables on a plate",
    "vegetable salad",
    "grilled meat with rice",
    "roasted vegetables",
]
INGREDIENTS = [
    "olives", "celery", "brussels sprouts", "chicken", "beef",
    "lamb", "rice", "potatoes", "tomatoes", "lettuce", "cucumber",
    "carrots", "broccoli", "cheese", "eggs", "beans", "peas",
    "fish", "shrimp",
]


class NutritionMLP(torch.nn.Module):
    """The exact six-output head used in the saved nutrition_mlp_best.pth."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(256, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(64, 6),
        )

    def forward(self, features):
        return self.net(features)


class NutriVisionService:
    """Load each model only once, then handle image predictions."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        food_path = MODEL_DIR / "dinov2_last4_best.pth"
        nutrition_path = MODEL_DIR / "nutrition_mlp_best.pth"
        for path in (food_path, nutrition_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing model checkpoint: {path}. See models/README.md."
                )

        food_ckpt = torch.load(food_path, map_location="cpu", weights_only=True)
        nutrition_ckpt = torch.load(
            nutrition_path, map_location="cpu", weights_only=True
        )

        self.classes = list(food_ckpt["class_names"])
        if len(self.classes) != 101:
            raise ValueError("Expected exactly 101 ordered Food-101 classes.")

        self.food_model = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=101,
            img_size=280,
        )
        self.food_model.load_state_dict(food_ckpt["model_state_dict"], strict=True)
        self.food_model.to(self.device).eval()

        self.nutrition_model = NutritionMLP()
        self.nutrition_model.load_state_dict(
            nutrition_ckpt["model_state_dict"], strict=True
        )
        self.nutrition_model.to(self.device).eval()

        self.feature_mean = nutrition_ckpt["feature_mean"].float().to(self.device)
        self.feature_std = nutrition_ckpt["feature_std"].float().to(self.device)
        self.target_mean = nutrition_ckpt["target_mean"].float().to(self.device)
        self.target_std = nutrition_ckpt["target_std"].float().to(self.device)
        if self.feature_mean.numel() != 768 or self.target_mean.numel() != 6:
            raise ValueError("Nutrition checkpoint has unexpected normalization shape.")

        # Identical deterministic preprocessing used in the saved evaluations.
        self.transform = T.Compose([
            T.Resize(320, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(280),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        self.recognizer = None
        if ENABLE_OPEN_VOCAB:
            # Lazy import keeps the core DINOv2 service usable without SigLIP2.
            from transformers import pipeline
            self.recognizer = pipeline(
                "zero-shot-image-classification",
                model="google/siglip2-base-patch16-256",
                device=0 if self.device.type == "cuda" else -1,
            )
        self.dish_candidates = list(dict.fromkeys(
            [name.replace("_", " ") for name in self.classes] + EXTRA_DISHES
        ))

    def _visual_suggestions(self, image, include_ingredients):
        if self.recognizer is None:
            return [], []

        dish_scores = self.recognizer(
            image,
            candidate_labels=self.dish_candidates,
            hypothesis_template="This is a photo of {}.",
        )
        dishes = [item["label"] for item in dish_scores[:5]]
        if not include_ingredients:
            return dishes, []

        width, height = image.size
        regions = [
            image,
            image.crop((0, 0, int(width * 0.7), int(height * 0.7))),
            image.crop((int(width * 0.3), 0, width, int(height * 0.7))),
            image.crop((0, int(height * 0.3), int(width * 0.7), height)),
            image.crop((int(width * 0.3), int(height * 0.3), width, height)),
        ]
        batch_results = self.recognizer(
            regions,
            candidate_labels=INGREDIENTS,
            hypothesis_template="This is a photo of {}.",
            batch_size=5,
        )
        score_map = {ingredient: 0.0 for ingredient in INGREDIENTS}
        for region_results in batch_results:
            for item in region_results:
                label = item["label"]
                score_map[label] = max(score_map[label], float(item["score"]))
        ingredients = sorted(score_map, key=score_map.get, reverse=True)[:5]
        return dishes, ingredients

    def predict(self, image: Image.Image, include_ingredients: bool = True):
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else nullcontext()
        )
        with torch.inference_mode():
            with autocast_context:
                tokens = self.food_model.forward_features(tensor)
                features = self.food_model.forward_head(tokens, pre_logits=True)
                logits = self.food_model.forward_head(tokens)

            probabilities = torch.softmax(logits.float(), dim=1)[0]
            values, indices = probabilities.topk(3)
            normalized_features = (
                features.float() - self.feature_mean
            ) / self.feature_std.clamp_min(1e-6)
            nutrition_output = self.nutrition_model(normalized_features)
            restored = (
                nutrition_output * self.target_std + self.target_mean
            )
            calories = torch.expm1(restored[:, 0].clamp(0, 9)).clamp_min(0).item()
            # Target index 1 = total_mass_g, trained on Nutrition5k.
            portion_mass = torch.expm1(restored[:, 1].clamp(0, 9)).clamp_min(0).item()

        top3 = [
            {
                "name": self.classes[int(index)],
                "softmax_score_percent": round(float(score) * 100, 2),
            }
            for score, index in zip(values, indices)
        ]
        dishes, ingredients = self._visual_suggestions(image, include_ingredients)
        suggestion, candidates = food_choice(top3[0]["name"], dishes)

        top1 = float(values[0])
        margin = float(values[0] - values[1])
        # Review heuristic ONLY. It is not a calibrated OOD classifier.
        requires_confirmation = (
            top1 < 0.50 or margin < 0.20 or suggestion is None
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        return {
            "suggested_food": suggestion,
            "food_candidates": candidates,
            "food101_top3": top3,
            "open_vocab_suggestions": dishes,
            "possible_ingredients": ingredients,
            "estimated_calories": round(calories, 1),
            "estimated_portion_mass_g": round(portion_mass, 1),
            "calorie_unit": "kcal",
            "requires_confirmation": bool(requires_confirmation),
            "user_can_correct_food": True,
            "message": (
                "Food and ingredient suggestions are unverified. Confirm the "
                "dish and portion size. The model estimates calories for the "
                "whole image, not for individually segmented ingredients."
            ),
            "device": self.device.type,
            "inference_seconds": round(elapsed, 3),
        }
