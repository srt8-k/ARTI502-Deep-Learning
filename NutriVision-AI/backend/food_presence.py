"""Optional Food/No-Food detector using the user's frozen ConvNeXt CNN.

The Food-101 CNN weights and the V2 linear-head checkpoint are read-only.
The two checkpoints MUST come from the same ConvNeXt feature architecture.
"""

import time
from contextlib import nullcontext
from pathlib import Path

import timm
import torch
from PIL import Image, ImageOps


class FoodPresenceDetector:
    """CNN classifies Food-101 + a frozen-feature binary Food/No-Food head."""

    def __init__(self, cnn_checkpoint: Path, head_checkpoint: Path, device):
        cnn_checkpoint = Path(cnn_checkpoint)
        head_checkpoint = Path(head_checkpoint)
        for path in (cnn_checkpoint, head_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f"Required V2 file missing: {path}")

        cnn = torch.load(cnn_checkpoint, map_location="cpu", weights_only=True)
        head = torch.load(head_checkpoint, map_location="cpu", weights_only=True)

        if cnn.get("model_name") != head.get("backbone_model_name"):
            raise ValueError("V2 head was trained with a different ConvNeXt architecture")
        self.classes = list(cnn["classes"])
        if len(self.classes) != 101:
            raise ValueError("Expected 101 Food-101 CNN classes")

        self.device = torch.device(device)
        self.backbone = timm.create_model(
            cnn["model_name"], pretrained=False, num_classes=len(self.classes)
        )
        self.backbone.load_state_dict(cnn["model_state_dict"], strict=True)
        self.backbone.to(self.device).eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        self.transform = timm.data.create_transform(
            **timm.data.resolve_model_data_config(self.backbone),
            is_training=False,
        )
        self.feature_mean = head["feature_mean"].float().to(self.device).flatten()
        self.feature_std = head["feature_std"].float().to(self.device).flatten()
        self.weight = head["linear_weight"].float().to(self.device).reshape(-1)
        self.bias = head["linear_bias"].float().to(self.device).reshape(())
        self.threshold = float(head["decision_threshold"])
        if not 0 < self.threshold < 1:
            raise ValueError("V2 decision threshold must be between 0 and 1")
        size = self.weight.numel()
        if any(x.numel() != size for x in (self.feature_mean, self.feature_std)):
            raise ValueError("V2 normalization or head dimensions do not match")
        if size != 1024:
            raise ValueError(f"Unexpected ConvNeXt feature width: {size}")

    def predict(self, image: Image.Image):
        """Return binary food score and CNN Food-101 ranking for one image."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        autocast_context = (
            torch.autocast(
                "cuda",
                dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
            )
            if self.device.type == "cuda" else nullcontext()
        )

        with torch.inference_mode():
            with autocast_context:
                maps = self.backbone.forward_features(tensor)
                features = self.backbone.forward_head(maps, pre_logits=True)
                logits = self.backbone.forward_head(maps)

            vector = features.float().reshape(-1)
            if vector.numel() != self.weight.numel():
                raise RuntimeError("CNN feature dimensions differ from the V2 head")
            normalized = (vector - self.feature_mean) / self.feature_std.clamp_min(1e-6)
            food_score = torch.sigmoid(normalized @ self.weight + self.bias).item()
            values, indices = torch.softmax(logits.float(), dim=1)[0].topk(3)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        cnn_top3 = [
            {
                "name": self.classes[int(index.item())],
                "softmax_score_percent": round(float(value.item()) * 100, 2),
            }
            for value, index in zip(values, indices)
        ]
        return {
            "food_present": bool(food_score >= self.threshold),
            "food_presence_score_percent": round(food_score * 100, 2),
            "food_presence_threshold": self.threshold,
            "cnn_food101_top3": cnn_top3,
            "inference_seconds": round(time.perf_counter() - start, 3),
        }
