"""Run from the repository root: python -m tools.diagnose_portion [image ...]

Defaults to the six local pilot photos. Reads checkpoints without writing to
the feedback database. Open-vocabulary naming is disabled by default so the
diagnostic can run offline. It does not supply any inputs to the nutrition head.
"""
import argparse
import json
import os
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="*")
    args = parser.parse_args()
    images = args.images or sorted(Path(".local/evaluation_real").glob("*/*.jpg"))
    if not images:
        parser.error("Supply at least one image path.")

    os.environ.setdefault("NUTRIVISION_OPEN_VOCAB", "0")
    from backend.ai_service import MODEL_DIR, NutriVisionService
    from backend.food_presence import FoodPresenceDetector

    service = NutriVisionService()
    detector = FoodPresenceDetector(
        Path(os.environ.get("NUTRIVISION_CONVNEXT_CHECKPOINT", MODEL_DIR / "convnext_stage3_best.pth")),
        Path(os.environ.get("NUTRIVISION_FOOD_PRESENCE_CHECKPOINT", MODEL_DIR / "food_presence_head_v2.pth")),
        service.device,
    )
    print(json.dumps({"transform": str(service.transform), "device": str(service.device),
                      "open_vocab": service.recognizer is not None}))
    for path in images:
        trace = {}

        def capture(module, inputs, output):
            trace["feature_shape"] = list(inputs[0].shape)
            trace["standardized"] = output[0].tolist()
            trace["inverse_standardized"] = (output[0] * service.target_std + service.target_mean).tolist()

        with Image.open(path) as image:
            detection = detector.predict(image)
            row = {"file": path.name, "pixels": image.size, "food": detection["food_present"],
                   "kcal": None, "grams": None}
            if detection["food_present"]:
                with service.nutrition_model.register_forward_hook(capture):
                    result = service.predict(image, include_ingredients=False)
                row.update(trace, cnn=detection["cnn_food101_top3"][0]["name"],
                           dino=result["food101_top3"][0]["name"], suggestion=result["suggested_food"],
                           kcal=result["estimated_calories"], grams=result["estimated_portion_mass_g"])
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
