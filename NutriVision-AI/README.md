# NutriVision AI

Food recognition and calorie estimation prototype for ARTI502 Deep Learning.

## Components

- **DINOv2**: fine-tuned Food-101 classifier (101 classes) and shared 768-dimensional visual features.
- **Nutrition MLP**: estimates total calories directly from the DINOv2 features; no second visual backbone is needed.
- **SigLIP2** *(optional)*: zero-shot suggestions for regional dishes (e.g., kabsa/mandi) and plausible ingredients; outputs are NOT verified ingredient detections.
- **FastAPI**: POST /predict, POST /feedback, and GET /health. SQLite stores text feedback and prediction metadata, not images.

The original two-model classification ensemble achieved **95.34% held-out Food-101 Top-1**, but this deployment uses DINOv2 alone (**95.03% held-out Top-1**). Nutrition MLP achieved **64.99 kcal MAE** on 507 Nutrition5k test images. These metrics do not establish performance of the complete system on arbitrary smartphone or Saudi dishes. See docs/evaluation_results.md.

## Structure

~~~text
NutriVision-AI/
  backend/
    __init__.py
    ai_service.py
    nutrivision_api.py
    feedback_store.py
    requirements.txt
    .env.example
  models/README.md
  tests/test_feedback_store.py
  docs/evaluation_results.md
  notebooks/README.md
  notebooks/API_Client_Demo.ipynb
  frontend/README.md
~~~

The team-only API usage example is in [notebooks/API_Client_Demo.ipynb](notebooks/API_Client_Demo.ipynb). The full training/research notebook and checkpoint weights remain in the team's private Drive and are not included.

## Run locally (Python 3.11+)

1. Clone the course repository and change into NutriVision-AI.
2. Create and activate a Python virtual environment.
3. Install PyTorch/torchvision appropriate to your platform, then: pip install -r backend/requirements.txt
4. Put dinov2_last4_best.pth and nutrition_mlp_best.pth in models/, or set NUTRIVISION_MODEL_DIR to their shared directory.
5. For the lighter version without SigLIP2, set NUTRIVISION_OPEN_VOCAB=0 before startup. Default is enabled (1); first run downloads the external SigLIP2 weights and needs substantial memory.
6. Start from the NutriVision-AI folder:

~~~bash
uvicorn backend.nutrivision_api:app --host 127.0.0.1 --port 8000
~~~

Open http://127.0.0.1:8000/docs for interactive API documentation.

**Windows PowerShell**: Set environment variables with $env:NUTRIVISION_MODEL_DIR="C:\path\to\checkpoints" and $env:NUTRIVISION_OPEN_VOCAB="0".
**Linux/macOS**: export NUTRIVISION_MODEL_DIR=/path/to/checkpoints and export NUTRIVISION_OPEN_VOCAB=0.

CUDA GPU is used automatically when available; CPU is supported. CPU startup and inference speed depend on device and enabled models. Only run one Uvicorn worker for this memory-intensive prototype.

## API contract

**GET /health** returns startup status and device.

**POST /predict** accepts multipart/form-data:
- image: JPEG or PNG, maximum 10 MB / 20 million pixels
- analyze_ingredients: optional boolean, default true

Example:

~~~bash
curl -X POST http://127.0.0.1:8000/predict -F "image=@food.jpg" -F "analyze_ingredients=true"
~~~

Response includes prediction_id, suggested_food, food101_top3 (raw softmax percentage), open_vocab_suggestions, possible_ingredients, estimated_calories, calorie_unit, requires_confirmation, device, inference_seconds, and message. No classification confidence is a calibrated probability. Always allow users to correct labels.

**POST /feedback** accepts JSON:

~~~json
{
  "prediction_id": "copy-the-id-from-predict",
  "corrected_food": "Chicken Kabsa",
  "confirmed_ingredients": ["chicken", "rice"]
}
~~~

Returns saved=true on success. Feedback storage does NOT retrain either model and does not store the uploaded picture.

## Tests

Install pytest and fastapi then, from NutriVision-AI, run: python -m pytest tests -q

Unit tests cover SQLite feedback. Inference integration tests require both private checkpoints, suitable hardware and downloading SigLIP2 when enabled. The deployment was smoke-tested in Colab; code in this repository still needs a fresh-machine reproducibility test before release.

## Development safety and limitations

- This is a **local development prototype**, not a hardened public service. Add authentication, authorization, CORS allow-list, rate limits, error monitoring, deployment configuration and a privacy/retention policy before exposing it to the Internet.
- The temporary confirmation heuristic is not a validated out-of-distribution detector. Food/ingredient suggestions can be wrong even when raw softmax is high.
- Calorie output is for the whole photographed plate; it does not estimate every ingredient's individual mass or accurately handle unseen recipes.
- Keep weights, images, personal data, tokens, .env and SQLite databases out of the public course repository.
