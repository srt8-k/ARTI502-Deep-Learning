# NutriVision AI — Team Handoff

**Status:** GitHub integration prototype; full model inference from a fresh machine is **not yet verified**. The former Colab prototype successfully handled three smoke-test images and an SQLite correction.

## Files & ownership

| Role | Requirements | GPU? |
|---|---|---|
| Backend/ML owner | Backend code and two private checkpoints | CUDA recommended, CPU possible |
| Frontend teammate | API address and [short API client notebook](../notebooks/API_Client_Demo.ipynb) | No |
| Report teammate | [Experiment metrics](evaluation_results.md) and private research archive | No |

## Model checkpoints (private)

The backend requires:
- dinov2_last4_best.pth — trained DINOv2 Food-101 backbone and classification head.
- nutrition_mlp_best.pth — trained nutrition head and normalization data.

Obtain these from the team's private Google Drive. Place them in the NutriVision-AI/models folder, or set the NUTRIVISION_MODEL_DIR environment variable to the directory containing both files. **Never commit checkpoint weights to this public GitHub repository.** The original 133-code-cell training notebook remains private.

SigLIP2, google/siglip2-base-patch16-256, is downloaded separately on first model startup. To conserve resources, set NUTRIVISION_OPEN_VOCAB=0 to disable this optional service; the DINOv2 classifier and nutrition head will still load.

## Local startup

Use Python 3.11+, a virtual environment and PyTorch/torchvision matching the host's CUDA or CPU configuration. From the NutriVision-AI folder:

~~~bash
python -m venv .venv
# Activate your virtual environment before the next command
python -m pip install -r backend/requirements.txt
uvicorn backend.nutrivision_api:app --host 127.0.0.1 --port 8000
~~~

Verify GET http://127.0.0.1:8000/health and interactive docs at http://127.0.0.1:8000/docs. Only the computer running this server can use 127.0.0.1. Colab's 127.0.0.1 is **not accessible to teammates over the Internet**; if remote frontend access is needed, deploy a secured, reachable backend. Teammates do not need to run the heavy model weights.

## API contract

- **GET /health:** returns status, selected device, SigLIP2 enabled state.
- **POST /predict:** multipart form with image file and optional analyze_ingredients boolean. Returns prediction_id, suggested_food, food101_top3 (uncalibrated scores), open_vocab_suggestions, possible_ingredients, estimated_calories, requires_confirmation and timing.
- **POST /feedback:** JSON with prediction_id, corrected_food, confirmed_ingredients. Returns saved=true on successful local SQLite storage. Correction **does not** retrain models or retroactively fix calorie prediction.

Example feedback request:

~~~json
{
  "prediction_id": "value-from-predict-response",
  "corrected_food": "Chicken Kabsa",
  "confirmed_ingredients": ["chicken", "rice"]
}
~~~

Be explicit with the frontend: Food-101 is closed-world; SigLIP2 suggests plausible names and ingredients, not verified object detections; one image cannot reliably establish portion weights.

## Finished and remaining

**Included:** DINOv2 classification, nutrition MLP inference, optional SigLIP2 suggestions, FastAPI, SQLite text feedback, team API client example, input-size checks, test workflow, metrics documentation and private-weight instructions.

**Verified in original Colab:** samples of hamburger, chicken kabsa and a mixed vegetable plate; successful feedback submission.

**Required before team demo:** (1) run the packaged GitHub backend from a clean environment using the private checkpoints; (2) check health, real image prediction and feedback; (3) integrate the frontend against the same backend; (4) capture consented demo screenshots with no secrets or private data.

**Required before public use:** authenticated access, rate limits, explicit privacy/retention controls, logging/monitoring, secure storage, security review and more rigorous evaluation on unknown Saudi/mixed meals.

## Metric integrity

- 95.03% held-out Food-101 Top-1: **single DINOv2** (deployed classification backbone).
- 95.34% held-out Food-101 Top-1: **separate DINOv2 + ConvNeXt ensemble** (not loaded by this API).
- 64.99 kcal MAE: nutrition MLP on **507 Nutrition5k test images**; NOT calibrated real-world kabsa accuracy.
- The combined SigLIP2 system has no quantitative food recognition accuracy established by the three smoke-test images.

## Automated checks

GitHub Actions checks Python syntax, ensures only the 5-code-cell API client notebook is committed with no execution outputs, and runs lightweight SQLite feedback tests. **These checks do not load the weights** or establish end-to-end inference correctness.
