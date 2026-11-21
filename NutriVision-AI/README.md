# NutriVision AI V3

A university deep-learning application that scans a meal photo and estimates total energy (kcal) and portion mass (g). The ARTI502 requirements are implemented by a ConvNeXt CNN trained on Food-101, an image-feature regression model trained on Nutrition5k, and a working React scanner backed by FastAPI.

## How a scan works

1. React uploads a JPEG or PNG to `POST /predict`. In V2 mode, ConvNeXt produces Food-101 predictions and the food-presence head checks its visual features. When the detector rejects an image, the API skips nutrition inference and returns null for both estimates.
2. For food images, a fine-tuned DINOv2 classifier supplies a second Food-101 ranking and a 768-dimensional visual representation. The deployed Nutrition MLP uses standardized features to predict whole-image calories and grams automatically. No scale reading or manual portion entry is required.
3. Optional SigLIP2 supplies regional dish and ingredient suggestions. The UI distinguishes provisional names from user selections. `POST /feedback` records a corrected name and confirmed ingredients in local SQLite. Corrections do not recalculate nutrition or retrain the models.

## Run on Windows

Use Python 3.11+ and Node.js 22 LTS with npm. The Python requirements cover PyTorch, torchvision, timm, Pillow, FastAPI, Uvicorn, multipart uploads and optional SigLIP2 support via Transformers. The frontend lockfile covers React, TypeScript, Vite, Tailwind and Vitest.

**Terminal 1 — API.** From the cloned course repository, enter `NutriVision-AI`. Obtain the four release weights from the team’s private checkpoint archive and place them in `models\` using these exact names:

| File | Role |
|---|---|
| `convnext_stage3_best.pth` | ConvNeXt Food-101 classifier |
| `food_presence_head_v2.pth` | Matching binary detector head |
| `dinov2_last4_best.pth` | Food-101 classifier and nutrition feature extractor |
| `nutrition_mlp_best.pth` | Released six-output nutrition regression head and normalization |

```powershell
cd NutriVision-AI
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
$env:NUTRIVISION_MODEL_DIR = (Resolve-Path .\models).Path
$env:NUTRIVISION_FOOD_PRESENCE_MODE = "v2"
$env:NUTRIVISION_OPEN_VOCAB = "0"
.\.venv\Scripts\python.exe -m uvicorn backend.nutrivision_api:app --host 127.0.0.1 --port 8000
```

Set the model directory to your private checkpoint folder if using another location. The explicit `v2` setting enables the CNN and food gate for the demo. The source defaults to detection off. The commands use CPU or available CUDA and disable optional external model downloads. To enable regional suggestions, set `NUTRIVISION_OPEN_VOCAB=1` in the API shell and restart it; the first run needs access to the Hugging Face model or its local cache.

**Terminal 2 — scanner.** From the course repository:

```powershell
cd NutriVision-AI\frontend
npm.cmd ci
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open [the scanner](http://127.0.0.1:5173/#scanner). Vite forwards requests to the API on port 8000. Check [API health](http://127.0.0.1:8000/health) for `ready` and `v2`. The [interactive API docs](http://127.0.0.1:8000/docs) accept direct image uploads.

## Verify the demo

Upload a meal, select **Analyze photo**, and check its name, kcal and grams. Open **Correct This Result** and save a food-name correction. Try an empty plate and check that a no-food decision suppresses both numbers. Uploads are limited to 10 MB and 20 million pixels. Uploaded photos are not stored in the feedback database.

Run these checks from `NutriVision-AI` after installing both dependency sets:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd frontend
npm.cmd test
npm.cmd run typecheck
npm.cmd run build
```

The repeatability integration test skips when private checkpoints or the local pilot pizza photo are absent. The scanner itself accepts your own photos.

## Measured results and limits

Historical held-out Food-101 top-1 accuracy was 92.08% for ConvNeXt and 95.03% for DINOv2. A paired evaluation on 507 Nutrition5k dishes measured **64.96 kcal MAE and 38.36 g mass MAE** for the release MLP. An experimental density head reached 60.98 kcal and 40.56 g, improving energy error while worsening mass error. The application retains the release MLP.

A single photo does not establish physical scale or reveal the recipe. The six-photo pilot tested recognition and food rejection; it had no measured nutritional ground truth. Regional dish suggestions can be wrong. See the [historical benchmarks](docs/evaluation_results.md) and [pilot evaluation](docs/real_world_evaluation_v3.md).

## Repository guide

| Directory | Contents |
|---|---|
| `backend/`, `frontend/` | FastAPI inference and feedback; React scanner and frontend tests |
| `models/` | Private checkpoint setup instructions |
| `tests/` | Backend, feedback and regression checks |
| `tools/`, `notebooks/` | Diagnostic and comparison scripts, API client and Colab experiment |
| `research/` | Supporting food-detection and suggestion-policy experiments |
| `docs/` | Evaluation evidence and handoff notes |
