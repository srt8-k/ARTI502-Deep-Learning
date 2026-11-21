# NutriVision AI frontend

React, TypeScript, Vite, Tailwind CSS, and Lucide React frontend for the existing FastAPI prototype.

## Local development

1. Start the backend from the repository root on `127.0.0.1:8000` as described in the root README.
2. In `frontend/`, run `npm install` and then `npm run dev`.
3. Open the URL printed by Vite, normally `http://localhost:5173`.

Vite proxies `/health`, `/predict`, and `/feedback` to the local backend. The browser calls same-origin paths; no backend CORS change is needed during local development. For a deployed build, configure the hosting reverse proxy to forward those three paths to the backend on the same origin. The Vite development proxy does not apply to built static files.

## Checks

Prediction and feedback requests time out after 180 seconds by default. To change this in PowerShell, set `$env:VITE_REQUEST_TIMEOUT_MS="120000"` before starting Vite or building the frontend. Retry manually after checking the backend. A timed-out prediction may still finish on the server. Retrying feedback updates the same prediction.

- `npm run typecheck`
- `npm test`
- `npm run build`

## API contract and limits

`POST /predict` sends `multipart/form-data` with required `image` and optional `analyze_ingredients` fields. The UI accepts JPEG and PNG files up to 10 MB. The server also rejects images larger than 20 million pixels; the UI displays that server error. The result is shown only after a successful API response.

`POST /feedback` sends `prediction_id`, `corrected_food`, and `confirmed_ingredients` as JSON. Local edits are marked as unsaved until that request succeeds. Feedback does not retrain the models or update the calorie estimate.

Food and ingredient names are unverified AI suggestions. Food-101 softmax scores are raw model scores, not calibrated probabilities. Calories are estimates for the photographed plate, not individual ingredients. Uploaded images are sent to the API for inference; the API stores prediction metadata and feedback, not image bytes.
