# Notebook source audit

**Source received:** NutriVision_Notebook.zip containing NutriVision_AI.ipynb.
**Status:** reviewed 133 code cells from the user's Colab export. The original uncleaned notebook is NOT committed to the public repository because its cell metadata contains Colab account identifiers and execution outputs.

## Confirmed against the uploaded notebook

| Notebook cell (0-indexed) | Purpose | Relevant backend / documentation |
|---|---|---|
| 99 | Saves \`dinov2_last4_best.pth\` with \`model_state_dict\` and \`class_names\` | \`backend/ai_service.py\` checkpoint loading |
| 100 | Development-set predictions: DINOv2 94.93% Top-1; ConvNeXt 92.10%; ensemble 95.29% | Research only; **not** production inference |
| 112 | Nutrition5k feature extraction using \`forward_features\` and \`forward_head(..., pre_logits=True)\` after Resize(320) + CenterCrop(280) | Same DINOv2 visual features in \`ai_service.py\` |
| 113 | \`NutritionMLP\` 768→256→64→6 with LayerNorm/GELU/dropout, saved means/std, selects \`Direct\` calories | Same MLP architecture and normalization in backend |
| 114 | 507 Nutrition5k evaluation images, 64.99 kcal MAE, 108.01 kcal RMSE, direct calorie method | \`docs/evaluation_results.md\` |
| 115 | Final 20,200-image held-out evaluation: DINOv2 95.03% Top-1, ensemble 95.34% Top-1 | DINOv2 only in the deployed API; ensemble remains a reported research metric |
| 116 | Shared-backbone inference logic, 280 px DINOv2 + nutrition MLP, CUDA/CPU | \`backend/ai_service.py\` |
| 122 | SigLIP2 zero-shot suggestions: kabsa, vegetables, olives, brussels sprouts | Optional \`ai_service.py\` open-vocabulary pipeline |
| 124 | Original Colab \`nutrivision_api.py\` prototype | Split into \`backend/ai_service.py\` + \`nutrivision_api.py\` |
| 127 | Original \`feedback_store.py\` | \`backend/feedback_store.py\` |
| 130–132 | Colab correction patch and successful \`/feedback\` response | Unit-tested feedback persistence |

## Accuracy reporting

- DINOv2 standalone 95.03% Food-101 Top-1 **on the 20,200 held-out images**.
- 95.34% is the separate **two-backbone ensemble**, which is NOT loaded by the production-oriented API.
- Nutrition MLP MAE 64.99 kcal on 507 Nutrition5k test images, not an error bound for unknown foods.
- The new SigLIP2-plus-DINOv2 combined system has only qualitative smoke examples, not a measured statistical accuracy.

## Reproducibility and outstanding checks

1. The source is a **chronological experiment archive** with 133 code cells and stale/failed trials. It is not a clean "Run All" reproduction pipeline.
2. The two checkpoint weights are required from the team's private Drive; they were not in the uploaded zip. The project cannot perform a clean inference cold-start in this repo without them.
3. A clean copy of the full notebook has been prepared in the conversation (outputs, execution counts and Colab account metadata removed). Add it to \`notebooks/NutriVision_AI_clean.ipynb\` when ready; do not commit the original raw export.
4. Only the SQLite feedback unit tests have been executed on GitHub Actions. The refactored API still needs an end-to-end test with actual checkpoints on the target machine.
5. SigLIP2 item suggestions are zero-shot relative to our project and do not count as verified object/ingredient detection.
6. Food-101 test data was split into development vs held-out partitions. Avoid reusing the held-out split for further model selection.

## Code comparison

- Confirmed \`NutritionMLP\` class architecture, model shape, checkpoint keys, DINOv2 \`timm\` architecture, feature extraction and normalization match the original tested cells.
- Current packaged backend intentionally differs in structure: package-relative imports, configurable model paths, image-size limits, safer fallback handling, and SQLite DB stored under a private local directory.
- The API confirmation flag remains heuristic. A top SigLIP2 label differing from Food-101 top-1 triggers confirmation in the packaged backend; this heuristic has not been systematically calibrated or evaluated.
