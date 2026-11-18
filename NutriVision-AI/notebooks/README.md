# NutriVision training notebook

## Source audit

The original Colab notebook \`NutriVision_AI.ipynb\` was received and reviewed: 133 code cells covering multiple earlier training attempts, official evaluations, and working API experiments. See [the source audit](../docs/notebook_audit.md) for a detailed mapping to the packaged backend.

## Before adding the notebook to this public repository

**Never upload the raw Google Colab export.** Its execution metadata contains user account identifiers and output images/logs. A sanitized version prepared in the project conversation removes execution outputs/counts and Colab-specific per-cell account metadata while keeping all source cells. Store that as:

\`notebooks/NutriVision_AI_clean.ipynb\`

This notebook is a research archive, not a single clean top-to-bottom runnable training pipeline. Do not use "Run All" without reworking the experiment dependencies and paths. Keep private weights, image uploads and datasets out of GitHub.

The notebook itself has not yet been uploaded to this repository because the GitHub text-only connector did not support copying the sandbox-generated .ipynb file directly.
