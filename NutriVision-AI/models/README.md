# Released V3 model checkpoints

The four model files are published separately from the Git source tree in the [official NutriVision AI V3 release](https://github.com/srt8-k/ARTI502-Deep-Learning/releases/tag/v3.0-submission).

Download [NutriVision-models-v3.zip](https://github.com/srt8-k/ARTI502-Deep-Learning/releases/download/v3.0-submission/NutriVision-models-v3.zip) (approximately 616 MiB) and extract the files directly into this `models/` folder:

- `convnext_stage3_best.pth` — Food-101 CNN
- `food_presence_head_v2.pth` — matching food/no-food head
- `dinov2_last4_best.pth` — Food-101 ViT and feature extractor
- `nutrition_mlp_best.pth` — released nutrition regression head

ZIP SHA-256: `3bd6c7012a6898189dec7c68c3d77e0d78cbf2f931552fe312e96c584a19df6f`

Alternatively, set `NUTRIVISION_MODEL_DIR` to the folder containing all four files. The backend loads the saved weights with strict state dictionaries. Experimental nutrition checkpoints use a different layout and cannot directly replace `nutrition_mlp_best.pth`.

SigLIP2 is optional and downloads `google/siglip2-base-patch16-256` when `NUTRIVISION_OPEN_VOCAB=1`. See [the Windows startup guide](../README.md#run-on-windows).
