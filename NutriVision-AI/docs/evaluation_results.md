# Recorded evaluation evidence (2026-11)

These are results reported from completed Colab runs. The GitHub prototype has NOT independently re-run these experiments. Detailed evidence is saved in the team's private Drive checkpoints and prediction artifacts.

## Food-101 classification
- 101 classes; official test split has 25,250 images.
- Development: 5,050 images (50/class) from the official test partition, used to choose ensemble/reranking settings.
- Held-out: remaining 20,200 images, evaluated once after fixing 75% DINOv2 + 25% ConvNeXt probability averaging.
- Held-out DINOv2: Top-1 95.03%, Top-5 99.13%.
- Held-out ConvNeXt: Top-1 92.08%, Top-5 98.90%.
- Held-out ensemble: Top-1 95.34%, Top-5 99.38%.
- These are NOT metrics for the open-vocabulary pipeline. The deployed core prototype uses DINOv2 only, not the 2-backbone ensemble.

## Calorie estimation
- Nutrition5k upper RGB images; 2,755 official training images and 507 official depth-test IDs with corresponding RGB images in the lightweight image subset.
- Ridge head on frozen DINOv2 features: validation MAE 79.69 kcal.
- Nutrition MLP on the same features: best validation MAE 53.02 kcal (epoch 23; direct calorie prediction).
- On 507 held-out evaluation images: MAE 64.99 kcal, RMSE 108.01 kcal, median absolute error 33.52 kcal; 62.72% within +/-50 kcal and 78.90% within +/-100 kcal.
- Previous ResNet18 experiment: test MAE 82.60 kcal, RMSE 114.11 kcal. Confirm matching image IDs and evaluation procedures before presenting as a strictly paired comparison; the team has previously viewed this test result.

## End-to-end prototype (qualitative smoke checks)
- Hamburger: suggested hamburger, about 479.6 kcal.
- Chicken kabsa: SigLIP2 suggested chicken kabsa, about 507.2 kcal; requires user confirmation.
- Mixed vegetables/olives: suggested mixed vegetables, about 132.7 kcal; requires user confirmation.
- Three successful API calls do NOT establish general recognition accuracy or calorie accuracy for Saudi/mixed meals.
- Measured shared DINOv2 + Nutrition MLP median inference function times in Colab: CPU 0.242 s; CUDA L4 0.018 s (models already loaded; excludes server and SigLIP2).
- Extended API with SigLIP2 in Colab: ~0.29-0.30 s for three sample images on GPU; highly environment dependent.

## Limitations / future evaluation
- Food-101 label coverage excludes kabsa, mandi, and many ingredient-level labels.
- SigLIP2 is zero-shot relative to this project and produces candidates, NOT validated ingredient detection or calibrated confidence.
- A single image does not uniquely determine ingredient weights, hidden oils, sauces or serving size.
- The nutrition model was validated on Nutrition5k, not on a representative Saudi-home-food smartphone dataset.
- Gather separate diverse, consented real-world examples with measured calories and evaluate classification, warnings, and nutrition independently.
- Never describe the combined open-vocabulary system as having 95.34% accuracy; that score belongs to the original 2-model Food-101 ensemble.
