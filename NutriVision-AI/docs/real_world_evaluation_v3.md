# NutriVision AI V3 — Real-World Pilot Evaluation

## Objective
Evaluate the deployed food-presence detector and inspect
food-recognition and nutrition outputs on six photographs.

## Experimental setup
- System: NutriVision AI V3
- Detection: ConvNeXt features with Food Presence Head V2
- Classification: ConvNeXt and DINOv2 (Food-101)
- Regional dish suggestions: SigLIP2
- Nutrition: DINOv2 features and six-output regression MLP
- Runtime: Windows, CPU, FastAPI
- Samples: 4 food images and 2 non-food images
- Classification threshold: 0.5
- No retraining or threshold tuning during this evaluation

## Food detection results
- True Positives: 4
- True Negatives: 2
- False Positives: 0
- False Negatives: 0
- Detection Accuracy: 6/6 = 100%
- Food Recall: 4/4 = 100%
- False Acceptance Rate: 0/2 = 0%

These are descriptive results for six images only.
They do not establish general population accuracy.

## Individual predictions

| Image | Reference Label | Food Detected | ConvNeXt Top-1 | SigLIP2 Top-1 | Estimated kcal | Estimated mass (g) |
|---|---|---|---|---|---|---|
| 01_pizza.jpg | pizza | True | pizza | pizza | 452.9 | 286.9 |
| 02_hamburger.jpg | hamburger | True | hamburger | hamburger | 562.4 | 355.9 |
| 03_mandi.jpg | mandi | True | chicken_curry | Saudi Arabian chicken kabsa with spiced rice | 578.9 | 637.2 |
| 04_kabsa.jpg | kabsa | True | chicken_curry | Saudi Arabian lamb kabsa with rice | 163.4 | 161.9 |
| 05_empty_plate.jpg | no_food | False |  |  |  |  |
| 06_kitchen_utensils.jpg | no_food | False |  |  |  |  |

## Discussion

Both Food-101 examples, pizza and hamburger, were
classified correctly by ConvNeXt and DINOv2.

Both regional-food examples were classified as
chicken curry by the Food-101 classifiers, illustrating
the limited coverage of Saudi dishes in Food-101.

SigLIP2 labeled the mandi image as chicken kabsa.
For kabsa it suggested lamb kabsa, matching the dish
family but not establishing the exact meat content.

The two non-food examples were correctly rejected.
No calories or portion mass were estimated for them.

## Nutrition evaluation limitations

The displayed calories and portion masses are
model predictions, not verified nutritional measurements.

These six photographs lack measured ground-truth
calories and mass. Therefore calorie MAE and mass MAE
cannot be calculated from this pilot.

The separate held-out Nutrition5k evaluation reported:
- Calorie MAE: 64.99 kcal
- Portion mass MAE: 38.31 g
- Number of test images: 507

These Nutrition5k metrics should not be interpreted
as verified errors on Saudi meals or arbitrary photos.

## Validity limitations

- Very small convenience sample (n=6)
- Image labels and downloaded content require visual audit
- No guarantee that samples represent real users
- This pilot must not be mixed with original benchmark metrics
- SigLIP2 suggestions are not validated ingredient detections
- Food-presence scores are not calibrated probabilities

## Evidence
See evaluation_6.csv for per-image observations.
Detailed model-training checkpoints and experiment
artifacts are preserved separately in the project Drive.
