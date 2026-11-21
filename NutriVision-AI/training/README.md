# NutriVision AI — Original Training Source

This folder provides **training source code from the user-supplied original Colab notebook** (`NutriVision_AI.ipynb`, 133 code cells), rather than replacing historical experiments with invented code. All output cells, execution counts, embedded output images and Colab widget state are stripped. The code source in each extracted original cell is preserved verbatim; a 0-based source cell index is saved in notebook cell metadata.

| Notebook | Original cell IDs (0-based) | What the source contains |
|---|---|---|
| `Original_NutriVision_Research.ipynb` | 0–132 | Full historical progression, including early baseline experiments and API prototyping; **not** a Run-All notebook |
| `ConvNeXt_Food101_Training.ipynb` | 64–66 | Transfer learning of ConvNeXt: frozen head stage, then stage-3 fine-tuning, validation, checkpoint saving |
| `DINOv2_Food101_Training.ipynb` | 68–70, 74–76, 98–99 | Head training, last-two and last-four block fine-tuning, validation, saved `dinov2_last4_best.pth` |
| `Nutrition_MLP_Training.ipynb` | 112–114 | Nutrition5k data preparation, DINOv2 frozen feature extraction, Ridge baseline, six-output regression MLP, official held-out check |
| `FoodPresence_V2_Training_Source.py` | Separate Nov 2026 Colab training source | Frozen ConvNeXt features plus weighted logistic regression on Food5K and Roboflow hard-negative examples; outputs the V2 head |

## Model training provenance

- **ConvNeXt Food-101**: `convnext_base.fb_in22k_ft_in1k_384`, 101 classes, original `convnext_stage3_best.pth` checkpoint; training code in original cells 64–66.
- **DINOv2 Food-101**: `vit_base_patch14_dinov2.lvd142m`, 101 classes, original `dinov2_last4_best.pth`; cells 68–70, 75, 98–99, dependent ensemble validation cells 74/76.
- **Nutrition MLP**: six-output 768→256→64→6 MLP, original `nutrition_mlp_best.pth`; features extracted from the fine-tuned DINOv2 and Nutrition5k; cells 112–114.
- **Food Presence V2**: the standalone `FoodPresence_V2_Training_Source.py` was located among the project session files (it is **not** a cell of the uploaded research notebook). It uses the frozen released ConvNeXt weights, a pre-existing V1 food-presence head and Food5K/Roboflow examples. This is source evidence for V2 development, **not** a fresh rerun or proof that its output hash matches the released V2 head.

## Reproduction and limits

1. Start the original staged-extract notebooks in Google Colab with a suitable GPU. Some sections need substantial GPU memory; the final DINOv2 last-four training asserts CUDA bfloat16 capability.
2. The added setup cell in the ConvNeXt and DINOv2 extracts mounts Drive, downloads Food-101, and loads the historical `data_split.pth` if available. If absent, it reconstructs the 90% training / 10% validation random split using seed 42, following the original notebook's split code. This is a reconstruction of the setup, not proof of byte-identical original training state.
3. Run ConvNeXt before the DINOv2 ensemble-comparison stages. DINOv2's subsequent stages depend on earlier saved checkpoints and validation prediction files. Training checkpoints may only be saved if the measured validation score improves.
4. For Nutrition MLP, place `dinov2_last4_best.pth` in the expected Drive directory. The original feature-building source uses public Nutrition5k data from a separate dataset mirror and official split lists. Its explicit shape assertions reflect the dataset state at training time.
5. **Do not use “Run All” on `Original_NutriVision_Research.ipynb`.** It is a historical log of iterative experiments, with cells that require previous checkpoints, particular runtime states and Google Colab commands.
6. Source code has been validated as well-formed Jupyter notebooks, but **none of these training experiments has been rerun** during preparation of this submission. Historical Food-101 and Nutrition5k performance figures in the project README are reported from prior runs, not certified by CI.
7. Food Presence V2 uses a historical project workspace path, `food_presence_head_food5k_v1.pth`, an original Roboflow dataset ZIP and Food5K. These **prerequisites are not in the release archive**, so the standalone V2 source does not currently reproduce the checkpoint from a fresh Git clone. Do not upload third-party image datasets or the private V1 file without verifying permissions.
8. The four deployed `.pth` inference weights are published at the [v3.0-submission GitHub Release](https://github.com/srt8-k/ARTI502-Deep-Learning/releases/tag/v3.0-submission). **Do not commit datasets, Drive files or model weights into the source repository.**

The uploaded archive contains a mixture of baseline, staged training, model comparison, inference and API experiments. The smaller model-specific notebooks and the V2 script demonstrate how the deployed models were developed. Only the original research notebook is an archival sequence of experiments; it is not a reproducible one-click training pipeline.

## Environment and execution notes

- `ConvNeXt_Food101_Training.ipynb`, `DINOv2_Food101_Training.ipynb` and `Nutrition_MLP_Training.ipynb` are **Colab-stage extracts**; they include setup or stage-selection dependencies. Do not claim they are tested end-to-end in an empty Colab runtime.
- `FoodPresence_V2_Training_Source.py` requires Colab, an NVIDIA GPU, a pre-existing V1 checkpoint, Food5K and the separately acquired original Roboflow zip; it deliberately refuses to overwrite an existing V2 checkpoint.
- Keep historical results separate from current demo inference. The official held-out numbers were not recomputed by the release CI.
- Training datasets and original baseline source images were deliberately not bundled into the public source repository.
