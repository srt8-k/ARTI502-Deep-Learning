# ARTI 502 — Assignment 2: CNN Training and Testing on CIFAR-10

PyTorch implementation and experimental report resources for the ARTI 502 Deep Learning assignment.

## Final measured result

**EfficientNetV2-XL (external CIFAR-10-pretrained checkpoint): 99.04% test accuracy (9,904 / 10,000), test cross-entropy loss 0.157579.** This score was measured in Google Colab on the official CIFAR-10 test split, using 224 × 224 bicubic resize and the documented CIFAR-10 normalization.

**Important provenance:** The EfficientNetV2-XL checkpoint was **pretrained on CIFAR-10 by its publisher**, [DeKUT-DSAIL](https://huggingface.co/DeKUT-DSAIL/efficientnetv2-xl-cifar10-upsample). Our work loads and evaluates that checkpoint and adds reporting, confusion-matrix and per-class analysis. We do **not** claim to have trained these XL weights. The test split had also been evaluated during earlier exploration, so it should not be portrayed as untouched for model-selection purposes.

## Files

| Path | Purpose |
|---|---|
| [Assignment2_CIFAR10_Clean.ipynb](Assignment2_CIFAR10_Clean.ipynb) | Main submission notebook: Tasks 1–6, executed outputs, augmentation demos, historical training plots and final benchmark |
| [Assignment2_CIFAR10_Clean.py](Assignment2_CIFAR10_Clean.py) | Runnable PyTorch source with CNN, training/evaluation functions, dataset visualization, augmentations and optional XL evaluation |
| [archive/Assignment2_Original_Experiments.ipynb](archive/Assignment2_Original_Experiments.ipynb) | Full 53-cell Colab record with prior CNN fine-tuning/ensemble experiments; retained for traceability |

Do not upload `.pth` or `.safetensors` checkpoints to regular GitHub. The 831 MB external XL weights are available from the credited model repository; experiment checkpoints were saved in the original Colab/Google Drive environment.

## Tasks covered

1. **CIFAR-10 dataset:** download, training/validation split and sample counts.
2. **CNN architecture:** convolution, ReLU, max pooling, and explicit training/evaluation functions.
3. **Tensor dimensions and visualization:** batch `(N, C, H, W)` shape and Matplotlib examples.
4. **Augmentation:** ToPILImage, flips, random resized crops, color jitter, rotation, normalization, noise and ToTensor.
5. **Experiments and results:** baseline/deeper CNN training histories, ConvNeXtV2 training/validation curves, model comparisons and measured XL test loss/accuracy, confusion matrix and per-class performance.
6. **Source submission:** readable `.ipynb` and standalone `.py` files.

## Verified historical results

| Experiment | Metric/split | Accuracy | Loss |
|---|---|---:|---:|
| Custom baseline CNN (20 epochs) | Best validation | 81.60% | See notebook history |
| Deeper CNN (20 epochs) | Best validation | 89.98% | See notebook history |
| ConvNeXtV2 (epoch 4) | Training | 99.06% | 0.0309 |
| ConvNeXtV2 (epoch 4, flip TTA) | Validation | 98.58% | 0.0494 |
| Three-CNN ensemble of locally fine-tuned models | Official test | 98.84% | 0.0396 |
| **EfficientNetV2-XL (third-party pretrained weights)** | **Official test** | **99.04%** | **0.157579** |

Training and validation curves belong to **ConvNeXtV2**, not EfficientNetV2-XL. The XL checkpoint's training history is not included or claimed.

## Run in Google Colab

1. Open [the notebook](Assignment2_CIFAR10_Clean.ipynb) in Colab and select a GPU runtime for large-model evaluation.
2. Run setup, Task 1–4, and the Task 5 evaluation only if a fresh reproduction is needed. Historical figures/outputs are already saved in the notebook.
3. The companion script can be run independently:

```bash
pip install torch torchvision timm "huggingface_hub>=0.25,<2" safetensors matplotlib numpy
python Assignment-2/Assignment2_CIFAR10_Clean.py --data-dir ./cifar_data --save-dir ./assignment2_outputs

# Optional: train the small baseline CNN (20 epochs)
python Assignment-2/Assignment2_CIFAR10_Clean.py --data-dir ./cifar_data --save-dir ./assignment2_outputs --train-baseline

# Optional: re-evaluate external XL checkpoint (GPU and ~831 MB download)
python Assignment-2/Assignment2_CIFAR10_Clean.py --data-dir ./cifar_data --save-dir ./assignment2_outputs --evaluate-xl
```

The `.py` script is a **companion reproduction/demo**, not a reimplementation of every advanced fine-tuning experiment. See the archived notebook for those Colab sessions. No retraining is required merely to inspect the final assignment evidence.

## Academic attribution and evaluation notes

- Original model checkpoint: [DeKUT-DSAIL/efficientnetv2-xl-cifar10-upsample](https://huggingface.co/DeKUT-DSAIL/efficientnetv2-xl-cifar10-upsample).
- Model architecture is created through `timm` as `tf_efficientnetv2_xl` with 10 output classes; publisher's `model.safetensors` is loaded strictly.
- Numbers in this README are taken from completed Colab experiments; they are **not** calculated by cloning the repository alone.
- Do not report validation accuracy as test accuracy, or associate ConvNeXtV2 training curves with the XL weights.
