# Assignment 2 - Training and Testing CNNs on CIFAR-10

**ARTI 502 - Deep Learning, Term 1 (2026) | Group 9MA1 / Team 4**

This folder contains reusable PyTorch scripts associated with Assignment 2. **The reported scores below were recorded in our Colab experiments; cloning these scripts does not reproduce the exact scores automatically.** Run full GPU experiments to validate results independently. CIFAR-10 training/validation indices use seed 42; the official test set is reserved for final evaluation.

## Source code
- [assignment2_cifar10.py](assignment2_cifar10.py): dataset loading, three-block baseline CNN, deeper CNN, modified ResNet18, training/evaluation functions, curves, and held-out testing.
- [efficientnetv2m.py](efficientnetv2m.py): ImageNet-pretrained **EfficientNetV2-M CNN**, CutMix, mixed precision, AdamW/OneCycleLR, 20-epoch fine-tuning, checkpoints, and validation-selected TTA.

- [augmentation_demo.py](augmentation_demo.py): reproducible Task 4 visualization of ToPILImage, horizontal flip, resized crop, color jitter, rotation, Gaussian noise, and normalization.

## Local/Colab setup
```bash
pip install torch torchvision timm matplotlib numpy
python Assignment-2/augmentation_demo.py
python Assignment-2/assignment2_cifar10.py --model baseline --epochs 20 --batch-size 128
python Assignment-2/assignment2_cifar10.py --model deeper --epochs 20 --batch-size 128
python Assignment-2/assignment2_cifar10.py --model resnet18 --epochs 100 --batch-size 128
python Assignment-2/efficientnetv2m.py --epochs 20 --batch-size 16 --accumulation 4
```
Run with CUDA for practical GPU training. Pretrained EfficientNetV2 weights are downloaded automatically; the CIFAR-10 dataset is downloaded through torchvision.

## Recorded CIFAR-10 results
These results are **not fabricated**; they are transcribed from the user's completed Colab logs and corresponding report screenshots. **Validation is not test**.

| Experiment | Best validation accuracy | Final test accuracy | Final test loss |
|---|---:|---:|---:|
| Baseline CNN | 81.60% | Not independently recorded | - |
| Deeper CNN | 89.98% | Not independently recorded | - |
| ResNet18 (no TTA) | 95.32% | 94.88% | 0.2008 |
| ResNet18 + horizontal flip | 95.72% | 95.37% | 0.1633 |
| EfficientNetV2-M (original) | 98.78% | Not separately recorded | - |
| **EfficientNetV2-M + horizontal flip** | **99.08%** | **98.59%** | **0.0990** |
| DINOv2 + flip (**Transformer; optional only**) | 98.58% | 98.48% | 0.1489 |

The strongest result *for the CNN assignment* is **98.59% on the official 10,000-image test set** using ImageNet-pretrained EfficientNetV2-M + horizontal flip TTA, selected on the 5,000-image validation split.

### Experimental caution
- Pretrained EfficientNetV2-M is a CNN but **not trained entirely from scratch**.
- DINOv2 is a **Vision Transformer**, provided only as an extra comparison; it does not replace the CNN requirements.
- Training loss/accuracy from the advanced CutMix experiment are calculated with mixed-label augmentation and therefore are not directly comparable to clean-image validation metrics.
- Test accuracy should only be measured after model selection using validation, not used for repeated hyperparameter tuning.
- The scripts are clean reusable implementations based on the Colab methodology; historical GPU run checkpoint files are **not** included in Git (large binaries).

### Attribution
The EfficientNetV2-M fine-tuning approach was informed by [hankyul2/EfficientNetV2-pytorch](https://github.com/hankyul2/EfficientNetV2-pytorch); this repository's implementation is original to this submission and uses the published timm model identifier \`tf_efficientnetv2_m.in21k_ft_in1k\`.
