# ARTI502 - Deep Learning

Course repository for **ARTI502 Deep Learning** assignments and applied work.

## Projects

- [Assignment 1 - MNIST neural network](Assignment-1/)
- [Assignment 2 - CIFAR-10 CNN](Assignment-2/)
- [NutriVision AI - food recognition and calorie estimation](NutriVision-AI/)

## Assignment 1

Training and testing a fully connected neural network on MNIST with PyTorch, SGD, learning-rate tuning, and architecture experiments. See [Assignment-1/README.md](Assignment-1/README.md).

## Assignment 2

CIFAR-10 experiments using CNNs, augmentation and transfer learning. Previously recorded EfficientNetV2-M results: 99.08% validation accuracy after flip TTA and 98.59% test accuracy, as reported by the course experiments. See [Assignment-2/README.md](Assignment-2/README.md).

## NutriVision AI

A prototype integrating DINOv2 classification, a nutrition regression MLP, optional SigLIP2 food/ingredient suggestions, FastAPI endpoints and SQLite feedback. See [NutriVision-AI/README.md](NutriVision-AI/README.md) for setup, model provenance, published project metrics and limitations.

Each assignment/project has its own setup instructions. The root requirements.txt applies only to the original coursework, **not** to the NutriVision backend.
