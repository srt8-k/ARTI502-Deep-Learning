# Assignment 1 - Training and Testing a Neural Network with MNIST

This folder contains the implementation for **ARTI502 Assignment 1** using PyTorch.

## Files

- `assignment1_dl.py` — Tasks 1–7: load MNIST, inspect data, build the baseline network, train it, plot training metrics, and test it.
- `tuning_experiments.py` — Task 8: compare learning rate, momentum, hidden neurons, and hidden-layer configurations.

## Baseline model

```text
Input: 784
Hidden layer: 128 neurons + ReLU
Output: 10 classes
Optimizer: SGD
Learning rate: 0.01
Momentum: 0.9
Epochs: 5
Batch size: 32
```

## Hyperparameter tuning results

| Experiment | Learning Rate | Momentum | Loss | Train Accuracy | Test Accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.01 | 0.9 | 0.0725 | 97.85% | 97.12% |
| Low Learning Rate | 0.001 | 0.9 | 0.2695 | 92.38% | 92.65% |
| High Learning Rate | 0.05 | 0.9 | 0.0467 | 98.49% | 96.99% |
| Lower Momentum | 0.01 | 0.5 | 0.2008 | 94.37% | 94.39% |

The baseline setting produced the highest test accuracy among these learning-rate and momentum experiments.

## Architecture tuning results

| Experiment | Hidden Layers | Loss | Train Accuracy | Test Accuracy |
|---|---|---:|---:|---:|
| 64 Neurons | [64] | 0.0869 | 97.39% | 96.79% |
| 128 Neurons | [128] | 0.0725 | 97.85% | 97.12% |
| 256 Neurons | [256] | 0.0664 | 98.04% | 97.40% |
| Two Hidden Layers | [128, 64] | 0.0575 | 98.15% | 96.85% |

The `[256]` model achieved the highest test accuracy in the architecture experiments.

## Run

From the repository root:

```bash
python Assignment-1/assignment1_dl.py
python Assignment-1/tuning_experiments.py
```

> Exact numerical results can vary slightly between runs because of random initialization and data shuffling. The tuning script fixes the random seed to make its comparisons reproducible.
