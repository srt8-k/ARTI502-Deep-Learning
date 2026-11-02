# ARTI502 - Deep Learning

Course repository for **ARTI502 Deep Learning** assignments and practical work.

## Repository Structure

```text
ARTI502-Deep-Learning/
├── Assignment-1/
│   ├── README.md
│   ├── assignment1_dl.py
│   └── tuning_experiments.py
├── Assignment-2/
│   ├── README.md
│   ├── assignment2_cifar10.py
│   ├── augmentation_demo.py
│   └── efficientnetv2m.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Assignment 1

**Training-Testing Neural Net with MNIST Dataset using PyTorch**

The first assignment covers:

- Loading and inspecting the MNIST dataset
- Building a simple fully connected neural network
- Training with SGD and CrossEntropyLoss
- Evaluating training and test accuracy
- Visualizing training loss and accuracy
- Hyperparameter tuning for learning rate and momentum
- Architecture tuning for hidden layers and neurons

See [Assignment-1/README.md](Assignment-1/README.md) for the experiment setup and results.

## Assignment 2

**Training-Testing CNN with CIFAR-10 using PyTorch**

The second assignment covers CIFAR-10 loading and augmentation, CNN and max-pooling architectures, GPU training, hyperparameter tuning, validation-based model selection, and final held-out test evaluation. The strongest recorded **CNN** was ImageNet-pretrained EfficientNetV2-M with 99.08% validation accuracy after flip TTA and **98.59% final test accuracy** (0.0990 test loss). Results are from completed Colab runs; the scripts support reproducing the methodology, and pretrained weights are disclosed.

See [Assignment-2/README.md](Assignment-2/README.md) for source code, setup and experimental details.

## Environment

- Python
- PyTorch
- torchvision
- matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt
```
