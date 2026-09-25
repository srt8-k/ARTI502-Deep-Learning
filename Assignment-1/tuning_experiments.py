import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# --------------------------------------------------
# Task 8 - Hyperparameter and Architecture Tuning
# --------------------------------------------------

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# --------------------------------------------------
# Baseline Network: 784 -> 128 -> 10
# --------------------------------------------------

class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# --------------------------------------------------
# Shared Training Function
# --------------------------------------------------

def train_model(
    model,
    train_loader,
    epochs,
    learning_rate,
    momentum
):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )

    final_loss = 0
    final_accuracy = 0

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        final_loss = running_loss / len(train_loader)
        final_accuracy = 100 * correct / total

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {final_loss:.4f} "
            f"- Train Accuracy: {final_accuracy:.2f}%"
        )

    return final_loss, final_accuracy


# --------------------------------------------------
# Shared Testing Function
# --------------------------------------------------

def test_model(model, test_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy


# --------------------------------------------------
# Hyperparameter Tuning
# --------------------------------------------------

def run_experiment(
    experiment_name,
    learning_rate,
    momentum,
    epochs=5
):
    print("\n" + "=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Momentum: {momentum}")
    print("=" * 60)

    torch.manual_seed(42)

    generator = torch.Generator().manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        generator=generator
    )

    model = SimpleNetwork()

    train_loss, train_accuracy = train_model(
        model,
        train_loader,
        epochs,
        learning_rate,
        momentum
    )

    test_accuracy = test_model(
        model,
        test_loader
    )

    print(f"Final Training Loss: {train_loss:.4f}")
    print(f"Final Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    return {
        "name": experiment_name,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }


experiments = [
    {
        "name": "Baseline",
        "learning_rate": 0.01,
        "momentum": 0.9
    },
    {
        "name": "Low Learning Rate",
        "learning_rate": 0.001,
        "momentum": 0.9
    },
    {
        "name": "High Learning Rate",
        "learning_rate": 0.05,
        "momentum": 0.9
    },
    {
        "name": "Lower Momentum",
        "learning_rate": 0.01,
        "momentum": 0.5
    }
]


results = []

for experiment in experiments:
    result = run_experiment(
        experiment_name=experiment["name"],
        learning_rate=experiment["learning_rate"],
        momentum=experiment["momentum"],
        epochs=5
    )

    results.append(result)


print("\n")
print("=" * 95)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 95)

print(
    f"{'Experiment':<22}"
    f"{'LR':<12}"
    f"{'Momentum':<12}"
    f"{'Loss':<12}"
    f"{'Train Acc':<15}"
    f"{'Test Acc':<15}"
)

print("-" * 95)

for result in results:
    print(
        f"{result['name']:<22}"
        f"{result['learning_rate']:<12}"
        f"{result['momentum']:<12}"
        f"{result['loss']:<12.4f}"
        f"{result['train_accuracy']:<15.2f}"
        f"{result['test_accuracy']:<15.2f}"
    )


# --------------------------------------------------
# Architecture Tuning
# --------------------------------------------------

class FlexibleNetwork(nn.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        layers = []
        input_size = 784

        for hidden_size in hidden_layers:
            layers.append(
                nn.Linear(input_size, hidden_size)
            )

            layers.append(
                nn.ReLU()
            )

            input_size = hidden_size

        layers.append(
            nn.Linear(input_size, 10)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.network(x)
        return x


def run_architecture_experiment(
    experiment_name,
    hidden_layers,
    epochs=5
):
    print("\n" + "=" * 60)
    print(f"Architecture Experiment: {experiment_name}")
    print(f"Hidden Layers: {hidden_layers}")
    print("=" * 60)

    torch.manual_seed(42)

    generator = torch.Generator().manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        generator=generator
    )

    model = FlexibleNetwork(hidden_layers)

    train_loss, train_accuracy = train_model(
        model,
        train_loader,
        epochs=epochs,
        learning_rate=0.01,
        momentum=0.9
    )

    test_accuracy = test_model(
        model,
        test_loader
    )

    print(f"Final Training Loss: {train_loss:.4f}")
    print(f"Final Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    return {
        "name": experiment_name,
        "architecture": str(hidden_layers),
        "loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }


architecture_experiments = [
    {
        "name": "64 Neurons",
        "hidden_layers": [64]
    },
    {
        "name": "128 Neurons",
        "hidden_layers": [128]
    },
    {
        "name": "256 Neurons",
        "hidden_layers": [256]
    },
    {
        "name": "Two Hidden Layers",
        "hidden_layers": [128, 64]
    }
]


architecture_results = []

for experiment in architecture_experiments:
    result = run_architecture_experiment(
        experiment_name=experiment["name"],
        hidden_layers=experiment["hidden_layers"],
        epochs=5
    )

    architecture_results.append(result)


print("\n")
print("=" * 90)
print("ARCHITECTURE TUNING RESULTS")
print("=" * 90)

print(
    f"{'Experiment':<22}"
    f"{'Architecture':<20}"
    f"{'Loss':<12}"
    f"{'Train Acc':<15}"
    f"{'Test Acc':<15}"
)

print("-" * 90)

for result in architecture_results:
    print(
        f"{result['name']:<22}"
        f"{result['architecture']:<20}"
        f"{result['loss']:<12.4f}"
        f"{result['train_accuracy']:<15.2f}"
        f"{result['test_accuracy']:<15.2f}"
    )
