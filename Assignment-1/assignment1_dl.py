import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# --------------------------------------------------
# Task 1 - Load PyTorch
# --------------------------------------------------

print("PyTorch version:", torch.__version__)
print("PyTorch loaded successfully!")


# --------------------------------------------------
# Task 2 - Load MNIST
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

print("MNIST training dataset loaded successfully!")
print("MNIST test dataset loaded successfully!")


# --------------------------------------------------
# Task 3 - Inspect Dataset
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

images, labels = next(iter(train_loader))

print("Images tensor size:", images.shape)
print("Labels tensor size:", labels.shape)

plt.imshow(images[0].squeeze(), cmap="gray")
plt.title(f"Label: {labels[0].item()}")
plt.axis("off")
plt.show()


# --------------------------------------------------
# Task 4 - Simple Neural Network
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


model = SimpleNetwork()

print(model)

outputs = model(images)
print("Network output size:", outputs.shape)


# --------------------------------------------------
# Task 5 - Training Function
# --------------------------------------------------

def train_network(
    net,
    dataloader,
    epochs=5,
    learning_rate=0.01,
    momentum=0.9
):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=momentum
    )

    batch_losses = []
    epoch_losses = []
    epoch_accuracies = []

    net.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_number, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            outputs = net(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            batch_losses.append(loss.item())
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            current_accuracy = 100 * correct / total

            if batch_number % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{batch_number}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Accuracy: {current_accuracy:.2f}%"
                )

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total

        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        print(
            f"End of Epoch {epoch + 1}: "
            f"Loss = {epoch_loss:.4f}, "
            f"Accuracy = {epoch_accuracy:.2f}%"
        )

        print("-" * 60)

    return batch_losses, epoch_losses, epoch_accuracies


# --------------------------------------------------
# Task 6 - Train Network and Plot Results
# --------------------------------------------------

trained_model = SimpleNetwork()

batch_losses, epoch_losses, epoch_accuracies = train_network(
    trained_model,
    train_loader,
    epochs=5,
    learning_rate=0.01,
    momentum=0.9
)

epochs_range = range(1, len(epoch_losses) + 1)

plt.figure()
plt.plot(epochs_range, epoch_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.grid()
plt.show()

plt.figure()
plt.plot(epochs_range, epoch_accuracies, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy per Epoch")
plt.grid()
plt.show()

plt.figure()
plt.plot(batch_losses)
plt.xlabel("Mini-Batch")
plt.ylabel("Loss")
plt.title("Training Loss per Mini-Batch")
plt.grid()
plt.show()


# --------------------------------------------------
# Task 7 - Test Network
# --------------------------------------------------

def test_network(net, dataloader):
    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total

    print(f"Test Accuracy: {test_accuracy:.2f}%")

    return test_accuracy


test_accuracy = test_network(
    trained_model,
    test_loader
)

test_images, test_labels = next(iter(test_loader))

trained_model.eval()

with torch.no_grad():
    test_outputs = trained_model(test_images)

_, test_predictions = torch.max(test_outputs, 1)

plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i].squeeze(), cmap="gray")
    plt.title(
        f"True: {test_labels[i].item()}\n"
        f"Pred: {test_predictions[i].item()}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()
