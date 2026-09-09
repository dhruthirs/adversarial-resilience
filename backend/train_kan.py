import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from efficient_kan import KAN


# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001

WEIGHTS_PATH = "weights/kan.pth"

print("Using device:", DEVICE)


# -----------------------------
# MNIST KAN
# Same architecture as models/kan.py
# -----------------------------
class MNISTKAN(nn.Module):

    def __init__(self):
        super().__init__()

        self.kan = KAN(
            layers_hidden=[784, 64, 10],
            grid_size=3,
            spline_order=3
        )

    def forward(self, x):

        x = x.reshape(x.size(0), -1).contiguous()

        return self.kan(x)


# -----------------------------
# MNIST Dataset
# -----------------------------
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# Model
# -----------------------------
model = MNISTKAN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    foreach=False
)


# -----------------------------
# Training
# -----------------------------
for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total


    # -----------------------------
    # Test
    # -----------------------------
    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            test_correct += (
                predictions == labels
            ).sum().item()

            test_total += labels.size(0)

    test_accuracy = 100 * test_correct / test_total


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )


# -----------------------------
# Save weights
# -----------------------------
os.makedirs("weights", exist_ok=True)

torch.save(
    model.state_dict(),
    WEIGHTS_PATH
)

print()
print("KAN training completed.")
print("Weights saved to:", WEIGHTS_PATH)