import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3

WEIGHTS_PATH = "weights/bilstm.pth"

print("Using device:", DEVICE)


# -----------------------------
# SimpleBiLSTM
# Same architecture as models/bilstm.py
# -----------------------------
class SimpleBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=28,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        # (B, 1, 28, 28)
        x = x.squeeze(1)

        # (B, 28, 28)
        output, _ = self.lstm(x)

        # Last time step
        output = output[:, -1, :]

        return self.fc(output)


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
model = SimpleBiLSTM().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
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
print("BiLSTM training completed.")
print("Weights saved to:", WEIGHTS_PATH)