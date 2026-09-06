import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -------------------------
# CNN Model
# -------------------------

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                64 * 7 * 7,
                128
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = x.view(
            x.size(0),
            -1
        )

        return self.classifier(x)


# -------------------------
# Load trained CNN
# -------------------------

model = SimpleCNN().to(device)

model.load_state_dict(
    torch.load(
        "./weights/cnn.pth",
        map_location=device
    )
)

model.eval()


# -------------------------
# MNIST Test Dataset
# -------------------------

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# -------------------------
# Calculate Accuracy
# -------------------------

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


accuracy = 100 * correct / total


# -------------------------
# Results
# -------------------------

print()
print("===== CNN Clean Accuracy =====")

print(
    f"Correct predictions: "
    f"{correct}/{total}"
)

print(
    f"Clean Accuracy: "
    f"{accuracy:.2f}%"
)