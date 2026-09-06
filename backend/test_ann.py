import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# -------------------------
# ANN Model
# -------------------------

class SimpleANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# -------------------------
# Load trained model
# -------------------------

model = SimpleANN().to(device)

model.load_state_dict(
    torch.load(
        "./weights/ann.pth",
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

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()


accuracy = 100 * correct / total


print()
print("===== ANN Clean Accuracy =====")
print(f"Correct predictions: {correct}/{total}")
print(f"Clean Accuracy: {accuracy:.2f}%")