"""Shared MNIST classifier definitions used by attacks and evaluation."""

import os

import torch
import torch.nn as nn
from vit_pytorch import ViT

from efficient_kan import KAN


class SimpleANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, image):
        return self.net(image.reshape(image.size(0), -1))


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, image):
        return self.classifier(
            self.features(image).reshape(image.size(0), -1)
        )


class SimpleLSTM(nn.Module):
    def __init__(self, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            28, 128, num_layers=2, batch_first=True,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(256 if bidirectional else 128, 10)

    def forward(self, image):
        output, _ = self.lstm(image.squeeze(1))
        return self.fc(output[:, -1, :])


class MNISTKAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.kan = KAN(
            layers_hidden=[784, 64, 10],
            grid_size=3,
            spline_order=3,
        )

    def forward(self, image):
        return self.kan(image.reshape(image.size(0), -1).contiguous())


class SimpleViT(ViT):
    def __init__(self):
        super().__init__(
            image_size=28, patch_size=7, num_classes=10, dim=64,
            depth=4, heads=4, mlp_dim=128, channels=1
        )


MODEL_CLASSES = {
    "ANN": SimpleANN,
    "CNN": SimpleCNN,
    "LSTM": SimpleLSTM,
    "BiLSTM": lambda: SimpleLSTM(bidirectional=True),
    "KAN": MNISTKAN,
    "ViT": SimpleViT,
}


def load_mnist_classifiers(weights_dir, device):
    classifiers = {}
    for name, model_factory in MODEL_CLASSES.items():
        model = model_factory()
        path = os.path.join(weights_dir, f"{name.lower()}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} checkpoint: {path}")
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=False)
        )
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if name in {"LSTM", "BiLSTM"}:
            model.train()
        classifiers[name] = model
    return classifiers
