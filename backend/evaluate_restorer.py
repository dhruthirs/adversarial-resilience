"""Evaluate the trained restorer independently with every MNIST classifier."""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from vit_pytorch import ViT

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from efficient_kan import KAN
from backend.gan_attack import AttackGenerator, generate_adversarial_image
from backend.restoration import WaveletSplineRestorer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights")
EPSILON = 0.25
BATCH_SIZE = 256


class SimpleANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256),
            nn.ReLU(), nn.Linear(256, 10)
        )

    def forward(self, image):
        return self.net(image.reshape(image.size(0), -1))


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
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
        self.kan = KAN(layers_hidden=[784, 64, 10], grid_size=3, spline_order=3)

    def forward(self, image):
        return self.kan(image.reshape(image.size(0), -1).contiguous())


class SimpleViT(ViT):
    def __init__(self):
        super().__init__(
            image_size=28, patch_size=7, num_classes=10, dim=64,
            depth=4, heads=4, mlp_dim=128, channels=1
        )


def load_models():
    models = {
        "ANN": SimpleANN(),
        "CNN": SimpleCNN(),
        "LSTM": SimpleLSTM(),
        "BiLSTM": SimpleLSTM(bidirectional=True),
        "KAN": MNISTKAN(),
        "ViT": SimpleViT(),
    }
    for name, model in models.items():
        path = os.path.join(WEIGHTS, f"{name.lower()}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} checkpoint: {path}")
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
        model.to(DEVICE).eval()
        if name in {"LSTM", "BiLSTM"}:
            model.train()
    return models


def adaptive_attack(model, restorer, images, labels, steps):
    adversarial = images.detach().clone()
    alpha = EPSILON if steps == 1 else 2 / 255
    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = F.cross_entropy(model(restorer(adversarial)), labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]
        adversarial = (
            adversarial.detach() + alpha * gradient.sign() - images
        ).clamp(-EPSILON, EPSILON).add(images).clamp(0, 1)
    return adversarial.detach()


def main():
    models = load_models()
    restorer = WaveletSplineRestorer().to(DEVICE)
    checkpoint = os.path.join(WEIGHTS, "restorer.pth")
    restorer.load_state_dict(
        torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    )
    restorer.eval()
    gan_generator = AttackGenerator(epsilon=EPSILON).to(DEVICE)
    gan_checkpoint = os.path.join(WEIGHTS, "gan_attack_all.pth")
    if not os.path.exists(gan_checkpoint):
        raise FileNotFoundError(
            f"Missing multi-classifier GAN checkpoint: {gan_checkpoint}. "
            "Run backend\\train_gan_all.py first."
        )
    gan_generator.load_state_dict(
        torch.load(gan_checkpoint, map_location=DEVICE, weights_only=False)
    )
    gan_generator.eval()
    loader = DataLoader(
        torchvision.datasets.MNIST(
            root=os.path.join(ROOT, "data"),
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    totals = {
        name: {
            "clean": 0,
            "fgsm": 0,
            "pgd": 0,
            "restored": 0,
            "recovered": 0,
            "fgsm_success": 0,
            "pgd_success": 0,
            "gan": 0,
            "gan_success": 0,
        }
        for name in models
    }
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        for name, model in models.items():
            with torch.no_grad():
                clean = model(images).argmax(1)
                clean_correct = clean.eq(labels)
            fgsm = adaptive_attack(model, restorer, images, labels, 1)
            pgd = adaptive_attack(model, restorer, images, labels, 10)
            gan, _ = generate_adversarial_image(
                gan_generator, images, EPSILON
            )
            with torch.no_grad():
                fgsm_prediction = model(restorer(fgsm)).argmax(1)
                pgd_prediction = model(restorer(pgd)).argmax(1)
                gan_prediction = model(restorer(gan)).argmax(1)
                restored_prediction = pgd_prediction
                fgsm_correct = fgsm_prediction.eq(labels)
                pgd_correct = pgd_prediction.eq(labels)
                restored_correct = restored_prediction.eq(labels)
                totals[name]["clean"] += clean_correct.sum().item()
                totals[name]["fgsm"] += fgsm_correct.sum().item()
                totals[name]["pgd"] += pgd_correct.sum().item()
                totals[name]["restored"] += restored_correct.sum().item()
                gan_correct = gan_prediction.eq(labels)
                totals[name]["gan"] += gan_correct.sum().item()
                totals[name]["recovered"] += (
                    restored_prediction.eq(clean).sum().item()
                )
                totals[name]["fgsm_success"] += (
                    (clean_correct & ~fgsm_correct).sum().item()
                )
                totals[name]["pgd_success"] += (
                    (clean_correct & ~pgd_correct).sum().item()
                )
                totals[name]["gan_success"] += (
                    (clean_correct & ~gan_correct).sum().item()
                )
        total += labels.numel()
        print(f"Processed {total}/10000 ({100 * total / 10000:.1f}%)")

    print(
        "\nModel      Clean   FGSM    FGSM ASR  FGSM Drop  "
        "PGD     PGD ASR  PGD Drop  GAN     GAN ASR  GAN Drop  "
        "Restored  Recovery"
    )
    for name, values in totals.items():
        clean_accuracy = 100 * values["clean"] / total
        fgsm_accuracy = 100 * values["fgsm"] / total
        pgd_accuracy = 100 * values["pgd"] / total
        restored_accuracy = 100 * values["restored"] / total
        fgsm_asr = (
            100 * values["fgsm_success"] / values["clean"]
            if values["clean"]
            else 0.0
        )
        pgd_asr = (
            100 * values["pgd_success"] / values["clean"]
            if values["clean"]
            else 0.0
        )
        fgsm_drop = clean_accuracy - fgsm_accuracy
        pgd_drop = clean_accuracy - pgd_accuracy
        gan_accuracy = 100 * values["gan"] / total
        gan_asr = (
            100 * values["gan_success"] / values["clean"]
            if values["clean"]
            else 0.0
        )
        gan_drop = clean_accuracy - gan_accuracy
        recovery = 100 * values["recovered"] / total
        print(
            f"{name:<9} "
            f"{clean_accuracy:6.2f}% "
            f"{fgsm_accuracy:6.2f}% "
            f"{fgsm_asr:8.2f}% "
            f"{fgsm_drop:9.2f}pp "
            f"{pgd_accuracy:6.2f}% "
            f"{pgd_asr:7.2f}% "
            f"{pgd_drop:8.2f}pp "
            f"{gan_accuracy:6.2f}% "
            f"{gan_asr:7.2f}% "
            f"{gan_drop:8.2f}pp "
            f"{restored_accuracy:8.2f}% "
            f"{recovery:8.2f}%"
        )


if __name__ == "__main__":
    main()
