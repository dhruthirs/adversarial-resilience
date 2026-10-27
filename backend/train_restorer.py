"""Train the genuine image restorer on adversarial/clean MNIST pairs."""

import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from vit_pytorch import ViT

# Allow `python backend\train_restorer.py` from the repository root.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from efficient_kan import KAN
from backend.restoration import WaveletSplineRestorer
from backend.gan_attack import AttackGenerator, generate_adversarial_image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 256
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "weights",
    "restorer.pth",
)


class MNISTKAN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kan = KAN(
            layers_hidden=[784, 64, 10],
            grid_size=3,
            spline_order=3,
        )

    def forward(self, image):
        return self.kan(image.reshape(image.size(0), -1).contiguous())


class SimpleANN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(784, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        )

    def forward(self, image):
        return self.net(image.reshape(image.size(0), -1))


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

    def forward(self, image):
        features = self.features(image)
        return self.classifier(features.reshape(image.size(0), -1))


class SimpleLSTM(torch.nn.Module):
    def __init__(self, bidirectional=False):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            28,
            128,
            num_layers=2,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = torch.nn.Linear(256 if bidirectional else 128, 10)

    def forward(self, image):
        output, _ = self.lstm(image.squeeze(1))
        return self.fc(output[:, -1, :])


class SimpleViT(ViT):
    def __init__(self):
        super().__init__(
            image_size=28,
            patch_size=7,
            num_classes=10,
            dim=64,
            depth=4,
            heads=4,
            mlp_dim=128,
            channels=1,
        )


ATTACK_EPSILON = 0.30


def fgsm_attack(model, images, labels, epsilon=ATTACK_EPSILON):
    adversarial = images.detach().requires_grad_(True)
    loss = F.cross_entropy(model(adversarial), labels)
    gradient = torch.autograd.grad(loss, adversarial)[0]
    return (images + epsilon * gradient.sign()).clamp(0.0, 1.0).detach()


def pgd_attack(
    model,
    images,
    labels,
    epsilon=ATTACK_EPSILON,
    alpha=2 / 255,
    steps=40,
):
    original = images.detach()
    adversarial = original.clone()

    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = F.cross_entropy(model(adversarial), labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]
        adversarial = adversarial.detach() + alpha * gradient.sign()
        perturbation = (adversarial - original).clamp(-epsilon, epsilon)
        adversarial = (original + perturbation).clamp(0.0, 1.0)

    return adversarial


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for restoration training. "
            "Use the .venv-gpu interpreter or install CUDA-enabled PyTorch."
        )

    dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    weights_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "weights"
    )
    classifiers = {
        "ANN": SimpleANN(),
        "CNN": SimpleCNN(),
        "LSTM": SimpleLSTM(),
        "BiLSTM": SimpleLSTM(bidirectional=True),
        "KAN": MNISTKAN(),
        "ViT": SimpleViT(),
    }
    for name, classifier in classifiers.items():
        checkpoint_path = os.path.join(weights_dir, f"{name.lower()}.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Missing {name} checkpoint: {checkpoint_path}"
            )
        classifier.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=DEVICE,
                weights_only=False,
            )
        )
        classifier.to(DEVICE).eval()
        if name in {"LSTM", "BiLSTM"}:
            classifier.train()
        for parameter in classifier.parameters():
            parameter.requires_grad_(False)

    restorer = WaveletSplineRestorer().to(DEVICE)
    optimizer = optim.AdamW(restorer.parameters(), lr=1e-3)
    gan_generator = AttackGenerator(epsilon=ATTACK_EPSILON).to(DEVICE)
    gan_checkpoint = os.path.join(weights_dir, "gan_attack_all.pth")
    if os.path.exists(gan_checkpoint):
        gan_generator.load_state_dict(
            torch.load(gan_checkpoint, map_location=DEVICE, weights_only=False)
        )
        gan_generator.eval()
        for parameter in gan_generator.parameters():
            parameter.requires_grad_(False)

    total_batches = EPOCHS * len(loader)
    completed_batches = 0
    started_at = time.perf_counter()

    for epoch in range(EPOCHS):
        restorer.train()
        running_loss = 0.0
        for batch_index, (clean, labels) in enumerate(loader, start=1):
            clean, labels = clean.to(DEVICE), labels.to(DEVICE)
            classifier_name = list(classifiers)[
                (batch_index - 1) % len(classifiers)
            ]
            attack_classifier = classifiers[classifier_name]

            if batch_index % 3 == 1:
                adversarial = fgsm_attack(attack_classifier, clean, labels)
            elif batch_index % 3 == 2:
                adversarial = pgd_attack(attack_classifier, clean, labels)
            elif gan_generator is not None:
                adversarial, _ = generate_adversarial_image(
                    gan_generator, clean, ATTACK_EPSILON
                )
            else:
                adversarial = pgd_attack(attack_classifier, clean, labels)
            restored = restorer(adversarial)
            reconstruction = F.mse_loss(restored, clean)
            classification = torch.stack(
                [
                    F.cross_entropy(classifier(restored), labels)
                    for classifier in classifiers.values()
                ]
            ).mean()
            loss = reconstruction + 0.1 * classification

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            completed_batches += 1

            elapsed = time.perf_counter() - started_at
            rate = completed_batches / max(elapsed, 1e-6)
            remaining = (total_batches - completed_batches) / max(rate, 1e-6)
            progress = 100.0 * completed_batches / total_batches
            epoch_progress = 100.0 * batch_index / len(loader)
            print(
                f"\rOverall: {progress:6.2f}% | "
                f"Epoch {epoch + 1}/{EPOCHS}: {epoch_progress:6.2f}% | "
                f"Batch {batch_index}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"ETA: {remaining / 60:.1f} min",
                end="",
                flush=True,
            )

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS} complete: "
            f"loss={running_loss / len(loader):.4f}"
        )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(restorer.state_dict(), OUTPUT_PATH)
    print(f"Saved trained restorer to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
