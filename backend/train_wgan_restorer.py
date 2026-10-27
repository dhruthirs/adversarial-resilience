"""Train WGAN-GP restoration for the shared MNIST classifier family."""

import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.gan_attack import generate_adversarial_image
from backend.mnist_models import load_mnist_classifiers
from backend.wgan_restoration import (
    RestorationCritic,
    WGANRestorer,
    gradient_penalty,
)
from backend.gan_attack import AttackGenerator, generate_adversarial_image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights")
OUTPUT = os.path.join(WEIGHTS, "wgan_restorer.pth")
EPOCHS = 10
BATCH_SIZE = 256
EPSILON = 0.30


def pgd_attack(model, images, labels, steps=40):
    original = images.detach()
    adversarial = original.clone()
    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = F.cross_entropy(model(adversarial), labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]
        adversarial = adversarial.detach() + (2 / 255) * gradient.sign()
        adversarial = (
            original + (adversarial - original).clamp(-EPSILON, EPSILON)
        ).clamp(0, 1)
    return adversarial.detach()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use .venv-gpu.")
    classifiers = load_mnist_classifiers(WEIGHTS, DEVICE)
    loader = DataLoader(
        datasets.MNIST(
            root=os.path.join(ROOT, "data"),
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    restorer = WGANRestorer().to(DEVICE)
    critic = RestorationCritic().to(DEVICE)
    g_optimizer = torch.optim.AdamW(restorer.parameters(), lr=1e-4)
    c_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-4)
    gan_generator = AttackGenerator(epsilon=EPSILON).to(DEVICE)
    gan_checkpoint = os.path.join(WEIGHTS, "gan_attack_all.pth")
    if os.path.exists(gan_checkpoint):
        gan_generator.load_state_dict(
            torch.load(gan_checkpoint, map_location=DEVICE, weights_only=False)
        )
        gan_generator.eval()
        for parameter in gan_generator.parameters():
            parameter.requires_grad_(False)
    total = EPOCHS * len(loader)
    completed = 0
    started = time.perf_counter()

    for epoch in range(EPOCHS):
        for batch, (clean, labels) in enumerate(loader, 1):
            clean, labels = clean.to(DEVICE), labels.to(DEVICE)
            attack_model = list(classifiers.values())[(batch - 1) % len(classifiers)]
            if batch % 3 == 0 and gan_generator is not None:
                attacked, _ = generate_adversarial_image(
                    gan_generator, clean, EPSILON
                )
            else:
                attacked = pgd_attack(attack_model, clean, labels)

            c_optimizer.zero_grad()
            with torch.no_grad():
                fake = restorer(attacked)
            c_loss = (
                critic(fake).mean()
                - critic(clean).mean()
                + 10.0 * gradient_penalty(critic, clean, fake, DEVICE)
            )
            c_loss.backward()
            c_optimizer.step()

            g_optimizer.zero_grad()
            restored = restorer(attacked)
            classifier_loss = torch.stack([
                F.cross_entropy(model(restored), labels)
                for model in classifiers.values()
            ]).mean()
            g_loss = (
                -critic(restored).mean()
                + 100.0 * F.mse_loss(restored, clean)
                + 0.1 * classifier_loss
            )
            g_loss.backward()
            g_optimizer.step()

            completed += 1
            progress = 100 * completed / total
            elapsed = time.perf_counter() - started
            eta = (total - completed) * elapsed / max(completed, 1)
            print(
                f"\rOverall: {progress:6.2f}% | Epoch {epoch + 1}/{EPOCHS} "
                f"Batch {batch}/{len(loader)} | Loss: {g_loss.item():.4f} "
                f"| ETA: {eta / 60:.1f} min",
                end="",
                flush=True,
            )
        print()
    os.makedirs(WEIGHTS, exist_ok=True)
    torch.save(restorer.state_dict(), OUTPUT)
    print(f"Saved WGAN restoration generator to {OUTPUT}")


if __name__ == "__main__":
    main()
