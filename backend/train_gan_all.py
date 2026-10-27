"""Train one GAN generator against every MNIST classifier."""

import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.gan_attack import (
    AttackGenerator,
    WGANCritic,
    critic_loss,
    generate_adversarial_image,
    generator_loss,
)
from backend.mnist_models import load_mnist_classifiers


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights")
EPSILON = 0.30
EPOCHS = 8
BATCH_SIZE = 128


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
        num_workers=0,
    )
    generator = AttackGenerator(epsilon=EPSILON).to(DEVICE)
    critic = WGANCritic().to(DEVICE)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=1e-4, betas=(0.5, 0.9)
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=1e-4, betas=(0.5, 0.9)
    )
    total_batches = EPOCHS * len(loader)
    completed = 0
    started = time.perf_counter()

    for epoch in range(EPOCHS):
        generator.train()
        critic.train()
        for batch, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            critic_optimizer.zero_grad()
            with torch.no_grad():
                fake, _ = generate_adversarial_image(
                    generator, images, EPSILON
                )
            c_loss, _, _, _ = critic_loss(
                critic, images, fake, DEVICE, lambda_gp=10.0
            )
            c_loss.backward()
            critic_optimizer.step()

            generator_optimizer.zero_grad()
            fake, delta = generate_adversarial_image(
                generator, images, EPSILON
            )
            losses = [
                generator_loss(
                    critic,
                    classifier,
                    images,
                    labels,
                    fake,
                    delta,
                    lambda_attack=25.0,
                    lambda_perturbation=0.15,
                    lambda_wgan=0.01,
                )[0]
                for classifier in classifiers.values()
            ]
            loss = torch.stack(losses).mean()
            loss.backward()
            generator_optimizer.step()

            completed += 1
            progress = 100 * completed / total_batches
            elapsed = time.perf_counter() - started
            eta = (total_batches - completed) * elapsed / completed
            print(
                f"\rOverall: {progress:6.2f}% | Epoch {epoch + 1}/{EPOCHS} "
                f"Batch {batch}/{len(loader)} | Loss: {loss.item():.4f} "
                f"| ETA: {eta / 60:.1f} min",
                end="",
                flush=True,
            )
        print()

    output = os.path.join(WEIGHTS, "gan_attack_all.pth")
    torch.save(generator.state_dict(), output)
    print(f"Saved multi-classifier GAN generator to {output}")


if __name__ == "__main__":
    main()
