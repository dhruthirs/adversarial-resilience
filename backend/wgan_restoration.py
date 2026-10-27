"""WGAN-GP restoration generator and critic for MNIST images."""

import torch
import torch.nn as nn

from backend.restoration import WaveletSplineRestorer


class WGANRestorer(nn.Module):
    """Learned generator used to reconstruct clean images from attacks."""

    def __init__(self):
        super().__init__()
        self.generator = WaveletSplineRestorer(channels=1)

    def forward(self, image):
        return self.generator(image)


class RestorationCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(128 * 4 * 4, 1)

    def forward(self, image):
        return self.fc(self.features(image).flatten(1))


def gradient_penalty(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = critic(mixed)
    gradients = torch.autograd.grad(
        score,
        mixed,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        only_inputs=True,
    )[0]
    return (gradients.flatten(1).norm(2, dim=1).sub(1).square()).mean()
