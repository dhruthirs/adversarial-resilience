import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Generator
# ============================================================

class AttackGenerator(nn.Module):
    """
    Generates an adversarial perturbation from a clean image.

    Input:
        x -> clean MNIST image [B, 1, 28, 28]

    Output:
        delta -> adversarial perturbation [B, 1, 28, 28]
    """

    def __init__(self, epsilon=0.25):
        super().__init__()

        self.epsilon = epsilon

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.encoder(x)

        raw_delta = self.decoder(features)

        # Restrict perturbation to [-epsilon, epsilon]
        delta = self.epsilon * raw_delta

        return delta


# ============================================================
# WGAN-GP Critic
# ============================================================

class WGANCritic(nn.Module):
    """
    WGAN-GP critic.

    Receives an image and produces a scalar score.
    Higher score should correspond to images that look
    more like the real clean-image distribution.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Linear(128 * 4 * 4, 1)

    def forward(self, x):
        features = self.features(x)

        features = features.view(features.size(0), -1)

        score = self.fc(features)

        return score


# ============================================================
# Generate adversarial image
# ============================================================

def generate_adversarial_image(generator, images, epsilon=0.20):
    delta = generator(images)

    # Scale the perturbation from the GAN's training epsilon
    # to the desired inference epsilon.
    scale = epsilon / generator.epsilon
    delta = delta * scale

    delta = torch.clamp(delta, -epsilon, epsilon)

    adversarial_images = torch.clamp(
        images + delta,
        0.0,
        1.0
    )

    return adversarial_images, delta


# ============================================================
# WGAN-GP Gradient Penalty
# ============================================================

def gradient_penalty(critic, real_images, fake_images, device):
    """
    Computes the gradient penalty used by WGAN-GP.
    """

    batch_size = real_images.size(0)

    alpha = torch.rand(
        batch_size,
        1,
        1,
        1,
        device=device
    )

    interpolated = (
        alpha * real_images
        + (1 - alpha) * fake_images
    )

    interpolated.requires_grad_(True)

    critic_score = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=critic_score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)

    gradient_norm = gradients.norm(2, dim=1)

    penalty = ((gradient_norm - 1) ** 2).mean()

    return penalty


# ============================================================
# Generator Loss
# ============================================================

def generator_loss(
    critic,
    classifier,
    clean_images,
    labels,
    adversarial_images,
    delta,
    lambda_attack=5.0,
    lambda_perturbation=1.0,
    lambda_wgan=0.05
):
    logits = classifier(adversarial_images)

    # Score of the correct class
    true_class_score = logits.gather(1, labels.unsqueeze(1)).squeeze(1)

    wrong_class_logits = logits.clone()
    wrong_class_logits.scatter_(1, labels.unsqueeze(1), float("-inf"))

    strongest_wrong_score = wrong_class_logits.max(dim=1).values

    margin = true_class_score - strongest_wrong_score

    attack_loss = F.relu(margin).mean()

    # Penalize perturbation
    perturbation_loss = torch.mean(
        torch.abs(delta)
    )

    # WGAN generator objective
    wgan_loss = -critic(
        adversarial_images
    ).mean()

    total_loss = (
        lambda_attack * attack_loss
        + lambda_perturbation * perturbation_loss
        + lambda_wgan * wgan_loss
    )

    return (
        total_loss,
        wgan_loss,
        attack_loss,
        perturbation_loss
    )


# ============================================================
# Critic Loss
# ============================================================

def critic_loss(
    critic,
    real_images,
    fake_images,
    device,
    lambda_gp=10.0
):
    """
    WGAN-GP critic loss.

    Critic tries to give:
        high score -> real clean images
        low score  -> generated adversarial images
    """

    real_score = critic(real_images).mean()

    fake_score = critic(
        fake_images.detach()
    ).mean()

    gp = gradient_penalty(
        critic,
        real_images,
        fake_images.detach(),
        device
    )

    loss = (
        fake_score
        - real_score
        + lambda_gp * gp
    )

    return (
        loss,
        real_score,
        fake_score,
        gp
    )