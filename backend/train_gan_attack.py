import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from backend.gan_attack import (
    AttackGenerator,
    WGANCritic,
    generate_adversarial_image,
    generator_loss,
    critic_loss,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 5

LEARNING_RATE_GENERATOR = 1e-4
LEARNING_RATE_CRITIC = 1e-4

CRITIC_STEPS = 1

EPSILON = 0.15

LAMBDA_ATTACK = 10.0
LAMBDA_PERTURBATION = 0.5
LAMBDA_WGAN = 0.01

LAMBDA_GP = 10.0

CNN_WEIGHTS = "weights/cnn.pth"
GAN_WEIGHTS = "weights/gan_attack_cnn_v3.pth"

# ============================================================
# Device
# ============================================================

print("=" * 60)
print("GAN ATTACK TRAINING")
print("=" * 60)

print(f"Using device: {DEVICE}")


# ============================================================
# Load trained CNN
# ============================================================

print("\nLoading trained CNN...")

cnn = SimpleCNN().to(DEVICE)

cnn.load_state_dict(
    torch.load(
        CNN_WEIGHTS,
        map_location=DEVICE,
        weights_only=True
    )
)

cnn.eval()

# Freeze CNN parameters.
# We still need gradients with respect to the adversarial image,
# so we DO NOT use torch.no_grad() during the attack loss.
for parameter in cnn.parameters():
    parameter.requires_grad = False

print("CNN loaded successfully.")
print("CNN parameters frozen.")


# ============================================================
# Load MNIST
# ============================================================

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print(f"\nMNIST training samples: {len(train_dataset)}")
print(f"Batch size: {BATCH_SIZE}")


# ============================================================
# Create Generator and Critic
# ============================================================

generator = AttackGenerator(
    epsilon=EPSILON
).to(DEVICE)

critic = WGANCritic().to(DEVICE)


# ============================================================
# Optimizers
# ============================================================

generator_optimizer = torch.optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE_GENERATOR,
    betas=(0.5, 0.9)
)

critic_optimizer = torch.optim.Adam(
    critic.parameters(),
    lr=LEARNING_RATE_CRITIC,
    betas=(0.5, 0.9)
)


# ============================================================
# Helper functions
# ============================================================

def set_requires_grad(model, value):
    """
    Enable/disable gradients for all parameters of a model.
    """

    for parameter in model.parameters():
        parameter.requires_grad = value


def calculate_attack_success(
    classifier,
    clean_images,
    labels,
    adversarial_images
):
    """
    Calculate attack success only among images that were
    initially classified correctly by the clean classifier.

    ASR =
        successful attacks
        -----------------------------
        initially correctly classified
    """

    with torch.no_grad():
        clean_predictions = classifier(
            clean_images
        ).argmax(dim=1)

        adversarial_predictions = classifier(
            adversarial_images
        ).argmax(dim=1)

        initially_correct = (
            clean_predictions == labels
        )

        successful_attacks = (
            initially_correct
            & (adversarial_predictions != labels)
        )

        correct_count = initially_correct.sum().item()
        success_count = successful_attacks.sum().item()

    if correct_count == 0:
        return 0.0, 0, 0

    asr = (
        success_count
        / correct_count
        * 100
    )

    return asr, success_count, correct_count


# ============================================================
# Training
# ============================================================

print("\nStarting GAN training...")
print("-" * 60)

for epoch in range(1, EPOCHS + 1):

    generator.train()
    critic.train()

    total_generator_loss = 0.0
    total_critic_loss = 0.0

    total_attack_loss = 0.0
    total_wgan_loss = 0.0
    total_perturbation_loss = 0.0

    total_asr = 0.0
    total_batches = 0

    total_perturbation = 0.0

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # ====================================================
        # Train Critic
        # ====================================================

        for _ in range(CRITIC_STEPS):

            critic_optimizer.zero_grad()

            # Generate adversarial images.
            # Generator is not updated during critic training.
            with torch.no_grad():
                adversarial_images, _ = (
                    generate_adversarial_image(
                        generator,
                        images,
                        EPSILON
                    )
                )

            (
                c_loss,
                real_score,
                fake_score,
                gp
            ) = critic_loss(
                critic,
                images,
                adversarial_images,
                DEVICE,
                LAMBDA_GP
            )

            c_loss.backward()

            critic_optimizer.step()

        # ====================================================
        # Train Generator
        # ====================================================

        generator_optimizer.zero_grad()

        # We don't need critic parameter gradients while
        # optimizing the generator.
        set_requires_grad(critic, False)

        adversarial_images, delta = (
            generate_adversarial_image(
                generator,
                images,
                EPSILON
            )
        )

        (
            g_loss,
            wgan_loss,
            attack_loss,
            perturbation_loss
        ) = generator_loss(
            critic,
            cnn,
            images,
            labels,
            adversarial_images,
            delta,
            lambda_attack=LAMBDA_ATTACK,
            lambda_perturbation=LAMBDA_PERTURBATION,
            lambda_wgan=LAMBDA_WGAN
        )

        g_loss.backward()

        generator_optimizer.step()

        set_requires_grad(critic, True)

        # ====================================================
        # Statistics
        # ====================================================

        asr, success_count, correct_count = (
            calculate_attack_success(
                cnn,
                images,
                labels,
                adversarial_images
            )
        )

        mean_perturbation = (
            delta.abs().mean().item()
        )

        total_generator_loss += g_loss.item()
        total_critic_loss += c_loss.item()

        total_attack_loss += attack_loss.item()
        total_wgan_loss += wgan_loss.item()
        total_perturbation_loss += (
            perturbation_loss.item()
        )

        total_asr += asr
        total_perturbation += mean_perturbation

        total_batches += 1

        # Print progress every 100 batches
        if (batch_index + 1) % 100 == 0:

            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Batch [{batch_index + 1}/{len(train_loader)}] "
                f"G Loss: {g_loss.item():.4f} "
                f"C Loss: {c_loss.item():.4f} "
                f"ASR: {asr:.2f}%"
            )

    # ========================================================
    # Epoch statistics
    # ========================================================

    avg_g_loss = (
        total_generator_loss
        / total_batches
    )

    avg_c_loss = (
        total_critic_loss
        / total_batches
    )

    avg_attack_loss = (
        total_attack_loss
        / total_batches
    )

    avg_wgan_loss = (
        total_wgan_loss
        / total_batches
    )

    avg_perturbation_loss = (
        total_perturbation_loss
        / total_batches
    )

    avg_asr = (
        total_asr
        / total_batches
    )

    avg_perturbation = (
        total_perturbation
        / total_batches
    )

    print("\n" + "=" * 60)
    print(f"Epoch {epoch}/{EPOCHS} completed")
    print("=" * 60)

    print(f"Average Generator Loss : {avg_g_loss:.4f}")
    print(f"Average Critic Loss    : {avg_c_loss:.4f}")
    print(f"Attack Loss            : {avg_attack_loss:.4f}")
    print(f"WGAN Loss              : {avg_wgan_loss:.4f}")
    print(
        f"Perturbation Loss      : "
        f"{avg_perturbation_loss:.4f}"
    )
    print(f"Average batch ASR      : {avg_asr:.2f}%")
    print(
        f"Mean |perturbation|    : "
        f"{avg_perturbation:.6f}"
    )


# ============================================================
# Save Generator
# ============================================================

os.makedirs(
    os.path.dirname(GAN_WEIGHTS),
    exist_ok=True
)

torch.save(
    generator.state_dict(),
    GAN_WEIGHTS
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"GAN generator saved to:")
print(GAN_WEIGHTS)