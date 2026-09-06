import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# -------------------------
# Device
# -------------------------

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
# PGD Attack
# -------------------------

def pgd_attack(model, images, labels, epsilon, alpha, steps):

    original_images = images.clone().detach()

    perturbed_images = images.clone().detach()

    for _ in range(steps):

        perturbed_images.requires_grad = True

        outputs = model(perturbed_images)

        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()

        loss.backward()

        data_grad = perturbed_images.grad.data

        # Gradient ascent
        perturbed_images = (
            perturbed_images
            + alpha * data_grad.sign()
        )

        # Keep perturbation within epsilon
        perturbation = (
            perturbed_images - original_images
        )

        perturbation = torch.clamp(
            perturbation,
            -epsilon,
            epsilon
        )

        perturbed_images = (
            original_images + perturbation
        )

        # Keep pixel values between 0 and 1
        perturbed_images = torch.clamp(
            perturbed_images,
            0,
            1
        ).detach()

    return perturbed_images


# -------------------------
# PGD Parameters
# -------------------------

epsilon = 0.25
alpha = 2 / 255
steps = 40


# -------------------------
# Evaluation
# -------------------------

clean_correct = 0
adversarial_correct = 0

successful_attacks = 0
eligible_samples = 0

total = 0


print()
print("===== PGD Attack =====")
print("Epsilon:", epsilon)
print("Alpha:", alpha)
print("Steps:", steps)


for images, labels in test_loader:

    images = images.to(device)
    labels = labels.to(device)

    # -------------------------
    # Clean prediction
    # -------------------------

    with torch.no_grad():
        clean_outputs = model(images)

    clean_predictions = clean_outputs.argmax(dim=1)

    clean_correct += (
        clean_predictions == labels
    ).sum().item()


    # -------------------------
    # Generate PGD images
    # -------------------------

    adversarial_images = pgd_attack(
        model,
        images,
        labels,
        epsilon,
        alpha,
        steps
    )


    # -------------------------
    # Adversarial prediction
    # -------------------------

    with torch.no_grad():
        adversarial_outputs = model(
            adversarial_images
        )

    adversarial_predictions = (
        adversarial_outputs.argmax(dim=1)
    )

    adversarial_correct += (
        adversarial_predictions == labels
    ).sum().item()


    # -------------------------
    # Calculate ASR
    # -------------------------

    clean_correct_mask = (
        clean_predictions == labels
    )

    successful_attack_mask = (
        clean_correct_mask
        & (adversarial_predictions != labels)
    )

    eligible_samples += (
        clean_correct_mask.sum().item()
    )

    successful_attacks += (
        successful_attack_mask.sum().item()
    )

    total += labels.size(0)


# -------------------------
# Final Results
# -------------------------

clean_accuracy = (
    100 * clean_correct / total
)

adversarial_accuracy = (
    100 * adversarial_correct / total
)

if eligible_samples > 0:
    attack_success_rate = (
        100 * successful_attacks / eligible_samples
    )
else:
    attack_success_rate = 0


print()
print("===== Results =====")

print(
    f"Clean Accuracy: "
    f"{clean_accuracy:.2f}%"
)

print(
    f"Adversarial Accuracy: "
    f"{adversarial_accuracy:.2f}%"
)

print(
    f"Initially Correct Samples: "
    f"{eligible_samples}"
)

print(
    f"Successful Attacks: "
    f"{successful_attacks}"
)

print(
    f"Attack Success Rate (ASR): "
    f"{attack_success_rate:.2f}%"
)