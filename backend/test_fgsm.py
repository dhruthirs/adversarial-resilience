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
# FGSM Attack
# -------------------------

def fgsm_attack(model, images, labels, epsilon):

    images = images.clone().detach().to(device)
    labels = labels.to(device)

    images.requires_grad = True

    outputs = model(images)

    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data

    perturbed_images = (
        images + epsilon * data_grad.sign()
    )

    perturbed_images = torch.clamp(
        perturbed_images,
        0,
        1
    )

    return perturbed_images.detach()


# -------------------------
# Evaluate FGSM
# -------------------------

epsilon = 0.25

clean_correct = 0
adversarial_correct = 0

successful_attacks = 0
eligible_samples = 0

total = 0


print()
print("===== FGSM Attack =====")
print("Epsilon:", epsilon)


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
    # Generate adversarial images
    # -------------------------

    adversarial_images = fgsm_attack(
        model,
        images,
        labels,
        epsilon
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

    # Only samples correctly classified
    # before the attack are considered.

    clean_correct_mask = (
        clean_predictions == labels
    )

    successful_attack_mask = (
        clean_correct_mask
        & (adversarial_predictions != labels)
    )

    eligible_samples += clean_correct_mask.sum().item()

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