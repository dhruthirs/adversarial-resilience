import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# CNN Model
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# -----------------------------
# Load Model
# -----------------------------
model = SimpleCNN().to(device)

model.load_state_dict(
    torch.load(
        "./weights/cnn.pth",
        map_location=device
    )
)

model.eval()


# -----------------------------
# MNIST Test Dataset
# -----------------------------
transform = transforms.ToTensor()

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


# -----------------------------
# FGSM Attack
# -----------------------------
def fgsm_attack(model, images, labels, epsilon):

    images = images.clone().detach().to(device)
    labels = labels.to(device)

    images.requires_grad = True

    outputs = model(images)

    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    loss.backward()

    gradient = images.grad.data.sign()

    adversarial_images = images + epsilon * gradient

    adversarial_images = torch.clamp(
        adversarial_images,
        0,
        1
    )

    return adversarial_images


# -----------------------------
# Evaluate FGSM
# -----------------------------
epsilon = 0.25

correct_clean = 0
correct_adversarial = 0
successful_attacks = 0

total = 0


for images, labels in test_loader:

    images = images.to(device)
    labels = labels.to(device)

    # Clean predictions
    with torch.no_grad():
        outputs = model(images)
        clean_predictions = outputs.argmax(dim=1)

    correct_mask = clean_predictions == labels

    correct_clean += correct_mask.sum().item()

    # Generate adversarial examples
    adversarial_images = fgsm_attack(
        model,
        images,
        labels,
        epsilon
    )

    # Adversarial predictions
    with torch.no_grad():
        outputs_adv = model(adversarial_images)
        adversarial_predictions = outputs_adv.argmax(dim=1)

    correct_adversarial += (
        adversarial_predictions == labels
    ).sum().item()

    # Attack succeeds when:
    # 1. Original prediction was correct
    # 2. Adversarial prediction becomes incorrect
    successful_attacks += (
        correct_mask &
        (adversarial_predictions != labels)
    ).sum().item()

    total += labels.size(0)


# -----------------------------
# Metrics
# -----------------------------
clean_accuracy = (correct_clean / total) * 100

adversarial_accuracy = (
    correct_adversarial / total
) * 100

if correct_clean > 0:
    attack_success_rate = (
        successful_attacks / correct_clean
    ) * 100
else:
    attack_success_rate = 0


# -----------------------------
# Results
# -----------------------------
print("\n===== CNN FGSM Attack =====")
print(f"Epsilon: {epsilon}")
print(f"Clean Accuracy: {clean_accuracy:.2f}%")
print(
    f"Adversarial Accuracy: "
    f"{adversarial_accuracy:.2f}%"
)
print(
    f"Initially Correct: "
    f"{correct_clean}"
)
print(
    f"Successful Attacks: "
    f"{successful_attacks}"
)
print(
    f"Attack Success Rate: "
    f"{attack_success_rate:.2f}%"
)