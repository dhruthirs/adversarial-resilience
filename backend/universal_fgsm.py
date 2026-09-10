import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vit_pytorch import ViT

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from efficient_kan import KAN


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DEVICE = torch.device("cpu")

BATCH_SIZE = 128
EPSILON = 0.25

WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "universal_fgsm_results.txt"
)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# --------------------------------------------------
# MODEL DEFINITIONS
# Must match the saved checkpoints
# --------------------------------------------------

class SimpleANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=28,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.squeeze(1)

        output, _ = self.lstm(x)

        output = output[:, -1, :]

        return self.fc(output)


class SimpleBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=28,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = x.squeeze(1)

        output, _ = self.lstm(x)

        output = output[:, -1, :]

        return self.fc(output)


class SimpleKAN(nn.Module):
    def __init__(self):
        super().__init__()

        self.kan = KAN(
            layers_hidden=[784, 64, 10],
            grid_size=3,
            spline_order=3
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.kan(x)


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
            channels=1
        )


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model(name):

    if name == "ANN":
        model = SimpleANN()
        path = os.path.join(WEIGHTS_DIR, "ann.pth")

    elif name == "CNN":
        model = SimpleCNN()
        path = os.path.join(WEIGHTS_DIR, "cnn.pth")

    elif name == "LSTM":
        model = SimpleLSTM()
        path = os.path.join(WEIGHTS_DIR, "lstm.pth")

    elif name == "BiLSTM":
        model = SimpleBiLSTM()
        path = os.path.join(WEIGHTS_DIR, "bilstm.pth")

    elif name == "KAN":
        model = SimpleKAN()
        path = os.path.join(WEIGHTS_DIR, "kan.pth")

    elif name == "ViT":
        model = SimpleViT()
        path = os.path.join(WEIGHTS_DIR, "vit.pth")

    else:
        raise ValueError(f"Unknown model: {name}")

    print(f"Loading {name} from: {path}")

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    # Handle either raw state_dict or checkpoint dictionaries
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "net" in checkpoint:
            state_dict = checkpoint["net"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    # Disable cuDNN because LSTM gradient computation
    # can cause problems during adversarial attacks.
    torch.backends.cudnn.enabled = False

    return model


# --------------------------------------------------
# FGSM
# --------------------------------------------------

def universal_fgsm(models, images, labels, epsilon):

    images = images.clone().detach().to(DEVICE)
    labels = labels.to(DEVICE)

    images.requires_grad = True

    combined_loss = 0.0

    # Calculate loss from all six models
    for model in models.values():

        outputs = model(images)

        loss = nn.CrossEntropyLoss()(outputs, labels)

        combined_loss += loss

    # Combined gradient
    gradients = torch.autograd.grad(
        combined_loss,
        images,
        retain_graph=False,
        create_graph=False
    )[0]

    # FGSM update
    adversarial_images = (
        images + epsilon * gradients.sign()
    )

    adversarial_images = torch.clamp(
        adversarial_images,
        0.0,
        1.0
    )

    return adversarial_images.detach()


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate():

    print("=" * 60)
    print("UNIVERSAL FGSM ATTACK")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epsilon: {EPSILON}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Test samples: {len(test_dataset)}")

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------

    model_names = [
        "ANN",
        "CNN",
        "LSTM",
        "BiLSTM",
        "KAN",
        "ViT"
    ]

    models = {}

    for name in model_names:
        models[name] = load_model(name)

    print("\nAll models loaded successfully.\n")

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    clean_correct = {
        name: 0 for name in model_names
    }

    adversarial_correct = {
        name: 0 for name in model_names
    }

    successful_attacks = {
        name: 0 for name in model_names
    }

    total_samples = 0

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    for batch_index, (images, labels) in enumerate(test_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # ----------------------------------------------
        # Clean predictions
        # ----------------------------------------------

        clean_predictions = {}

        for name, model in models.items():

            with torch.no_grad():

                outputs = model(images)

                predictions = outputs.argmax(dim=1)

                clean_predictions[name] = predictions

                clean_correct[name] += (
                    (predictions == labels)
                    .sum()
                    .item()
                )

        # ----------------------------------------------
        # Generate Universal FGSM
        # ----------------------------------------------

        adversarial_images = universal_fgsm(
            models,
            images,
            labels,
            EPSILON
        )

        # ----------------------------------------------
        # Evaluate adversarial images
        # ----------------------------------------------

        for name, model in models.items():

            with torch.no_grad():

                outputs = model(adversarial_images)

                predictions = outputs.argmax(dim=1)

                adversarial_correct[name] += (
                    (predictions == labels)
                    .sum()
                    .item()
                )

                # Attack success:
                # originally correct AND now wrong
                success = (
                    (clean_predictions[name] == labels)
                    &
                    (predictions != labels)
                )

                successful_attacks[name] += (
                    success.sum().item()
                )

        total_samples += labels.size(0)

        if (batch_index + 1) % 10 == 0:

            print(
                f"Processed "
                f"{total_samples}/{len(test_dataset)} samples"
            )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("UNIVERSAL FGSM RESULTS")
    print("=" * 60)

    results = []

    for name in model_names:

        clean_accuracy = (
            clean_correct[name]
            / total_samples
            * 100
        )

        adversarial_accuracy = (
            adversarial_correct[name]
            / total_samples
            * 100
        )

        if clean_correct[name] > 0:

            asr = (
                successful_attacks[name]
                / clean_correct[name]
                * 100
            )

        else:

            asr = 0.0

        print(
            f"{name:8s} | "
            f"Clean: {clean_accuracy:6.2f}% | "
            f"Adv: {adversarial_accuracy:6.2f}% | "
            f"ASR: {asr:6.2f}%"
        )

        results.append(
            (
                name,
                clean_accuracy,
                adversarial_accuracy,
                asr,
                successful_attacks[name]
            )
        )

    # --------------------------------------------------
    # Average ASR
    # --------------------------------------------------

    average_asr = sum(
        result[3] for result in results
    ) / len(results)

    print("-" * 60)
    print(f"Average Universal FGSM ASR: {average_asr:.2f}%")
    print("=" * 60)

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    with open(OUTPUT_FILE, "w") as f:

        f.write("UNIVERSAL FGSM ATTACK RESULTS\n")
        f.write("==============================\n\n")

        f.write(f"Epsilon: {EPSILON}\n")
        f.write(f"Test Samples: {total_samples}\n")
        f.write("Models: ANN, CNN, LSTM, BiLSTM, KAN, ViT\n\n")

        f.write(
            "Model        Clean Acc    Adv Acc      ASR      Successful\n"
        )
        f.write(
            "----------------------------------------------------------\n"
        )

        for (
            name,
            clean_accuracy,
            adversarial_accuracy,
            asr,
            successful
        ) in results:

            f.write(
                f"{name:<12}"
                f"{clean_accuracy:>8.2f}%    "
                f"{adversarial_accuracy:>8.2f}%    "
                f"{asr:>7.2f}%    "
                f"{successful}\n"
            )

        f.write("\n")
        f.write(
            f"Average Universal FGSM ASR: "
            f"{average_asr:.2f}%\n"
        )

    print(f"\nResults saved to:")
    print(OUTPUT_FILE)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    evaluate()