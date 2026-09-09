import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from backend.gan_attack import AttackGenerator, generate_adversarial_image


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPSILON = 0.25

CNN_WEIGHTS = "weights/cnn.pth"
GAN_WEIGHTS = "weights/gan_attack_cnn_v2.pth"


# --------------------------------------------------
# LOAD CNN
# --------------------------------------------------

print("=" * 60)
print("GAN ATTACK TESTING")
print("=" * 60)

print(f"Using device: {DEVICE}")

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

print("CNN loaded successfully.")


# --------------------------------------------------
# LOAD GAN GENERATOR
# --------------------------------------------------

print("\nLoading trained GAN generator...")

generator = AttackGenerator(epsilon=0.15).to(DEVICE)

generator.load_state_dict(
    torch.load(
        GAN_WEIGHTS,
        map_location=DEVICE,
        weights_only=True
    )
)

generator.eval()

print("GAN generator loaded successfully.")


# --------------------------------------------------
# LOAD MNIST TEST DATA
# --------------------------------------------------

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
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

print(f"\nMNIST test samples: {len(test_dataset)}")


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

total_samples = 0

clean_correct = 0
adversarial_correct = 0

initially_correct = 0
successful_attacks = 0

total_perturbation = 0.0


print("\nStarting GAN attack evaluation...")
print("-" * 60)


with torch.no_grad():

    for batch_index, (images, labels) in enumerate(test_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clean prediction
        clean_logits = cnn(images)
        clean_predictions = clean_logits.argmax(dim=1)

        # Generate GAN adversarial images
        adversarial_images, delta = generate_adversarial_image(
            generator,
            images,
            EPSILON
        )

        # Adversarial prediction
        adversarial_logits = cnn(adversarial_images)
        adversarial_predictions = adversarial_logits.argmax(dim=1)

        # Accuracy
        clean_correct += (
            clean_predictions == labels
        ).sum().item()

        adversarial_correct += (
            adversarial_predictions == labels
        ).sum().item()

        # ASR:
        # Only images correctly classified before attack
        correct_before_attack = (
            clean_predictions == labels
        )

        successful_attack = (
            correct_before_attack
            & (adversarial_predictions != labels)
        )

        initially_correct += (
            correct_before_attack
        ).sum().item()

        successful_attacks += (
            successful_attack
        ).sum().item()

        # Perturbation
        total_perturbation += (
            delta.abs().mean().item()
            * images.size(0)
        )

        total_samples += images.size(0)

        if (batch_index + 1) % 20 == 0:
            current_asr = (
                successful_attacks / initially_correct * 100
                if initially_correct > 0
                else 0.0
            )

            print(
                f"Batch [{batch_index + 1}/{len(test_loader)}] "
                f"Current ASR: {current_asr:.2f}%"
            )


# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

clean_accuracy = (
    clean_correct / total_samples * 100
)

adversarial_accuracy = (
    adversarial_correct / total_samples * 100
)

attack_success_rate = (
    successful_attacks / initially_correct * 100
    if initially_correct > 0
    else 0.0
)

mean_perturbation = (
    total_perturbation / total_samples
)


print("\n")
print("=" * 60)
print("GAN ATTACK RESULTS")
print("=" * 60)

print(f"Clean Accuracy          : {clean_accuracy:.2f}%")
print(f"GAN Adversarial Accuracy: {adversarial_accuracy:.2f}%")
print(f"GAN Attack Success Rate : {attack_success_rate:.2f}%")
print(f"Mean |perturbation|     : {mean_perturbation:.6f}")

print("\nAdditional information:")
print(f"Initially correct      : {initially_correct}")
print(f"Successful attacks     : {successful_attacks}")
print(f"Test samples           : {total_samples}")

print("\n" + "=" * 60)
print("GAN ATTACK TESTING COMPLETED")
print("=" * 60)