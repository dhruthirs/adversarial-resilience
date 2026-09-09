import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchattacks

from models.cnn import SimpleCNN
from backend.gan_attack import (
    AttackGenerator,
    generate_adversarial_image
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 128
EPSILON = 0.25

CNN_WEIGHTS = "weights/cnn.pth"
GAN_WEIGHTS = "weights/gan_attack_cnn_v2.pth"


# --------------------------------------------------
# LOAD CNN
# --------------------------------------------------

print("=" * 60)
print("CNN ATTACK COMPARISON")
print("=" * 60)

print(f"Using device: {DEVICE}")

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
# LOAD GAN
# --------------------------------------------------

generator = AttackGenerator(
    epsilon=0.15
).to(DEVICE)

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
# DATASET
# --------------------------------------------------

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Test samples: {len(test_dataset)}")


# --------------------------------------------------
# ATTACKS
# --------------------------------------------------

fgsm = torchattacks.FGSM(
    cnn,
    eps=EPSILON
)

pgd = torchattacks.PGD(
    cnn,
    eps=EPSILON,
    alpha=2 / 255,
    steps=40
)


# --------------------------------------------------
# METRIC STORAGE
# --------------------------------------------------

attack_names = [
    "FGSM",
    "PGD",
    "GAN"
]

correct_clean = 0

correct_fgsm = 0
correct_pgd = 0
correct_gan = 0

initially_correct = 0

successful_fgsm = 0
successful_pgd = 0
successful_gan = 0

total_gan_perturbation = 0.0

total_samples = 0


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

print("\nStarting evaluation...")
print("-" * 60)

for batch_index, (images, labels) in enumerate(test_loader):

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    # ----------------------------------------------
    # CLEAN
    # ----------------------------------------------

    with torch.no_grad():

        clean_logits = cnn(images)

        clean_predictions = clean_logits.argmax(
            dim=1
        )

    clean_correct_mask = (
        clean_predictions == labels
    )

    initially_correct += (
        clean_correct_mask.sum().item()
    )

    correct_clean += (
        clean_correct_mask.sum().item()
    )


    # ----------------------------------------------
    # FGSM
    # ----------------------------------------------

    adv_fgsm = fgsm(
        images,
        labels
    )

    with torch.no_grad():

        fgsm_predictions = cnn(
            adv_fgsm
        ).argmax(dim=1)

    correct_fgsm += (
        fgsm_predictions == labels
    ).sum().item()

    successful_fgsm += (
        clean_correct_mask
        & (fgsm_predictions != labels)
    ).sum().item()


    # ----------------------------------------------
    # PGD
    # ----------------------------------------------

    adv_pgd = pgd(
        images,
        labels
    )

    with torch.no_grad():

        pgd_predictions = cnn(
            adv_pgd
        ).argmax(dim=1)

    correct_pgd += (
        pgd_predictions == labels
    ).sum().item()

    successful_pgd += (
        clean_correct_mask
        & (pgd_predictions != labels)
    ).sum().item()


    # ----------------------------------------------
    # GAN
    # ----------------------------------------------

    with torch.no_grad():

        adv_gan, delta = generate_adversarial_image(
            generator,
            images,
            EPSILON
        )

        gan_predictions = cnn(
            adv_gan
        ).argmax(dim=1)

    correct_gan += (
        gan_predictions == labels
    ).sum().item()

    successful_gan += (
        clean_correct_mask
        & (gan_predictions != labels)
    ).sum().item()

    total_gan_perturbation += (
        delta.abs().mean().item()
        * images.size(0)
    )

    total_samples += images.size(0)


    if (batch_index + 1) % 20 == 0:

        print(
            f"Batch [{batch_index + 1}/{len(test_loader)}]"
        )


# --------------------------------------------------
# FINAL METRICS
# --------------------------------------------------

clean_accuracy = (
    correct_clean
    / total_samples
    * 100
)

fgsm_accuracy = (
    correct_fgsm
    / total_samples
    * 100
)

pgd_accuracy = (
    correct_pgd
    / total_samples
    * 100
)

gan_accuracy = (
    correct_gan
    / total_samples
    * 100
)


fgsm_asr = (
    successful_fgsm
    / initially_correct
    * 100
)

pgd_asr = (
    successful_pgd
    / initially_correct
    * 100
)

gan_asr = (
    successful_gan
    / initially_correct
    * 100
)


mean_gan_perturbation = (
    total_gan_perturbation
    / total_samples
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n")
print("=" * 70)
print("FINAL CNN ATTACK COMPARISON")
print("=" * 70)

print(
    f"{'Attack':<15}"
    f"{'Accuracy':<15}"
    f"{'ASR':<15}"
)

print("-" * 70)

print(
    f"{'Clean':<15}"
    f"{clean_accuracy:.2f}%"
)

print(
    f"{'FGSM':<15}"
    f"{fgsm_accuracy:.2f}%"
    f"{fgsm_asr:.2f}%"
)

print(
    f"{'PGD':<15}"
    f"{pgd_accuracy:.2f}%"
    f"{pgd_asr:.2f}%"
)

print(
    f"{'GAN':<15}"
    f"{gan_accuracy:.2f}%"
    f"{gan_asr:.2f}%"
)

print("-" * 70)

print(f"Initially correct: {initially_correct}")

print(
    f"FGSM successful attacks: "
    f"{successful_fgsm}"
)

print(
    f"PGD successful attacks: "
    f"{successful_pgd}"
)

print(
    f"GAN successful attacks: "
    f"{successful_gan}"
)

print(
    f"\nGAN Mean |perturbation|: "
    f"{mean_gan_perturbation:.6f}"
)

print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)