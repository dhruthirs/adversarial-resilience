import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from backend.gan_attack import AttackGenerator, generate_adversarial_image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CNN_WEIGHTS = "weights/cnn.pth"
GAN_WEIGHTS = "weights/gan_attack_cnn.pth"

EPSILON = 0.25


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

cnn = SimpleCNN().to(DEVICE)

cnn.load_state_dict(
    torch.load(
        CNN_WEIGHTS,
        map_location=DEVICE,
        weights_only=True
    )
)

cnn.eval()


generator = AttackGenerator(epsilon=EPSILON).to(DEVICE)

generator.load_state_dict(
    torch.load(
        GAN_WEIGHTS,
        map_location=DEVICE,
        weights_only=True
    )
)

generator.eval()


# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)


# --------------------------------------------------
# FIND SUCCESSFUL ATTACKS
# --------------------------------------------------

successful_examples = []

with torch.no_grad():

    for image, label in dataset:

        image_batch = image.unsqueeze(0).to(DEVICE)
        label_tensor = torch.tensor([label]).to(DEVICE)

        clean_output = cnn(image_batch)
        clean_prediction = clean_output.argmax(dim=1).item()

        # We only consider images CNN classified correctly
        if clean_prediction != label:
            continue

        adversarial_image, delta = generate_adversarial_image(
            generator,
            image_batch,
            EPSILON
        )

        adversarial_output = cnn(adversarial_image)
        adversarial_prediction = adversarial_output.argmax(dim=1).item()

        # Keep only successfully attacked examples
        if adversarial_prediction != label:

            successful_examples.append(
                (
                    image.squeeze().cpu(),
                    delta.squeeze().cpu(),
                    adversarial_image.squeeze().cpu(),
                    label,
                    clean_prediction,
                    adversarial_prediction
                )
            )

        if len(successful_examples) >= 5:
            break


# --------------------------------------------------
# VISUALIZE
# --------------------------------------------------

fig, axes = plt.subplots(
    5,
    3,
    figsize=(9, 14)
)

for row, example in enumerate(successful_examples):

    clean_image = example[0]
    delta = example[1]
    adversarial_image = example[2]

    label = example[3]
    clean_prediction = example[4]
    adversarial_prediction = example[5]

    # Clean image
    axes[row, 0].imshow(
        clean_image,
        cmap="gray"
    )

    axes[row, 0].set_title(
        f"Clean\nTrue: {label}\nPred: {clean_prediction}"
    )

    # Perturbation
    axes[row, 1].imshow(
        delta,
        cmap="gray"
    )

    axes[row, 1].set_title(
        "GAN Perturbation"
    )

    # Adversarial image
    axes[row, 2].imshow(
        adversarial_image,
        cmap="gray"
    )

    axes[row, 2].set_title(
        f"GAN Adversarial\nPred: {adversarial_prediction}"
    )

    axes[row, 0].axis("off")
    axes[row, 1].axis("off")
    axes[row, 2].axis("off")


plt.tight_layout()

output_path = "evaluation/gan_attack_examples.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("=" * 60)
print("GAN VISUALIZATION COMPLETED")
print("=" * 60)
print(f"Successful examples found: {len(successful_examples)}")
print(f"Saved to: {output_path}")