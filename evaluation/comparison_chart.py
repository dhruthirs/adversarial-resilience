"""Generate CSV and chart comparing attacks and restoration methods."""

import csv
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.gan_attack import AttackGenerator, generate_adversarial_image
from backend.mnist_models import load_mnist_classifiers
from backend.restoration import WaveletSplineRestorer
from backend.wgan_restoration import WGANRestorer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = os.path.join(ROOT, "weights")
OUTPUT_CSV = os.path.join(ROOT, "evaluation", "restoration_comparison.csv")
OUTPUT_PNG = os.path.join(ROOT, "evaluation", "restoration_comparison.png")
EPSILON = 0.30


def strong_pgd(model, images, labels):
    original = images.detach()
    adversarial = (
        original + torch.empty_like(original).uniform_(-EPSILON, EPSILON)
    ).clamp(0, 1)
    for _ in range(40):
        adversarial.requires_grad_(True)
        loss = F.cross_entropy(model(adversarial), labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]
        adversarial = (
            original + (adversarial.detach() + 2 / 255 * gradient.sign()
            - original).clamp(-EPSILON, EPSILON)
        ).clamp(0, 1)
    return adversarial.detach()


def strong_fgsm(model, images, labels):
    adversarial = images.detach().requires_grad_(True)
    loss = F.cross_entropy(model(adversarial), labels)
    gradient = torch.autograd.grad(loss, adversarial)[0]
    return (images + EPSILON * gradient.sign()).clamp(0, 1).detach()


def gan_refined_attack(model, generator, images, labels):
    """Use the GAN perturbation as a start, then maximize classifier loss."""
    adversarial, _ = generate_adversarial_image(generator, images, EPSILON)
    original = images.detach()
    for _ in range(10):
        adversarial.requires_grad_(True)
        loss = F.cross_entropy(model(adversarial), labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]
        adversarial = (
            original
            + (adversarial.detach() + 2 / 255 * gradient.sign() - original)
            .clamp(-EPSILON, EPSILON)
        ).clamp(0, 1)
    return adversarial.detach()


def main():
    if not os.path.exists(os.path.join(WEIGHTS, "wgan_restorer.pth")):
        raise FileNotFoundError(
            "Train weights\\wgan_restorer.pth with "
            "backend\\train_wgan_restorer.py first."
        )
    models = load_mnist_classifiers(WEIGHTS, DEVICE)
    spline = WaveletSplineRestorer().to(DEVICE)
    spline.load_state_dict(torch.load(
        os.path.join(WEIGHTS, "restorer.pth"),
        map_location=DEVICE,
        weights_only=False,
    ))
    wgan = WGANRestorer().to(DEVICE)
    wgan.load_state_dict(torch.load(
        os.path.join(WEIGHTS, "wgan_restorer.pth"),
        map_location=DEVICE,
        weights_only=False,
    ))
    spline.eval()
    wgan.eval()
    attack_generator = AttackGenerator(epsilon=EPSILON).to(DEVICE)
    attack_generator.load_state_dict(torch.load(
        os.path.join(WEIGHTS, "gan_attack_all.pth"),
        map_location=DEVICE,
        weights_only=False,
    ))
    attack_generator.eval()
    loader = DataLoader(
        torchvision.datasets.MNIST(
            root=os.path.join(ROOT, "data"), train=False, download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=256,
        shuffle=False,
    )
    totals = {
        name: {key: 0 for key in (
            "clean", "fgsm_success", "pgd_success", "gan_success",
            "fgsm_correct", "pgd_correct", "gan_correct",
            "gan_refined_success", "gan_refined_correct",
            "bspline", "wgan", "combined",
            "bspline_success", "wgan_success", "combined_success",
        )}
        for name in models
    }
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        for name, model in models.items():
            with torch.no_grad():
                clean_prediction = model(images).argmax(1)
                clean_correct = clean_prediction.eq(labels)
            pgd = strong_pgd(model, images, labels)
            gan_attack, _ = generate_adversarial_image(
                attack_generator, images, EPSILON
            )
            refined_gan_attack = gan_refined_attack(
                model, attack_generator, images, labels
            )
            fgsm = strong_fgsm(model, images, labels)
            with torch.no_grad():
                fgsm_prediction = model(fgsm).argmax(1)
                pgd_prediction = model(pgd).argmax(1)
                gan_prediction = model(gan_attack).argmax(1)
                refined_gan_prediction = model(refined_gan_attack).argmax(1)
                bspline_prediction = model(spline(pgd)).argmax(1)
                wgan_prediction = model(wgan(pgd)).argmax(1)
                combined_prediction = model(wgan(spline(pgd))).argmax(1)
                totals[name]["clean"] += clean_correct.sum().item()
                totals[name]["fgsm_success"] += (
                    (clean_correct & ~fgsm_prediction.eq(labels)).sum().item()
                )
                totals[name]["fgsm_correct"] += fgsm_prediction.eq(labels).sum().item()
                totals[name]["pgd_success"] += (
                    (clean_correct & ~pgd_prediction.eq(labels)).sum().item()
                )
                totals[name]["pgd_correct"] += pgd_prediction.eq(labels).sum().item()
                totals[name]["gan_success"] += (
                    (clean_correct & ~gan_prediction.eq(labels)).sum().item()
                )
                totals[name]["gan_correct"] += gan_prediction.eq(labels).sum().item()
                totals[name]["gan_refined_success"] += (
                    (clean_correct & ~refined_gan_prediction.eq(labels)).sum().item()
                )
                totals[name]["gan_refined_correct"] += refined_gan_prediction.eq(labels).sum().item()
                totals[name]["bspline"] += bspline_prediction.eq(labels).sum().item()
                totals[name]["wgan"] += wgan_prediction.eq(labels).sum().item()
                totals[name]["combined"] += combined_prediction.eq(labels).sum().item()
                attacked = clean_correct & ~pgd_prediction.eq(labels)
                totals[name]["bspline_success"] += (
                    (attacked & bspline_prediction.eq(labels)).sum().item()
                )
                totals[name]["wgan_success"] += (
                    (attacked & wgan_prediction.eq(labels)).sum().item()
                )
                totals[name]["combined_success"] += (
                    (attacked & combined_prediction.eq(labels)).sum().item()
                )
        total += labels.numel()
        print(f"Processed {total}/10000 ({100 * total / 10000:.1f}%)")

    rows = []
    for name, values in totals.items():
        clean = 100 * values["clean"] / total
        row = {
            "model": name,
            "clean_accuracy": clean,
            "fgsm_asr": 100 * values["fgsm_success"] / values["clean"],
            "pgd_asr": 100 * values["pgd_success"] / values["clean"],
            "gan_asr": 100 * values["gan_success"] / values["clean"],
            "gan_refined_asr": (
                100 * values["gan_refined_success"] / values["clean"]
            ),
            "bspline_accuracy": 100 * values["bspline"] / total,
            "wgan_accuracy": 100 * values["wgan"] / total,
            "bspline_wgan_accuracy": 100 * values["combined"] / total,
            "bspline_restoration_success": (
                100 * values["bspline_success"] / values["pgd_success"]
                if values["pgd_success"] else 0
            ),
            "wgan_restoration_success": (
                100 * values["wgan_success"] / values["pgd_success"]
                if values["pgd_success"] else 0
            ),
            "bspline_wgan_restoration_success": (
                100 * values["combined_success"] / values["pgd_success"]
                if values["pgd_success"] else 0
            ),
        }
        row["fgsm_drop"] = clean - 100 * values["fgsm_correct"] / total
        row["pgd_drop"] = clean - 100 * values["pgd_correct"] / total
        row["gan_drop"] = clean - 100 * values["gan_correct"] / total
        row["gan_refined_drop"] = (
            clean - 100 * values["gan_refined_correct"] / total
        )
        rows.append(row)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["model"] for row in rows]
    x = range(len(labels))
    plt.figure(figsize=(14, 7))
    for key, title in (
        ("clean_accuracy", "Clean"),
        ("bspline_accuracy", "B-spline"),
        ("wgan_accuracy", "WGAN"),
        ("bspline_wgan_accuracy", "B-spline + WGAN"),
    ):
        plt.plot(x, [row[key] for row in rows], marker="o", label=title)
    plt.xticks(list(x), labels)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("MNIST Attack and Restoration Comparison")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=160)
    print(f"Saved {OUTPUT_CSV}")
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
