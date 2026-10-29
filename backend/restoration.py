"""Differentiable B-spline and wavelet restoration for MNIST-sized inputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletSplineRestorer(nn.Module):
    """Inference-time restoration block for [0, 1] grayscale images.

    The cubic B-spline kernel provides stable low-pass reconstruction. The
    wavelet responses identify coarse and fine-scale structure, while the
    learned blend keeps the initial transformation close to identity.
    """

    def __init__(self, channels=1):
        super().__init__()
        self.channels = channels

        spline = torch.tensor(
            [1.0, 4.0, 6.0, 4.0, 1.0],
            dtype=torch.float32,
        ) / 16.0
        spline_kernel = spline[:, None] @ spline[None, :]

        coords = torch.arange(7, dtype=torch.float32) - 3
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        radius_squared = xx.square() + yy.square()

        mexican_hat = (
            1.0 - radius_squared / (1.2**2)
        ) * torch.exp(-radius_squared / (2 * 1.2**2))
        mexican_hat -= mexican_hat.mean()

        morlet = torch.exp(-radius_squared / (2 * 1.5**2)) * torch.cos(1.5 * xx)
        morlet -= morlet.mean()

        self.register_buffer(
            "spline_kernel",
            spline_kernel[None, None].repeat(channels, 1, 1, 1),
        )
        self.register_buffer(
            "mexican_hat",
            (mexican_hat / mexican_hat.abs().sum().clamp_min(1e-8))
            .view(1, 1, 7, 7)
            .repeat(channels, 1, 1, 1),
        )
        self.register_buffer(
            "morlet",
            (morlet / morlet.abs().sum().clamp_min(1e-8))
            .view(1, 1, 7, 7)
            .repeat(channels, 1, 1, 1),
        )

        # These are intentionally fixed defaults. Without a trained
        # restorer checkpoint, a near-zero learnable blend would be identity.
        self.register_buffer(
            "projection_strength",
            torch.tensor(0.85),
        )
        self.register_buffer(
            "wavelet_strength",
            torch.tensor(0.10),
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(channels * 3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.refinement = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
        )

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != self.channels:
            raise ValueError(
                f"Expected [batch, {self.channels}, height, width], "
                f"got {tuple(image.shape)}"
            )

        spline = F.conv2d(
            image,
            self.spline_kernel.to(device=image.device, dtype=image.dtype),
            padding=2,
            groups=self.channels,
        )
        coarse = F.conv2d(
            image,
            self.mexican_hat.to(device=image.device, dtype=image.dtype),
            padding=3,
            groups=self.channels,
        )
        fine = F.conv2d(
            image,
            self.morlet.to(device=image.device, dtype=image.dtype),
            padding=3,
            groups=self.channels,
        )

        strength = self.projection_strength.to(
            device=image.device,
            dtype=image.dtype,
        )
        wavelet_strength = self.wavelet_strength.to(
            device=image.device,
            dtype=image.dtype,
        )
        projected = (1.0 - strength) * image + strength * spline

        # Mexican-Hat and Morlet responses represent high-frequency detail.
        # Subtracting them suppresses adversarial detail instead of adding it.
        filtered = projected - wavelet_strength * (coarse + fine)
        features = self.encoder(
            torch.cat((filtered, coarse, fine), dim=1)
        )
        restored = filtered + self.refinement(features)

        return restored.clamp(0.0, 1.0)


def select_restoration_candidate(candidates, classifier, reference_prediction):
    names = list(candidates)
    images = torch.stack([candidates[name] for name in names], dim=1)
    logits = torch.stack(
        [classifier(candidates[name]) for name in names],
        dim=1,
    )
    predictions = logits.argmax(dim=2)
    confidence = logits.softmax(dim=2).amax(dim=2)
    reference = torch.as_tensor(
        reference_prediction,
        device=images.device,
        dtype=predictions.dtype,
    )
    if reference.ndim == 0:
        reference = reference.expand(images.size(0))
    matches = predictions.eq(reference[:, None])
    matching_scores = confidence.masked_fill(~matches, float("-inf"))
    has_match = matches.any(dim=1)
    matching_choice = matching_scores.argmax(dim=1)
    fallback_choice = confidence.argmax(dim=1)
    choice = torch.where(has_match, matching_choice, fallback_choice)
    batch = torch.arange(images.size(0), device=images.device)
    return images[batch, choice], logits[batch, choice]
