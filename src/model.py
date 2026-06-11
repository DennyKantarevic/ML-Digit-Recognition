"""Neural network model for handwritten digit recognition."""

from __future__ import annotations

import torch
from torch import nn


class MNISTCNN(nn.Module):
    """A small convolutional neural network for 28x28 grayscale digit images."""

    def __init__(self) -> None:
        super().__init__()

        # Convolution layers learn local image patterns such as strokes, curves,
        # and corners. Pooling reduces image size while keeping important signal.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # After two 2x2 pooling layers, each 28x28 image becomes 32 feature maps
        # of size 7x7. The classifier converts those features into 10 digit logits.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw class scores for digit labels 0 through 9."""
        features = self.features(images)
        return self.classifier(features)

