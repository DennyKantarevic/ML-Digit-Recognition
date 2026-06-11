"""Small shared helpers used by the training, evaluation, and prediction scripts."""

from __future__ import annotations

import torch


def get_device(preferred_device: str | None = None) -> torch.device:
    """Choose a compute device, using a GPU when one is available."""
    if preferred_device:
        return torch.device(preferred_device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple Silicon Macs expose the GPU through the MPS backend.
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

