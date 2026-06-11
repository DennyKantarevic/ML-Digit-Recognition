"""Evaluate a trained MNIST model on the test dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    # Allows running this file directly with: python src/evaluate.py
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import get_mnist_loaders
from src.model import MNISTCNN
from src.utils import get_device

DEFAULT_MODEL_PATH = Path("models/mnist_model.pth")


def load_trained_model(model_path: str | Path, device: torch.device) -> MNISTCNN:
    """Load a saved MNISTCNN checkpoint from disk."""
    model = MNISTCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # The training script saves metadata with the state dict. This fallback also
    # supports loading a plain state_dict if a user saves one manually.
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Return model accuracy as a percentage."""
    model.eval()
    correct_predictions = 0
    total_examples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        correct_predictions += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    return 100.0 * correct_predictions / total_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MNIST CNN.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-limit", type=int, default=2_000)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    _, test_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_limit=1,
        test_limit=args.test_limit,
    )
    model = load_trained_model(args.model_path, device)

    accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()

