"""Train the MNIST digit recognition model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    # Allows running this file directly with: python src/train.py
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import get_mnist_loaders
from src.evaluate import evaluate_model
from src.model import MNISTCNN
from src.utils import get_device

DEFAULT_MODEL_PATH = Path("models/mnist_model.pth")


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one full pass over the training data and return average loss."""
    model.train()
    total_loss = 0.0
    total_examples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients, compute new predictions, and measure the error.
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, labels)

        # Backpropagation calculates gradients, then the optimizer updates weights.
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def save_model(
    model: torch.nn.Module,
    model_path: str | Path,
    accuracy: float,
) -> None:
    """Save model weights and useful metadata for later evaluation/prediction."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "accuracy": accuracy,
            "classes": list(range(10)),
        },
        model_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on the MNIST dataset.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-limit", type=int, default=10_000)
    parser.add_argument("--test-limit", type=int, default=2_000)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    train_loader, test_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )

    model = MNISTCNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    final_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        average_loss = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )
        final_accuracy = evaluate_model(model, test_loader, device)

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"Training Loss: {average_loss:.4f} - "
            f"Test Accuracy: {final_accuracy:.2f}%"
        )

    save_model(model, args.model_path, final_accuracy)
    print(f"Saved trained model to {args.model_path}")


if __name__ == "__main__":
    main()

