"""Dataset loading and preprocessing for MNIST."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Standard MNIST normalization values. Normalization helps the model train more
# smoothly by keeping pixel values on a consistent scale.
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def build_mnist_transforms() -> transforms.Compose:
    """Create the preprocessing pipeline used for MNIST images."""
    return transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    """Return a smaller view of a dataset for quick local experiments."""
    if limit is None:
        return dataset

    if limit <= 0:
        raise ValueError("Dataset limit must be a positive integer.")

    return Subset(dataset, range(min(limit, len(dataset))))


def get_mnist_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 64,
    train_limit: int | None = 10_000,
    test_limit: int | None = 2_000,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Download MNIST if needed and return train/test data loaders."""
    transform = build_mnist_transforms()

    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    train_dataset = limit_dataset(train_dataset, train_limit)
    test_dataset = limit_dataset(test_dataset, test_limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader

