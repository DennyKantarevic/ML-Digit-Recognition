import torch
from PIL import Image

from src.data import build_mnist_transforms


def test_mnist_transform_prepares_single_channel_28_by_28_tensor():
    transform = build_mnist_transforms()
    image = Image.new("L", (40, 40), color=128)

    tensor = transform(image)

    assert tensor.shape == (1, 28, 28)
    assert torch.isfinite(tensor).all()

