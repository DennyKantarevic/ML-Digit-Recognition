import torch

from src.model import MNISTCNN


def test_mnist_cnn_returns_one_logit_per_digit():
    model = MNISTCNN()
    sample_images = torch.randn(4, 1, 28, 28)

    logits = model(sample_images)

    assert logits.shape == (4, 10)

