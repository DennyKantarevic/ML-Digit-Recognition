import torch

from src.predict import predict_digit_from_tensor


class AlwaysSevenModel(torch.nn.Module):
    def forward(self, images):
        logits = torch.zeros(images.shape[0], 10)
        logits[:, 7] = 12.0
        return logits


def test_predict_digit_from_tensor_returns_digit_and_confidence():
    image_tensor = torch.zeros(1, 1, 28, 28)

    digit, confidence = predict_digit_from_tensor(
        AlwaysSevenModel(),
        image_tensor,
        device=torch.device("cpu"),
    )

    assert digit == 7
    assert 0.99 <= confidence <= 1.0

