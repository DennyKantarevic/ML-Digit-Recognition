"""Use a trained MNIST model to predict a digit from a new image."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

if __package__ is None or __package__ == "":
    # Allows running this file directly with: python src/predict.py
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import MNIST_MEAN, MNIST_STD
from src.evaluate import DEFAULT_MODEL_PATH, load_trained_model
from src.utils import get_device


def preprocess_image(image_path: str | Path, invert: bool = False) -> torch.Tensor:
    """Prepare a local image file so it matches the MNIST training format."""
    image = Image.open(image_path).convert("L")

    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )
    image_tensor = transform(image)

    # MNIST digits are light strokes on a dark background. Invert photos or
    # drawings that use dark ink on a light background.
    if invert:
        image_tensor = 1.0 - image_tensor

    normalize = transforms.Normalize(MNIST_MEAN, MNIST_STD)
    return normalize(image_tensor).unsqueeze(0)


@torch.no_grad()
def predict_digit_from_tensor(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[int, float]:
    """Return the predicted digit and confidence for one preprocessed image."""
    model.eval()
    image_tensor = image_tensor.to(device)

    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)
    confidence, predicted_digit = torch.max(probabilities, dim=1)

    return int(predicted_digit.item()), float(confidence.item())


def predict_digit(
    image_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device_name: str | None = None,
    invert: bool = False,
) -> tuple[int, float]:
    """Load a trained model and predict the digit in an image file."""
    device = get_device(device_name)
    model = load_trained_model(model_path, device)
    image_tensor = preprocess_image(image_path, invert=invert)
    return predict_digit_from_tensor(model, image_tensor, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a digit from an image file.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Use when the image has a dark digit on a light background.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    digit, confidence = predict_digit(
        image_path=args.image_path,
        model_path=args.model_path,
        device_name=args.device,
        invert=args.invert,
    )

    print(f"Predicted Digit: {digit}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()

