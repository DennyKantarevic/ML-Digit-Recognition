# ML-Digit-Recognition

Built in Python with PyTorch — a basic machine learning project that trains a neural network to recognize handwritten digits from image data. It covers the full ML workflow, including preprocessing, model training, testing, accuracy evaluation, and using the trained model to make predictions on new digit inputs. The project was a strong introduction to computer vision, neural networks, and how AI models learn patterns from data.

## What This Project Does

This project trains a small convolutional neural network (CNN) on the MNIST handwritten digit dataset. MNIST contains 28x28 grayscale images of digits from 0 to 9, and the model learns to classify each image into the correct digit class.

The workflow includes:

- Downloading MNIST with `torchvision.datasets.MNIST`
- Preprocessing images with torchvision transforms
- Creating train and test data loaders
- Defining a CNN model in PyTorch
- Training the model over multiple epochs
- Evaluating test accuracy
- Saving trained weights to `models/mnist_model.pth`
- Predicting a digit from a new image file

## Technologies Used

- Python
- PyTorch
- Torchvision
- Pillow
- Pytest

## Project Structure

```text
ML-Digit-Recognition/
├── data/
├── models/
├── src/
│   ├── data.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Install Dependencies

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Train the Model

Run training from the terminal:

```bash
python -m src.train
```

By default, training uses 2 epochs, 10,000 training images, and 2,000 test images so it finishes quickly on a normal laptop. The script automatically uses CUDA or Apple Silicon MPS when available, otherwise it uses CPU.

To train longer or use more data:

```bash
python -m src.train --epochs 5 --train-limit 60000 --test-limit 10000
```

Example output:

```text
Using device: cpu
Epoch 1/2 - Training Loss: 0.8152 - Test Accuracy: 88.40%
Epoch 2/2 - Training Loss: 0.2867 - Test Accuracy: 93.75%
Saved trained model to models/mnist_model.pth
```

## Evaluate the Model

After training, evaluate the saved model:

```bash
python -m src.evaluate
```

Example output:

```text
Test Accuracy: 93.75%
```

## Make a Prediction

After training, pass an image file to the prediction script:

```bash
python -m src.predict path/to/digit.png
```

If the image has a dark digit on a light background, use `--invert` so it better matches the MNIST format:

```bash
python -m src.predict path/to/digit.png --invert
```

Example output:

```text
Predicted Digit: 7
Confidence: 99.82%
```

## Run Tests

```bash
pytest
```

## What I Learned

This project helped me understand how an image recognition model is built end to end. I learned how to load and preprocess image data, define a neural network with convolutional layers, train the model with backpropagation, measure accuracy on test data, save trained weights, and reuse a trained model for predictions on new images.

