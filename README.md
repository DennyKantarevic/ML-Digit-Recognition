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

## Prerequisites

Install these before running the project:

- Python 3.10 or newer
- Visual Studio Code
- VS Code Python extension from Microsoft
- Git, if you want to clone the repository from GitHub

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

## Run the Project in VS Code

### 1. Open the Project

Clone the repository or open the local folder in VS Code.

```bash
git clone https://github.com/DennyKantarevic/ML-Digit-Recognition.git
cd ML-Digit-Recognition
code .
```

If you already have the project folder, open VS Code and choose:

```text
File -> Open Folder -> ML-Digit-Recognition
```

### 2. Open the VS Code Terminal

In VS Code, open:

```text
Terminal -> New Terminal
```

Run the commands below in the VS Code terminal, not in the Python Debug Console.

## Setup on macOS

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm PyTorch is installed:

```bash
python -c "import torch; print(torch.__version__)"
```

## Setup on Windows

Create a virtual environment:

```powershell
py -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm PyTorch is installed:

```powershell
python -c "import torch; print(torch.__version__)"
```

## Select the Python Interpreter in VS Code

After creating `.venv`, tell VS Code to use it:

1. Press `Cmd + Shift + P` on macOS or `Ctrl + Shift + P` on Windows.
2. Search for `Python: Select Interpreter`.
3. Select the interpreter inside this project.

On macOS, it should look like:

```bash
.venv/bin/python
```

On Windows, it should look like:

```text
.\.venv\Scripts\python.exe
```

Open a new VS Code terminal after selecting the interpreter.

## Train the Model

Run training from the terminal:

```bash
python -m src.train
```

By default, training uses 2 epochs, 10,000 training images, and 2,000 test images so it finishes quickly on a normal laptop. The script automatically uses CUDA, Apple Silicon MPS, or CPU depending on what is available.

For a stronger MNIST result, train on the full dataset:

```bash
python -m src.train --epochs 8 --train-limit 60000 --test-limit 10000 --batch-size 64
```

Example output:

```text
Using device: mps
Epoch 1/8 - Training Loss: 0.2132 - Test Accuracy: 97.89%
Epoch 2/8 - Training Loss: 0.0755 - Test Accuracy: 98.61%
Epoch 3/8 - Training Loss: 0.0569 - Test Accuracy: 98.58%
Epoch 4/8 - Training Loss: 0.0471 - Test Accuracy: 98.75%
Epoch 5/8 - Training Loss: 0.0398 - Test Accuracy: 98.86%
Epoch 6/8 - Training Loss: 0.0318 - Test Accuracy: 98.96%
Epoch 7/8 - Training Loss: 0.0312 - Test Accuracy: 98.89%
Epoch 8/8 - Training Loss: 0.0252 - Test Accuracy: 99.06%
Saved trained model to models/mnist_model.pth
```

Your exact numbers may vary slightly, especially if you train on CPU, CUDA, or Apple Silicon MPS.

## Evaluate the Model

After training, evaluate the saved model:

```bash
python -m src.evaluate --test-limit 10000
```

Example output:

```text
Test Accuracy: 99.06%
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
python -m pytest tests -q
```

## Common VS Code Issue: No Module Named `torch`

If you see this error:

```text
ModuleNotFoundError: No module named 'torch'
```

VS Code is probably using the wrong Python interpreter, or the terminal is not using the virtual environment.

Fix it:

1. Open the Command Palette.
2. Run `Python: Select Interpreter`.
3. Choose the interpreter inside `.venv`.
4. Open a new terminal.
5. Activate the environment again.
6. Reinstall dependencies if needed.

macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__)"
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__)"
```

Avoid using the VS Code Python Debug Console for these setup commands. Use the VS Code terminal.

## What I Learned

This project helped me understand how an image recognition model is built end to end. I learned how to load and preprocess image data, define a neural network with convolutional layers, train the model with backpropagation, measure accuracy on test data, save trained weights, and reuse a trained model for predictions on new images.
