# Handwritten Digit Recognition using Convolutional Neural Networks (CNN)

A production-quality deep learning project that classifies handwritten digits (0–9) using the MNIST dataset and a Convolutional Neural Network built with TensorFlow/Keras.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Project Overview

This project implements an end-to-end image classification pipeline for recognizing handwritten digits. It covers the full machine learning workflow:

- **Data loading and preprocessing** — MNIST dataset via Keras
- **Model architecture** — Custom CNN with convolutional, pooling, and dense layers
- **Training** — With callbacks (EarlyStopping, ModelCheckpoint, learning rate scheduling)
- **Evaluation** — Accuracy, loss, classification report, and confusion matrix
- **Inference** — Predict digits from custom images using OpenCV
- **Visualization** — Learning curves, sample predictions, and confusion matrix heatmaps

**Target accuracy:** 98–99% on the MNIST test set.

---

## Dataset Description

| Property | Value |
|----------|-------|
| **Dataset** | MNIST Handwritten Digits |
| **Total images** | 70,000 |
| **Training set** | 60,000 images |
| **Test set** | 10,000 images |
| **Image size** | 28 × 28 pixels (grayscale) |
| **Classes** | 10 (digits 0–9) |

**Official sources:**
- [MNIST Database](https://yann.lecun.com/exdb/mnist/)
- [Keras MNIST API](https://keras.io/api/datasets/mnist/)
- [PyTorch MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)

---

## CNN Architecture

```
Input (28×28×1)
    │
    ▼
[Data Augmentation]  ← optional (rotation, translation, zoom)
    │
    ▼
Conv2D (32 filters, 3×3, ReLU) → BatchNorm → MaxPooling2D (2×2)
    │
    ▼
Conv2D (64 filters, 3×3, ReLU) → BatchNorm → MaxPooling2D (2×2)
    │
    ▼
Flatten
    │
    ▼
Dense (128, ReLU) → Dropout (0.5)
    │
    ▼
Dense (10, Softmax)  →  Output: digit 0–9
```

| Layer | Details |
|-------|---------|
| Optimizer | Adam (lr=0.001) |
| Loss | Sparse Categorical Crossentropy |
| Metric | Accuracy |
| Epochs | 10 (default) |
| Batch size | 32 |
| Validation split | 10% |

**Production enhancements included:**
- Batch Normalization
- Data Augmentation
- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau (learning rate scheduler)
- TensorBoard logging

---

## Folder Structure

```
Handwritten_Digit_Recognition/
│
├── data/                    # Dataset cache (auto-downloaded by Keras)
├── models/                  # Saved models (cnn_model.keras, cnn_model.h5)
├── images/                  # Custom images for inference
├── notebooks/               # Jupyter notebook walkthrough
│   └── Digit_Recognition.ipynb
├── src/                     # Source code modules
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # Evaluation and metrics
│   ├── predict.py           # Custom image inference
│   └── utils.py             # Shared utilities
├── results/                 # Plots and reports
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
├── requirements.txt
├── README.md
└── main.py                  # Interactive menu entry point
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup

```bash
# Clone or navigate to the project directory
cd Handwritten_Digit_Recognition

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Requirements

| Package | Purpose |
|---------|---------|
| tensorflow | Deep learning framework |
| numpy | Numerical computing |
| pandas | Data manipulation |
| matplotlib | Plotting |
| seaborn | Statistical visualizations |
| scikit-learn | Metrics and evaluation |
| opencv-python | Image preprocessing for inference |
| jupyter | Interactive notebook |

---

## Usage

### Interactive Menu (Recommended)

```bash
python main.py
```

```
Main Menu:
  1. Train Model
  2. Evaluate Model
  3. Predict Image
  4. Exit
```

### Training

```bash
# Default training (10 epochs, batch size 32, with augmentation)
python -m src.train

# Custom parameters
python -m src.train --epochs 15 --batch-size 64 --no-augmentation
```

Training outputs:
- `models/cnn_model.keras` — Keras native format
- `models/cnn_model.h5` — HDF5 format (backward compatible)
- `results/accuracy_plot.png` — Training vs validation accuracy
- `results/loss_plot.png` — Training vs validation loss
- `results/tensorboard_logs/` — TensorBoard event files

View TensorBoard:

```bash
tensorboard --logdir results/tensorboard_logs
```

### Evaluation

```bash
python -m src.evaluate
```

Outputs:
- Test loss and accuracy printed to console
- Per-class precision, recall, and F1-score
- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/sample_predictions.png`

### Prediction on Custom Images

1. Place a handwritten digit image in the `images/` folder (PNG, JPG, etc.)
2. Run prediction:

```bash
# Single image
python -m src.predict --image images/my_digit.png

# Batch mode (all images in images/ folder)
python -m src.predict

# Show preprocessed image
python -m src.predict --image images/my_digit.png --show
```

Example output:

```
Predicted Digit: 7
Confidence: 99.2%
```

**Note:** Images are auto-inverted if they appear as dark text on a light background (MNIST uses white digits on black background).

### Jupyter Notebook

For a step-by-step interactive walkthrough:

```bash
jupyter notebook notebooks/Digit_Recognition.ipynb
```

---

## Results

Expected performance after 10 epochs:

| Metric | Expected Value |
|--------|----------------|
| Test Accuracy | 98–99% |
| Test Loss | < 0.10 |
| Training Time | ~2–5 min (CPU) / ~30–60 sec (GPU) |

Generated visualizations:

| File | Description |
|------|-------------|
| `accuracy_plot.png` | Training and validation accuracy over epochs |
| `loss_plot.png` | Training and validation loss over epochs |
| `confusion_matrix.png` | Heatmap of true vs predicted labels |
| `sample_predictions.png` | 25 test images with predictions |
| `model_architecture.png` | CNN layer diagram (requires graphviz) |

---

## Module Reference

| Module | Description |
|--------|-------------|
| `src/train.py` | Full training pipeline with callbacks and plot generation |
| `src/evaluate.py` | Test set evaluation, metrics, and visualizations |
| `src/predict.py` | OpenCV-based custom image inference |
| `src/utils.py` | Shared data loading, model building, and plotting utilities |
| `main.py` | Interactive CLI menu |

---

## Future Improvements

- [ ] Deploy model as a REST API (Flask/FastAPI)
- [ ] Build a web UI for real-time digit drawing and prediction
- [ ] Experiment with deeper architectures (ResNet, VGG-style blocks)
- [ ] Hyperparameter tuning with Keras Tuner
- [ ] Export to TensorFlow Lite for mobile deployment
- [ ] Support multi-digit image segmentation
- [ ] Add unit tests and CI/CD pipeline
- [ ] Docker containerization for reproducible environments

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No saved model found` | Run training first (`python -m src.train`) |
| Low accuracy on custom images | Ensure digit is centered, use dark-on-light or light-on-dark; script auto-inverts |
| TensorFlow GPU not detected | Install `tensorflow` GPU build or use CPU (works fine for MNIST) |
| Architecture diagram fails | Install [Graphviz](https://graphviz.org/) and `pydot` |

---

## License

This project is open source and available for educational purposes.

---

## Acknowledgments

- [MNIST Dataset](https://yann.lecun.com/exdb/mnist/) by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) teams
