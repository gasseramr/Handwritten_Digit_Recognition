"""
Shared utility functions for the Handwritten Digit Recognition project.

This module centralizes dataset loading, preprocessing, model architecture,
and plotting helpers so that train.py, evaluate.py, and predict.py stay modular.
"""

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# ---------------------------------------------------------------------------
# Project paths (resolved relative to the project root directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Default model filenames
MODEL_KERAS_PATH = MODELS_DIR / "cnn_model.keras"
MODEL_H5_PATH = MODELS_DIR / "cnn_model.h5"

# Training hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
NUM_CLASSES = 10
IMAGE_SHAPE = (28, 28, 1)

# Class labels for digits 0-9
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]


def ensure_directories() -> None:
    """Create all required project directories if they do not already exist."""
    for directory in (DATA_DIR, MODELS_DIR, IMAGES_DIR, RESULTS_DIR, NOTEBOOKS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_mnist_data():
    """
    Load the MNIST dataset from TensorFlow/Keras.

    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test) with raw uint8 pixel values.
    """
    # Download (if needed) and load the official MNIST split
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test


def preprocess_data(x_train, y_train, x_test, y_test):
    """
    Normalize pixel values and reshape images for CNN input.

    Steps:
    1. Scale pixels from [0, 255] to [0.0, 1.0]
    2. Add a channel dimension: (28, 28) -> (28, 28, 1)
    3. Labels remain integers (0-9) for SparseCategoricalCrossentropy

    Returns
    -------
    tuple
        Preprocessed (X_train, y_train, X_test, y_test).
    """
    # Normalize to [0, 1] for faster and more stable training
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # CNN expects shape (height, width, channels)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return x_train, y_train, x_test, y_test


def build_cnn_model(
    input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES, use_augmentation=False
):
    """
    Build a Convolutional Neural Network for digit classification.

    Architecture:
    - Conv2D(32) + MaxPooling2D
    - Conv2D(64) + MaxPooling2D
    - Flatten -> Dense(128) -> Dropout(0.5) -> Dense(10, softmax)

    Optional improvements (production enhancements):
    - BatchNormalization after conv layers
    - Data augmentation pipeline (applied during training)

    Returns
    -------
    keras.Model
        Compiled-ready Sequential CNN model.
    """
    model_layers = []

    # Optional data augmentation (active only during training)
    if use_augmentation:
        model_layers.extend(
            [
                layers.RandomRotation(factor=0.05, name="random_rotation"),
                layers.RandomTranslation(
                    height_factor=0.1,
                    width_factor=0.1,
                    name="random_translation",
                ),
                layers.RandomZoom(
                    height_factor=0.1, width_factor=0.1, name="random_zoom"
                ),
            ]
        )

    model_layers.extend(
        [
            # Input layer: 28x28 grayscale images
            layers.Input(shape=input_shape, name="input"),

            # Block 1: extract low-level features (edges, strokes)
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                name="conv2d_1",
            ),
            layers.BatchNormalization(name="batch_norm_1"),
            layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_1"),

            # Block 2: extract higher-level patterns
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                name="conv2d_2",
            ),
            layers.BatchNormalization(name="batch_norm_2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_2"),

            # Fully connected classifier head
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="dense_hidden"),
            layers.Dropout(0.5, name="dropout"),
            layers.Dense(num_classes, activation="softmax", name="output"),
        ]
    )

    model = keras.Sequential(model_layers, name="mnist_cnn")
    return model


def compile_model(model):
    """
    Compile the model with Adam optimizer and sparse categorical cross-entropy.

    SparseCategoricalCrossentropy is used because labels are integers (0-9),
    not one-hot encoded vectors.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def get_data_augmentation():
    """
    Create a lightweight data augmentation pipeline.

    Augmentation helps the model generalize by exposing it to small
    transformations during training (rotation, shift, zoom).
    """
    return keras.Sequential(
        [
            layers.RandomRotation(factor=0.05, name="random_rotation"),
            layers.RandomTranslation(
                height_factor=0.1, width_factor=0.1, name="random_translation"
            ),
            layers.RandomZoom(height_factor=0.1, width_factor=0.1, name="random_zoom"),
        ],
        name="data_augmentation",
    )


def get_training_callbacks(model_keras_path=MODEL_KERAS_PATH):
    """
    Return Keras callbacks for robust training.

    Includes:
    - EarlyStopping: halt training when validation loss stops improving
    - ModelCheckpoint: save the best model weights
    - ReduceLROnPlateau: reduce learning rate on plateau
    - TensorBoard: log metrics for visualization
    """
    ensure_directories()

    log_dir = RESULTS_DIR / "tensorboard_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_keras_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
        ),
    ]


def plot_sample_images(x_data, y_data, num_samples=10, save_path=None):
    """
    Visualize sample MNIST images with their true labels.

    Parameters
    ----------
    x_data : np.ndarray
        Image array, shape (N, 28, 28) or (N, 28, 28, 1).
    y_data : np.ndarray
        Integer labels.
    num_samples : int
        Number of images to display.
    save_path : Path or str, optional
        If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for idx in range(min(num_samples, len(x_data))):
        # Squeeze channel dimension for grayscale display
        image = x_data[idx].squeeze()
        axes[idx].imshow(image, cmap="gray")
        axes[idx].set_title(f"Label: {y_data[idx]}")
        axes[idx].axis("off")

    plt.suptitle("Sample MNIST Training Images", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Sample images saved to: {save_path}")

    plt.close(fig)


def plot_training_history(history, results_dir=RESULTS_DIR):
    """
    Plot and save accuracy and loss curves from training history.

    Saves:
    - results/accuracy_plot.png
    - results/loss_plot.png
    """
    ensure_directories()
    results_dir = Path(results_dir)

    # --- Accuracy plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    plt.title("Model Accuracy vs. Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    accuracy_path = results_dir / "accuracy_plot.png"
    plt.savefig(accuracy_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Accuracy plot saved to: {accuracy_path}")

    # --- Loss plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("Model Loss vs. Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_path = results_dir / "loss_plot.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss plot saved to: {loss_path}")


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Create a seaborn heatmap of the confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    save_path : Path or str, optional
        Path to save the figure.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    plt.close()


def plot_sample_predictions(x_test, y_test, y_pred, num_samples=25, save_path=None):
    """
    Visualize predictions alongside true labels for random test samples.

    Correct predictions are shown with a green title; incorrect ones in red.
    """
    indices = np.random.choice(len(x_test), size=num_samples, replace=False)
    rows = 5
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(14, 14))
    axes = axes.flatten()

    for plot_idx, data_idx in enumerate(indices):
        image = x_test[data_idx].squeeze()
        true_label = y_test[data_idx]
        pred_label = y_pred[data_idx]
        color = "green" if true_label == pred_label else "red"

        axes[plot_idx].imshow(image, cmap="gray")
        axes[plot_idx].set_title(
            f"True: {true_label} | Pred: {pred_label}",
            color=color,
            fontsize=10,
        )
        axes[plot_idx].axis("off")

    plt.suptitle(
        "Sample Predictions (Green = Correct, Red = Incorrect)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Sample predictions saved to: {save_path}")

    plt.close(fig)


def save_model(model, keras_path=MODEL_KERAS_PATH, h5_path=MODEL_H5_PATH):
    """
    Save the trained model in both .keras and .h5 formats.

    The .keras format is the recommended Keras 3 native format.
    The .h5 format is provided for backward compatibility.
    """
    ensure_directories()
    model.save(str(keras_path))
    model.save(str(h5_path))
    print(f"Model saved to: {keras_path}")
    print(f"Model saved to: {h5_path}")


def load_saved_model(model_path=None):
    """
    Load a saved Keras model from disk.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the model file. Defaults to MODEL_KERAS_PATH.

    Returns
    -------
    keras.Model
        Loaded model ready for evaluation or inference.
    """
    if model_path is None:
        model_path = MODEL_KERAS_PATH

    model_path = Path(model_path)
    if not model_path.exists():
        # Fall back to .h5 if .keras is missing
        if MODEL_H5_PATH.exists():
            model_path = MODEL_H5_PATH
        else:
            raise FileNotFoundError(
                f"No saved model found at {MODEL_KERAS_PATH} or {MODEL_H5_PATH}. "
                "Please train the model first."
            )

    print(f"Loading model from: {model_path}")
    return keras.models.load_model(str(model_path))


def preprocess_custom_image(image_path):
    """
    Load and preprocess a custom image for inference.

    Steps:
    1. Read image with OpenCV
    2. Convert to grayscale
    3. Resize to 28x28
    4. Normalize pixel values to [0, 1]
    5. Reshape to (1, 28, 28, 1)

    Parameters
    ----------
    image_path : str or Path
        Path to the input image file.

    Returns
    -------
    np.ndarray
        Preprocessed image batch of shape (1, 28, 28, 1).
    """
    image_path = str(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image in BGR format (OpenCV default)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Convert to single-channel grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to MNIST-compatible dimensions
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # MNIST digits are white on black background; invert if image is dark-on-light
    if np.mean(resized) > 127:
        resized = 255 - resized

    # Normalize and add batch + channel dimensions
    normalized = resized.astype("float32") / 255.0
    processed = normalized.reshape(1, 28, 28, 1)

    return processed


def visualize_model_architecture(model, save_path=None):
    """
    Save a visual diagram of the CNN architecture using Keras plot_model.

    Requires pydot and graphviz to be installed for image export.
    """
    try:
        keras.utils.plot_model(
            model,
            to_file=str(save_path) if save_path else str(RESULTS_DIR / "model_architecture.png"),
            show_shapes=True,
            show_layer_names=True,
            dpi=150,
        )
        print(f"Model architecture diagram saved.")
    except Exception as exc:
        print(f"Could not generate architecture diagram: {exc}")
        print("Install graphviz and pydot for architecture visualization.")
