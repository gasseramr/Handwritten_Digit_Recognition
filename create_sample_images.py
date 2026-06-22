"""
Generate sample digit images from the MNIST test set for prediction testing.

Run once after setup to populate the images/ folder with example digits.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.datasets import mnist

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def export_sample_digits(num_samples=10):
    """Save random MNIST test images as PNG files for predict.py testing."""
    (_, _), (x_test, y_test) = mnist.load_data()

    indices = np.random.choice(len(x_test), size=num_samples, replace=False)

    for idx in indices:
        image = x_test[idx]
        label = y_test[idx]
        filename = IMAGES_DIR / f"mnist_digit_{label}_{idx}.png"
        cv2.imwrite(str(filename), image)

    print(f"Exported {num_samples} sample images to {IMAGES_DIR}")


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    export_sample_digits(count)
