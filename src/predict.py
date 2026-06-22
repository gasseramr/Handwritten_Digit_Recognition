"""
Prediction script for custom handwritten digit images.

This module handles:
- Loading a saved CNN model
- Preprocessing custom images with OpenCV
- Running inference and reporting confidence scores
"""

import argparse
from pathlib import Path

import numpy as np

from src.utils import IMAGES_DIR, load_saved_model, preprocess_custom_image


def predict_digit(image_path, model_path=None, show_image=False):
    """
    Predict the digit in a custom image file.

    Parameters
    ----------
    image_path : str or Path
        Path to the input image (any common format supported by OpenCV).
    model_path : str or Path, optional
        Path to the saved model. Uses default if not provided.
    show_image : bool
        If True, display the preprocessed 28x28 image using matplotlib.

    Returns
    -------
    tuple
        (predicted_digit, confidence, all_probabilities)
    """
    image_path = Path(image_path)

    print("=" * 60)
    print("CUSTOM IMAGE PREDICTION")
    print("=" * 60)
    print(f"Input image: {image_path}")

    # Load the trained model
    model = load_saved_model(model_path)

    # Preprocess the image for CNN input
    processed_image = preprocess_custom_image(image_path)

    # Run forward pass to get class probabilities
    predictions = model.predict(processed_image, verbose=0)
    probabilities = predictions[0]

    # Select the class with the highest probability
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit]) * 100

    # Display results
    print("\n" + "-" * 40)
    print(f"Predicted Digit: {predicted_digit}")
    print(f"Confidence:      {confidence:.1f}%")
    print("-" * 40)

    # Show probability distribution across all 10 classes
    print("\nClass Probabilities:")
    for digit, prob in enumerate(probabilities):
        bar = "#" * int(prob * 40)
        print(f"  {digit}: {prob * 100:5.1f}% {bar}")

    # Optionally display the preprocessed image
    if show_image:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 4))
        plt.imshow(processed_image.squeeze(), cmap="gray")
        plt.title(f"Predicted: {predicted_digit} ({confidence:.1f}%)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return predicted_digit, confidence, probabilities


def predict_batch(image_dir=None, model_path=None):
    """
    Run prediction on all images in a directory.

    Parameters
    ----------
    image_dir : str or Path, optional
        Directory containing images. Defaults to project images/ folder.
    model_path : str or Path, optional
        Path to the saved model.
    """
    image_dir = Path(image_dir) if image_dir else IMAGES_DIR

    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        return

    # Supported image extensions
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    image_files = [
        f for f in image_dir.iterdir() if f.suffix.lower() in extensions
    ]

    if not image_files:
        print(f"No images found in: {image_dir}")
        print("Place custom digit images in the images/ folder and retry.")
        return

    print(f"Found {len(image_files)} image(s) in {image_dir}\n")

    for image_file in sorted(image_files):
        predict_digit(image_file, model_path=model_path)
        print()


def main():
    """Command-line entry point for prediction."""
    parser = argparse.ArgumentParser(
        description="Predict handwritten digits from custom images."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image file for prediction",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory of images to predict (batch mode)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the saved model file",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the preprocessed image",
    )
    args = parser.parse_args()

    if args.image:
        predict_digit(
            args.image, model_path=args.model_path, show_image=args.show
        )
    elif args.image_dir:
        predict_batch(image_dir=args.image_dir, model_path=args.model_path)
    else:
        predict_batch(model_path=args.model_path)


if __name__ == "__main__":
    main()
