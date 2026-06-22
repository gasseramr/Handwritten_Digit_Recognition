"""
Evaluation script for the MNIST Handwritten Digit Recognition CNN.

This module handles:
- Loading a saved model
- Evaluating on the MNIST test set
- Generating classification reports and confusion matrices
- Visualizing sample predictions
"""

import argparse

import numpy as np
from sklearn.metrics import classification_report

from src.utils import (
    RESULTS_DIR,
    load_mnist_data,
    load_saved_model,
    plot_confusion_matrix,
    plot_sample_predictions,
    preprocess_data,
)


def evaluate_model(model_path=None, verbose=1):
    """
    Evaluate the trained CNN on the MNIST test set.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the saved model. Uses default if not provided.
    verbose : int
        Verbosity for model.evaluate().

    Returns
    -------
    dict
        Dictionary containing test metrics and predictions.
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Load the saved model from disk
    model = load_saved_model(model_path)

    print("\nLoading and preprocessing test data...")
    x_train, y_train, x_test_raw, y_test = load_mnist_data()
    _, _, x_test, y_test = preprocess_data(x_train, y_train, x_test_raw, y_test)

    print(f"Test set size: {len(x_test)} samples")

    # Evaluate overall loss and accuracy on the test set
    print("\n" + "-" * 40)
    print("Test Set Metrics")
    print("-" * 40)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=verbose)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # Generate predictions for all test samples
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Per-class precision, recall, and F1-score
    print("\n" + "-" * 40)
    print("Classification Report")
    print("-" * 40)
    report = classification_report(
        y_test,
        y_pred,
        target_names=[str(i) for i in range(10)],
        digits=4,
    )
    print(report)

    # Save classification report to a text file
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("MNIST CNN Classification Report\n")
        file.write("=" * 40 + "\n")
        file.write(f"Test Loss: {test_loss:.4f}\n")
        file.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
        file.write(report)
    print(f"Classification report saved to: {report_path}")

    # Confusion matrix heatmap
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        y_test,
        y_pred,
        save_path=RESULTS_DIR / "confusion_matrix.png",
    )

    # Visualize 25 random predictions
    print("Generating sample prediction visualization...")
    plot_sample_predictions(
        x_test,
        y_test,
        y_pred,
        num_samples=25,
        save_path=RESULTS_DIR / "sample_predictions.png",
    )

    # Count misclassifications
    misclassified = np.sum(y_pred != y_test)
    print(f"\nMisclassified samples: {misclassified} / {len(y_test)}")

    print("\nEvaluation completed successfully!")

    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_pred_probs": y_pred_probs,
    }


def main():
    """Command-line entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNN on the MNIST test set."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the saved model file",
    )
    args = parser.parse_args()

    evaluate_model(model_path=args.model_path)


if __name__ == "__main__":
    main()
