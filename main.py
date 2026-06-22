"""
Main entry point for the Handwritten Digit Recognition project.

Provides an interactive menu to:
1. Train the CNN model
2. Evaluate the model on the test set
3. Predict digits from custom images
"""

import sys
from pathlib import Path

# Add project root to Python path for module imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_model
from src.predict import predict_batch, predict_digit
from src.train import train_model
from src.utils import IMAGES_DIR, MODEL_KERAS_PATH, ensure_directories


def print_banner():
    """Display the project welcome banner."""
    banner = """
    ============================================================
    |   Handwritten Digit Recognition using CNN (MNIST)        |
    |   Deep Learning Image Classification System              |
    ============================================================
    """
    print(banner)


def print_menu():
    """Display the main menu options."""
    print("\nMain Menu:")
    print("  1. Train Model")
    print("  2. Evaluate Model")
    print("  3. Predict Image")
    print("  4. Exit")
    print("-" * 40)


def handle_train():
    """Menu option 1: Train the CNN model."""
    print("\n>>> Starting Model Training...")
    try:
        epochs_input = input("Enter number of epochs [default: 10]: ").strip()
        epochs = int(epochs_input) if epochs_input else 10

        batch_input = input("Enter batch size [default: 32]: ").strip()
        batch_size = int(batch_input) if batch_input else 32

        aug_input = input("Use data augmentation? [Y/n]: ").strip().lower()
        use_augmentation = aug_input != "n"

        train_model(
            epochs=epochs,
            batch_size=batch_size,
            use_augmentation=use_augmentation,
        )
    except ValueError as exc:
        print(f"Invalid input: {exc}")
    except Exception as exc:
        print(f"Training failed: {exc}")


def handle_evaluate():
    """Menu option 2: Evaluate the trained model."""
    print("\n>>> Starting Model Evaluation...")
    try:
        if not MODEL_KERAS_PATH.exists():
            print("No trained model found. Please train the model first (Option 1).")
            return

        results = evaluate_model()
        accuracy = results["test_accuracy"]
        print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

        if accuracy >= 0.98:
            print("Target accuracy (>98%) achieved!")
        else:
            print("Tip: Try training for more epochs or enabling data augmentation.")
    except Exception as exc:
        print(f"Evaluation failed: {exc}")


def handle_predict():
    """Menu option 3: Predict digit from a custom image."""
    print("\n>>> Custom Image Prediction")
    try:
        if not MODEL_KERAS_PATH.exists():
            print("No trained model found. Please train the model first (Option 1).")
            return

        print(f"\nDefault images folder: {IMAGES_DIR}")
        image_path = input("Enter image path (or press Enter for batch mode): ").strip()

        if image_path:
            # Single image prediction
            predict_digit(image_path, show_image=False)
        else:
            # Batch prediction on all images in images/ folder
            predict_batch()
    except Exception as exc:
        print(f"Prediction failed: {exc}")


def main():
    """Run the interactive main menu loop."""
    ensure_directories()
    print_banner()

    while True:
        print_menu()
        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            handle_train()
        elif choice == "2":
            handle_evaluate()
        elif choice == "3":
            handle_predict()
        elif choice == "4":
            print("\nThank you for using Handwritten Digit Recognition. Goodbye!")
            break
        else:
            print("Invalid option. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
