"""
Training script for the MNIST Handwritten Digit Recognition CNN.

This module handles:
- Dataset loading and preprocessing
- Model building and compilation
- Training with callbacks (EarlyStopping, ModelCheckpoint, etc.)
- Saving training plots and the trained model
"""

import argparse

import numpy as np
import tensorflow as tf

from src.utils import (
    BATCH_SIZE,
    EPOCHS,
    MODEL_H5_PATH,
    MODEL_KERAS_PATH,
    RESULTS_DIR,
    VALIDATION_SPLIT,
    build_cnn_model,
    compile_model,
    ensure_directories,
    get_training_callbacks,
    load_mnist_data,
    plot_sample_images,
    plot_training_history,
    preprocess_data,
    save_model,
    visualize_model_architecture,
)


def train_model(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    use_augmentation=True,
    verbose=1,
):
    """
    Full training pipeline for the MNIST CNN.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size for gradient updates.
    validation_split : float
        Fraction of training data held out for validation.
    use_augmentation : bool
        Whether to apply data augmentation during training.
    verbose : int
        Keras training verbosity level.

    Returns
    -------
    tuple
        (trained_model, history, x_test, y_test)
    """
    # Ensure output directories exist
    ensure_directories()

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    print("=" * 60)
    print("STEP 1: Loading MNIST Dataset")
    print("=" * 60)

    # Load raw MNIST data
    x_train_raw, y_train, x_test_raw, y_test = load_mnist_data()

    # Display dataset information
    print(f"Training data shape:   {x_train_raw.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing data shape:    {x_test_raw.shape}")
    print(f"Testing labels shape:  {y_test.shape}")
    print(f"Number of classes:     {len(np.unique(y_train))}")
    print(f"Pixel value range:     [{x_train_raw.min()}, {x_train_raw.max()}]")

    # Visualize sample images before preprocessing
    plot_sample_images(
        x_train_raw,
        y_train,
        num_samples=10,
        save_path=RESULTS_DIR / "sample_training_images.png",
    )

    print("\n" + "=" * 60)
    print("STEP 2: Data Preprocessing")
    print("=" * 60)

    # Normalize and reshape for CNN input
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train_raw, y_train, x_test_raw, y_test
    )

    print(f"Normalized training shape: {x_train.shape}")
    print(f"Normalized testing shape:  {x_test.shape}")
    print(f"Label dtype:               {y_train.dtype}")
    print(f"Pixel value range (train): [{x_train.min():.2f}, {x_train.max():.2f}]")

    print("\n" + "=" * 60)
    print("STEP 3: Building CNN Model")
    print("=" * 60)

    # Build and compile the CNN (augmentation layers are training-only)
    model = build_cnn_model(use_augmentation=use_augmentation)
    model = compile_model(model)

    # Print layer-by-layer architecture summary
    model.summary()

    # Save architecture diagram (optional, requires graphviz)
    visualize_model_architecture(
        model, save_path=RESULTS_DIR / "model_architecture.png"
    )

    print("\n" + "=" * 60)
    print("STEP 4: Training Model")
    print("=" * 60)
    print(f"Epochs:            {epochs}")
    print(f"Batch size:        {batch_size}")
    print(f"Validation split:  {validation_split}")

    # Configure training callbacks
    callbacks = get_training_callbacks(model_keras_path=MODEL_KERAS_PATH)

    # Train the model; history stores metrics per epoch
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose,
    )

    # Report final epoch metrics
    final_epoch = len(history.history["accuracy"])
    print("\n" + "-" * 40)
    print(f"Training completed after {final_epoch} epoch(s)")
    print(f"Final Training Accuracy:   {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss:       {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss:     {history.history['val_loss'][-1]:.4f}")
    print("-" * 40)

    print("\n" + "=" * 60)
    print("STEP 5: Plotting Learning Curves")
    print("=" * 60)

    # Generate and save accuracy/loss plots
    plot_training_history(history)

    print("\n" + "=" * 60)
    print("STEP 6: Saving Model")
    print("=" * 60)

    # Save the trained model for inference (augmentation layers are inactive at predict time)
    save_model(model, keras_path=MODEL_KERAS_PATH, h5_path=MODEL_H5_PATH)

    print("\nTraining pipeline completed successfully!")
    return model, history, x_test, y_test


def main():
    """Command-line entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train a CNN on the MNIST handwritten digit dataset."
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Training batch size"
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable data augmentation during training",
    )
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_augmentation=not args.no_augmentation,
    )


if __name__ == "__main__":
    main()
