import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from pathlib import Path

def generate_dataset(base_dir, image_size, batch_size):
    """
    Loads train, validation, and test datasets from a directory.

    Parameters:
    base_dir (str or Path): Base directory containing 'train', 'validation', and 'test' subdirectories.
    image_size (tuple): Size to which each image will be resized.
    batch_size (int): Number of images to be included in each batch.

    Returns:
    train_dataset (tf.data.Dataset): Training dataset.
    validation_dataset (tf.data.Dataset): Validation dataset.
    test_dataset (tf.data.Dataset): Test dataset.
    """
    base_dir = Path(base_dir)

    train_dataset = image_dataset_from_directory(
        base_dir / "train",
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
    )

    validation_dataset = image_dataset_from_directory(
        base_dir / "validation",
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
    )

    test_dataset = image_dataset_from_directory(
        base_dir / "test",
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
    )

    return train_dataset, validation_dataset, test_dataset