import tensorflow as tf
import os

# Data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomBrightness(0.1),
])

def load_data(batch_size):
    # Path to your dataset
    dataset_path = "C:/Users/User/OneDrive/Desktop/DATASETS/malaysian_food_processed/"
    
    # Check if the directory exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist.")

    # Load training dataset with augmentation and normalization
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(dataset_path, "train"),
        labels='inferred',
        label_mode='int',
        image_size=(224, 224),
        batch_size=batch_size,
        seed=123,
        shuffle=True
    ).map(lambda x, y: (data_augmentation(x) / 255.0, y))  # Normalize images to [0, 1]

    # Load validation dataset with normalization (no augmentation)
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(dataset_path, "validation"),
        labels='inferred',
        label_mode='int',
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=False
    ).map(lambda x, y: (x / 255.0, y))  # Normalize images to [0, 1]

    # Load test dataset with normalization (no augmentation)
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(dataset_path, "test"),  # Assuming your test set is here
        labels='inferred',
        label_mode='int',
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=False
    ).map(lambda x, y: (x / 255.0, y))  # Normalize images to [0, 1]

    # Prefetching and caching for performance optimization
    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
