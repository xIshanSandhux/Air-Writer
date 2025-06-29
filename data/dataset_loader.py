import scipy.io
import numpy as np
import tensorflow as tf

def load_emnist_dataset(batch_size=128):
    mat = scipy.io.loadmat("data/emnist-byclass.mat")

    # Unpack the structured numpy array
    train_struct = mat["dataset"]["train"][0, 0]
    test_struct = mat["dataset"]["test"][0, 0]

    # Extract images and labels
    train_images = train_struct["images"][0, 0]
    train_labels = train_struct["labels"][0, 0]
    test_images = test_struct["images"][0, 0]
    test_labels = test_struct["labels"][0, 0]

    print(f"Raw training images shape: {train_images.shape}")
    print(f"Raw training labels shape: {train_labels.shape}")

    # Reshape and reorient
    train_images = train_images.reshape((-1, 28, 28), order="F").transpose(0, 2, 1)
    test_images = test_images.reshape((-1, 28, 28), order="F").transpose(0, 2, 1)

    # Normalize and expand dims
    train_images = np.expand_dims(train_images.astype(np.float32) / 255.0, -1)
    test_images = np.expand_dims(test_images.astype(np.float32) / 255.0, -1)

    # Flatten labels
    train_labels = train_labels.flatten().astype(np.int32)
    test_labels = test_labels.flatten().astype(np.int32)

    print(f"Processed training images shape: {train_images.shape}")
    print(f"Processed test images shape: {test_images.shape}")
    print(f"Training labels range: {train_labels.min()} to {train_labels.max()}")

    # Wrap into tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

    print(" EMNIST dataset loaded successfully.")
    return train_ds, test_ds
