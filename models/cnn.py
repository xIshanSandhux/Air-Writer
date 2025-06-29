from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import sys
import os
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from data.dataset_loader import load_emnist_dataset

def create_emnist_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(62, activation='softmax')  # 62 classes for ByClass split
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("Loading EMNIST dataset...")
    train_ds, test_ds = load_emnist_dataset()

    print("Creating model...")
    model = create_emnist_model()

    print("Training model...")
    model.fit(train_ds, validation_data=test_ds, epochs=5)

    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/emnist_cnn_model.h5")
    print("Model saved at: saved_models/emnist_cnn_model.h5")

if __name__ == "__main__":
    main()
