"""
Helper Script to Generate a Standard FER2013 CNN Model for Testing.
Run this script to create 'backend/models/emotion_model.h5' with standard architecture:
Input (48, 48, 1) -> Conv2D -> MaxPool -> Dropout -> Dense -> Softmax (7 classes).
"""

from pathlib import Path

def create_model():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        print("[Error] TensorFlow is required to generate the model. Please install tensorflow first.")
        return

    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax', name='fer2013_output')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    output_path = Path(__file__).resolve().parent / "emotion_model.h5"
    model.save(str(output_path))
    print(f"[Success] Generated standard FER2013 CNN model saved to:\n  -> {output_path}")
    print("Model Input Shape:", model.input_shape)
    print("Model Output Shape:", model.output_shape)

if __name__ == "__main__":
    create_model()
