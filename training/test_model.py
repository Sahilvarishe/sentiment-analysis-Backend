"""Predict facial emotion probabilities for one image using a saved model."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to an input image.")
    parser.add_argument(
        "--model",
        type=Path,
        default=BASE_DIR / "models" / "resnet50_emotion_best.keras",
        help="Path to the trained Keras model.",
    )
    args = parser.parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image file not found: {args.image}")
    if not args.model.is_file():
        raise SystemExit(f"Model file not found: {args.model}")

    try:
        with Image.open(args.image) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").resize((224, 224))
    except Exception as error:
        raise SystemExit(f"Could not open image: {error}") from error

    pixels = np.asarray(image, dtype=np.float32)
    model_input = tf.keras.applications.resnet50.preprocess_input(pixels)
    model = tf.keras.models.load_model(args.model)
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise SystemExit(f"Expected {len(CLASS_NAMES)} model outputs, got {model.output_shape[-1]}.")
    probabilities = model.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    print(f"Predicted Emotion: {CLASS_NAMES[predicted_index].capitalize()}")
    print(f"Confidence: {probabilities[predicted_index] * 100:.2f}%")
    print("\nEmotion Probabilities:")
    for name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{name.capitalize():<10}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()