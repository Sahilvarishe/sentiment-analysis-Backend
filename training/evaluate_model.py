"""Evaluate a saved emotion model on the held-out test split only."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


BASE_DIR = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
AUTOTUNE = tf.data.AUTOTUNE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=BASE_DIR / "models" / "resnet50_emotion_best.keras",
        help="Path to the trained Keras model.",
    )
    args = parser.parse_args()
    test_dir = BASE_DIR / "Dataset" / "test"
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if not args.model.is_file():
        raise SystemExit(f"Model file not found: {args.model}")
    if not test_dir.is_dir():
        raise SystemExit(f"Test dataset directory not found: {test_dir}")

    test_data = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=(224, 224),
        batch_size=32,
        shuffle=False,
    )
    if test_data.class_names != CLASS_NAMES:
        raise SystemExit("Test dataset class order does not match the model class order.")

    def resnet_preprocess(images, labels):
        return tf.keras.applications.resnet50.preprocess_input(images), labels

    test_data = test_data.map(resnet_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    model = tf.keras.models.load_model(args.model)
    test_metrics = model.evaluate(test_data, verbose=1, return_dict=True)
    predictions = np.argmax(model.predict(test_data, verbose=1), axis=1)
    true_labels = np.concatenate([labels.numpy() for _, labels in test_data], axis=0)

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        true_labels, predictions, labels=np.arange(len(CLASS_NAMES)), average="weighted", zero_division=0
    )
    report = classification_report(
        true_labels,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    report_text = (
        f"Test loss: {test_metrics['loss']:.4f}\n"
        f"Test accuracy: {test_metrics['accuracy']:.4f}\n"
        f"Weighted precision: {precision:.4f}\n"
        f"Weighted recall: {recall:.4f}\n"
        f"Weighted F1-score: {f1_score:.4f}\n\n"
        f"Classification report\n{report}"
    )
    print(report_text)
    (results_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    matrix = confusion_matrix(true_labels, predictions, labels=np.arange(len(CLASS_NAMES)))
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted emotion",
        ylabel="True emotion",
        title="Emotion Classification Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(results_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    print(f"Evaluation artifacts saved under: {results_dir}")


if __name__ == "__main__":
    main()