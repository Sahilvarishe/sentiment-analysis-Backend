"""Train a seven-class facial emotion classifier using ImageNet ResNet50."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Dataset"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def count_images(split_dir: Path) -> list[int]:
    """Count supported image files in each class directory."""
    counts = []
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Required class directory is missing: {class_dir}")
        count = sum(
            1 for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if count == 0:
            raise ValueError(f"No supported images found in {class_dir}")
        counts.append(count)
    return counts


def make_dataset(split_dir: Path, subset: str | None = None, shuffle: bool = False):
    """Load images as RGB float tensors with stable, explicit label ordering."""
    return tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
        validation_split=0.2 if subset else None,
        subset=subset,
    )


def prepare_dataset(dataset, augmentation=None):
    """Augment training pixels before applying the ResNet50 input transform."""

    def transform(images, labels):
        if augmentation is not None:
            images = augmentation(images, training=True)
        images = tf.keras.applications.resnet50.preprocess_input(images)
        return images, labels

    return dataset.map(transform, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def make_augmentation():
    """Apply mild geometric and contrast changes without distorting faces."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04, fill_mode="reflect"),
            tf.keras.layers.RandomZoom(0.08, fill_mode="reflect"),
            tf.keras.layers.RandomTranslation(0.04, 0.04, fill_mode="reflect"),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="facial_augmentation",
    )


def build_model():
    """Build the requested ResNet50 transfer-learning classification head."""
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3), name="preprocessed_rgb_image")
    features = base_model(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(0.4)(features)
    features = tf.keras.layers.Dense(256, activation="relu")(features)
    features = tf.keras.layers.Dropout(0.3)(features)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="emotion_probabilities")(features)
    model = tf.keras.Model(inputs, outputs, name="resnet50_facial_emotion")
    return model, base_model


def compile_model(model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def save_training_graphs(history: dict[str, list[float]]) -> None:
    epochs = range(1, len(history["accuracy"]) + 1)
    for metric, title, filename, ylabel in (
        ("accuracy", "Training and Validation Accuracy", "training_accuracy.png", "Accuracy"),
        ("loss", "Training and Validation Loss", "training_loss.png", "Loss"),
    ):
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history[metric], label=f"Training {metric}")
        plt.plot(epochs, history[f"val_{metric}"], label=f"Validation {metric}")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / filename, dpi=160)
        plt.close()


def main() -> None:
    train_dir = DATA_DIR / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    counts = count_images(train_dir)
    print("Class distribution\n------------------")
    distribution = pd.DataFrame({"emotion": CLASS_NAMES, "images": counts})
    for row in distribution.itertuples(index=False):
        print(f"{row.emotion:<10}: {row.images}")

    ratio = max(counts) / min(counts)
    class_weights = None
    if ratio >= 1.5:
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(CLASS_NAMES)),
            y=np.repeat(np.arange(len(CLASS_NAMES)), counts),
        )
        class_weights = {index: float(weight) for index, weight in enumerate(weights)}
        print(f"Class imbalance detected (largest/smallest = {ratio:.2f}); using balanced class weights.")
    else:
        print("No significant class imbalance detected; class weights are not needed.")

    training_raw = make_dataset(train_dir, subset="training", shuffle=True)
    validation_raw = make_dataset(train_dir, subset="validation", shuffle=True)
    if training_raw.class_names != CLASS_NAMES or validation_raw.class_names != CLASS_NAMES:
        raise RuntimeError("Dataset label order differs from the model's fixed class order.")

    augmentation = make_augmentation()
    training_data = prepare_dataset(training_raw, augmentation=augmentation)
    validation_data = prepare_dataset(validation_raw)
    model, base_model = build_model()
    model.summary()

    best_model_path = MODEL_DIR / "resnet50_emotion_best.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    print("\nStage 1: feature extraction (ResNet50 frozen)")
    compile_model(model, learning_rate=1e-3)
    stage_one = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=15,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            checkpoint,
        ],
    )

    print("\nStage 2: fine-tuning the last 25% of ResNet50 layers")
    base_model.trainable = True
    fine_tune_at = int(len(base_model.layers) * 0.75)
    for index, layer in enumerate(base_model.layers):
        layer.trainable = index >= fine_tune_at and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
    compile_model(model, learning_rate=1e-5)
    stage_two = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=20,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            checkpoint,
        ],
    )

    combined_history = {
        key: stage_one.history.get(key, []) + stage_two.history.get(key, [])
        for key in ("accuracy", "val_accuracy", "loss", "val_loss")
    }
    save_training_graphs(combined_history)
    final_model_path = MODEL_DIR / "resnet50_emotion_final.keras"
    model.save(final_model_path)
    print(f"Best validation-accuracy model: {best_model_path}")
    print(f"Final model: {final_model_path}")
    print(f"Training graphs saved under: {RESULTS_DIR}")
    print("The test split was not loaded or used during training. Run evaluate_model.py for final evaluation.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Training setup error: {error}") from error