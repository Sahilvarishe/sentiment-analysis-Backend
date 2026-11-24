"""Run emotion prediction for the largest face found in each webcam frame."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=BASE_DIR / "models" / "resnet50_emotion_best.keras",
        help="Path to the trained Keras model.",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera device index.")
    args = parser.parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Model file not found: {args.model}")

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    if face_detector.empty():
        raise SystemExit(f"Could not load Haar Cascade: {cascade_path}")

    model = tf.keras.models.load_model(args.model)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open webcam device {args.camera}.")

    print("Webcam is using the largest detected face. Press q in the video window to quit.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the webcam.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            if len(faces) > 0:
                x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
                face_bgr = frame[y:y + height, x:x + width]
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                face_rgb = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_AREA)
                model_input = tf.keras.applications.resnet50.preprocess_input(
                    np.asarray(face_rgb, dtype=np.float32)
                )
                probabilities = model.predict(model_input[None, ...], verbose=0)[0]
                emotion_index = int(np.argmax(probabilities))
                emotion = CLASS_NAMES[emotion_index]
                confidence = float(probabilities[emotion_index])

                cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 200, 80), 2)
                cv2.putText(
                    frame,
                    f"Emotion: {emotion.capitalize()} ({confidence:.0%})",
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (40, 200, 80),
                    2,
                )
                if len(faces) > 1:
                    cv2.putText(
                        frame,
                        f"Using largest face ({len(faces)} detected)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Facial Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()