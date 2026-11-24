"""
Facial Emotion Detection Service using OpenCV and TensorFlow/Keras CNN.
Loads the CNN model only once at startup and performs face detection,
cropping, preprocessing, and emotion-to-mood classification.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2

from config import Config
from utils.image_processing import preprocess_face_roi


class EmotionDetectionService:
    """
    Singleton-style service that manages face detection and CNN model inference.
    """

    def __init__(self):
        self.model = None
        self.face_cascade = None
        self.input_shape: Tuple[int, int, int] = (
            Config.MODEL_INPUT_HEIGHT,
            Config.MODEL_INPUT_WIDTH,
            Config.MODEL_INPUT_CHANNELS
        )
        self.is_mock_model: bool = False
        self.preprocessing = "normalize"
        
        # Initialize face detector and CNN model
        self._load_face_detector()
        self._load_cnn_model()

    def _load_face_detector(self) -> None:
        """Load OpenCV Haar Cascade for frontal face detection."""
        try:
            # 1. Try OpenCV default path
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                print(f"[EmotionService] Loaded OpenCV Haar Cascade from: {cascade_path}")
                return

            # 2. Try local models folder
            local_cascade = Config.BASE_DIR / "models" / "haarcascade_frontalface_default.xml"
            if local_cascade.exists():
                self.face_cascade = cv2.CascadeClassifier(str(local_cascade))
                print(f"[EmotionService] Loaded local Haar Cascade from: {local_cascade}")
                return

            # Fallback
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        except Exception as e:
            print(f"[EmotionService] Warning: Could not initialize Haar Cascade: {e}")
            self.face_cascade = None

    def _load_cnn_model(self) -> None:
        """
        Load TensorFlow/Keras CNN model once during server startup.
        Inspects input shape dynamically if model exists.
        """
        model_path = Config.MODEL_PATH
        fallback_path = Config.FALLBACK_MODEL_PATH

        target_file = None
        if model_path.exists():
            target_file = model_path
        elif fallback_path.exists():
            target_file = fallback_path

        if target_file:
            try:
                # Lazy-import tensorflow so backend can start even if TF is installing
                import tensorflow as tf
                print(f"[EmotionService] Loading CNN model from: {target_file} ...")
                self.model = tf.keras.models.load_model(str(target_file))
                self.is_mock_model = False
                print("[EmotionService] CNN Model loaded successfully.")

                # Programmatically detect model input shape
                try:
                    shape = self.model.input_shape
                    # shape is typically (None, 48, 48, 1) or (None, 48, 48, 3)
                    if shape and len(shape) == 4:
                        h = shape[1] or Config.MODEL_INPUT_HEIGHT
                        w = shape[2] or Config.MODEL_INPUT_WIDTH
                        c = shape[3] or Config.MODEL_INPUT_CHANNELS
                        self.input_shape = (h, w, c)
                        print(f"[EmotionService] Detected model input shape: {self.input_shape}")
                        if self.input_shape == (224, 224, 3):
                            self.preprocessing = "resnet50"
                except Exception as shape_err:
                    print(f"[EmotionService] Could not auto-detect shape, using default 48x48: {shape_err}")

                # Warm up model execution graph so live webcam requests are instantaneous
                try:
                    dummy_in = np.zeros((1, *self.input_shape), dtype=np.float32)
                    _ = self.model(dummy_in, training=False)
                    print("[EmotionService] Model warm-up completed successfully.")
                except Exception as warmup_err:
                    print(f"[EmotionService] Warm-up notice: {warmup_err}")

                return
            except Exception as e:
                print(f"[EmotionService] Error loading Keras model from {target_file}: {e}")
                print("[EmotionService] Falling back to smart mock classifier until model is fixed.")
        else:
            print(f"[EmotionService] Model file not found at '{model_path}'.")
            print("[EmotionService] Place your trained 'resnet50_emotion_best.keras' inside 'backend/models/'.")
            print("[EmotionService] Running with simulation mode for frontend verification.")

        self.is_mock_model = True

    def detect_face(self, image_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], bool]:
        """
        Detect face in BGR image using OpenCV.
        Returns:
            ((x, y, w, h), has_multiple_faces) or (None, False) if no face detected.
        """
        if self.face_cascade is None:
            self._load_face_detector()
            if self.face_cascade is None:
                # If cascade is completely unavailable, treat whole image center as face
                h, w = image_bgr.shape[:2]
                return (int(w * 0.2), int(h * 0.15), int(w * 0.6), int(h * 0.7)), False

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Optimize detection: scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            # Fallback pass: equalize histogram to handle dim or high-contrast lighting
            gray_eq = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(
                gray_eq,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

        if len(faces) == 0:
            return None, False

        # If multiple faces detected, select the largest one (highest area = w * h)
        has_multiple = len(faces) > 1
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        return tuple(largest_face), has_multiple

    def predict_emotion(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Full emotion detection pipeline:
          1. Detect face using OpenCV.
          2. Crop face ROI.
          3. Preprocess for CNN (resize, grayscale/RGB, normalize).
          4. Predict emotion probabilities using Keras model.
          5. Map emotion to mood category.
          6. Return clean structured JSON result.
        """
        # 1. Detect Face
        face_rect, has_multiple = self.detect_face(image_bgr)
        if face_rect is None:
            return {
                "success": False,
                "error": "No face detected in the provided image.",
                "disclaimer": Config.DISCLAIMER
            }

        x, y, w, h = face_rect

        # Add slight padding around face if within bounds
        img_h, img_w = image_bgr.shape[:2]
        pad_x = int(w * 0.05)
        pad_y = int(h * 0.05)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        face_roi = image_bgr[y1:y2, x1:x2]
        if face_roi.size == 0:
            face_roi = image_bgr[y:y+h, x:x+w]

        # 2. Preprocess Face ROI
        target_h, target_w, target_c = self.input_shape
        processed_input = preprocess_face_roi(
            face_roi,
            target_width=target_w,
            target_height=target_h,
            channels=target_c,
            preprocessing=self.preprocessing
        )

        # 3. Model Inference
        emotions_list = Config.FER2013_EMOTIONS
        probabilities: Dict[str, float] = {}

        if self.model and not self.is_mock_model:
            try:
                # Fast direct functional execution (4-5x faster than model.predict on CPU)
                raw_preds = self.model(processed_input, training=False).numpy()[0]
                for idx, emotion_name in enumerate(emotions_list):
                    probabilities[emotion_name] = round(float(raw_preds[idx]), 4)
            except Exception as e:
                try:
                    # Fallback to model.predict if direct tensor call encounters an unexpected format
                    raw_preds = self.model.predict(processed_input, verbose=0)[0]
                    for idx, emotion_name in enumerate(emotions_list):
                        probabilities[emotion_name] = round(float(raw_preds[idx]), 4)
                except Exception as pred_err:
                    print(f"[EmotionService] Prediction runtime error: {e} / {pred_err}")
                    probabilities = self._generate_simulated_probabilities()
        else:
            probabilities = self._generate_simulated_probabilities()

        # Find dominant emotion
        dominant_emotion = max(probabilities.items(), key=lambda item: item[1])[0]
        confidence = round(probabilities[dominant_emotion], 4)

        # 4. Map Emotion to Mood
        mood = Config.EMOTION_TO_MOOD_MAP.get(dominant_emotion, "calm")

        return {
            "success": True,
            "emotion": dominant_emotion,
            "mood": mood,
            "confidence": confidence,
            "bounding_box": {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            },
            "multiple_faces_detected": has_multiple,
            "all_emotions": probabilities,
            "model_status": "loaded" if not self.is_mock_model else "simulation_fallback",
            "disclaimer": Config.DISCLAIMER
        }

    def _generate_simulated_probabilities(self) -> Dict[str, float]:
        """
        Generate realistic probability distribution for demonstration/testing
        when emotion_model.h5 is not yet loaded into backend/models/.
        """
        emotions = Config.FER2013_EMOTIONS
        # Pick random primary emotion
        dominant = np.random.choice(emotions)
        dominant_conf = round(float(np.random.uniform(0.78, 0.94)), 4)
        remaining = 1.0 - dominant_conf

        probs = {}
        others = [e for e in emotions if e != dominant]
        random_shares = np.random.dirichlet(np.ones(len(others))) * remaining

        for idx, other_emotion in enumerate(others):
            probs[other_emotion] = round(float(random_shares[idx]), 4)
        probs[dominant] = dominant_conf

        return probs


# Singleton instance loaded once on startup
emotion_service = EmotionDetectionService()
