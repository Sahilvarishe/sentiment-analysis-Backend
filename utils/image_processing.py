"""
Image Processing Utilities for Facial Emotion Recognition.
Handles in-memory image decoding, format validation, and preprocessing for CNN models.
No user images are permanently stored on disk.
"""

import base64
import io
import re
from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from config import Config


def is_allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed image extension."""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in Config.ALLOWED_EXTENSIONS


def decode_image_from_bytes(file_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode raw byte array into an OpenCV BGR image numpy array in-memory.
    Returns None if decoding fails.
    """
    try:
        # Convert bytes to uint8 numpy array and decode using OpenCV
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None and img.size > 0:
            return img

        # Fallback to Pillow if OpenCV fails
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = pil_img.convert("RGB")
        img_np = np.array(pil_img)
        # Convert RGB to BGR for OpenCV consistency
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[ImageProcessing] Error decoding image bytes: {e}")
        return None


def decode_base64_image(base64_str: str) -> Optional[np.ndarray]:
    """
    Decode Base64 Data URL or raw Base64 string into an OpenCV BGR image.
    Handles 'data:image/...;base64,' headers automatically.
    """
    try:
        # Remove data URI prefix if present
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]

        # Clean any whitespace or newlines
        base64_str = re.sub(r"\s+", "", base64_str)
        img_bytes = base64.b64decode(base64_str)
        return decode_image_from_bytes(img_bytes)
    except Exception as e:
        print(f"[ImageProcessing] Error decoding base64 image: {e}")
        return None


def preprocess_face_roi(
    face_roi: np.ndarray,
    target_width: int = 48,
    target_height: int = 48,
    channels: int = 1,
    preprocessing: str = "normalize"
) -> np.ndarray:
    """
    Preprocess cropped face Region of Interest (ROI) for CNN input.
    
    Steps:
    1. Convert to the requested channel format (BGR input to RGB for 3 channels).
    2. Resize to the model's expected dimensions.
    3. Apply ResNet50 preprocessing or [0, 1] normalization.
    4. Add a batch dimension.
    """
    # 1. Color conversion
    if channels == 1:
        if len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
            processed = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        elif len(face_roi.shape) == 3 and face_roi.shape[2] == 4:
            processed = cv2.cvtColor(face_roi, cv2.COLOR_BGRA2GRAY)
        else:
            processed = face_roi.copy()
    else:
        # OpenCV frames are BGR; ResNet50 expects RGB images.
        if len(face_roi.shape) == 2:
            processed = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        elif face_roi.shape[2] == 4:
            processed = cv2.cvtColor(face_roi, cv2.COLOR_BGRA2RGB)
        else:
            processed = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

    # 2. Resize to target dimension using high quality interpolation
    processed = cv2.resize(
        processed,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )

    # ResNet50 preprocessing converts RGB values to its expected BGR/mean scale.
    processed = processed.astype("float32")
    if preprocessing == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        processed = preprocess_input(processed)
    elif preprocessing == "normalize":
        processed /= 255.0
    else:
        raise ValueError(f"Unsupported preprocessing mode: {preprocessing}")

    # 4. Reshape into batch tensor (1, height, width, channels)
    if channels == 1:
        processed = np.expand_dims(processed, axis=-1)  # (48, 48, 1)
    
    batch_tensor = np.expand_dims(processed, axis=0)     # (1, 48, 48, channels)
    return batch_tensor
