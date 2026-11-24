"""
Configuration settings for Mental Health Detection and Song Recommendation Backend.
Loaded from environment variables with sensible defaults for local development.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from services.mood_mapping import EMOTION_TO_MOOD_MAP

# Base backend directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file inside backend/
ENV_PATH = BASE_DIR / '.env'
load_dotenv(dotenv_path=ENV_PATH)


class Config:
    """Application configuration parameters."""

    BASE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data"

    # Server settings
    HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    PORT = int(os.getenv("FLASK_PORT", 5000))
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")

    # Security & Upload settings
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB maximum upload size
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

    # CORS settings (allow frontend requests)
    CORS_ORIGINS = [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5000",
        "http://localhost:5000"
    ]

    # Authentication session settings
    FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "").strip()
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").lower() in ("true", "1", "yes")

    # Optional MongoDB persistence for emotion prediction metadata
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017").strip()
    MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "mindpulse")
    MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "emotion_history")

    # CNN Model settings
    # Primary model path (supports .h5 and .keras)
    MODEL_FILENAME = os.getenv("MODEL_FILENAME", "resnet50_emotion_best.keras")
    MODEL_PATH = BASE_DIR / "models" / MODEL_FILENAME
    FALLBACK_MODEL_PATH = BASE_DIR / "models" / "emotion_model.keras"

    # ResNet50 input dimensions; loaded model shape overrides these defaults.
    MODEL_INPUT_WIDTH = int(os.getenv("MODEL_INPUT_WIDTH", 224))
    MODEL_INPUT_HEIGHT = int(os.getenv("MODEL_INPUT_HEIGHT", 224))
    MODEL_INPUT_CHANNELS = int(os.getenv("MODEL_INPUT_CHANNELS", 3))

    # FER2013 standard emotion classes in alphabetical order
    FER2013_EMOTIONS = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise"
    ]

    # Facial emotion-to-mood mapping (not a mental health diagnosis).
    EMOTION_TO_MOOD_MAP = EMOTION_TO_MOOD_MAP

    # Complete list of application mood categories
    # Note: 'focused' is a wellness mood category, not directly detected by FER2013.
    APPLICATION_MOODS = [
        "happy",
        "sad",
        "anxious",
        "stressed",
        "calm",
        "focused",
        "energetic"
    ]

    # YouTube Data API v3 settings
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
    YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
    DEFAULT_MAX_RECOMMENDATIONS = int(os.getenv("DEFAULT_MAX_RECOMMENDATIONS", 6))

    # Mood-based curated YouTube search queries (configurable)
    MOOD_SEARCH_QUERIES = {
        "happy": [
            "happy songs",
            "uplifting music",
            "feel good songs",
            "cheerful acoustic pop"
        ],
        "sad": [
            "comforting songs",
            "relaxing emotional songs",
            "peaceful music",
            "soothing piano instrumental"
        ],
        "anxious": [
            "anxiety relief music",
            "relaxing music",
            "calming songs",
            "432hz anxiety release healing sounds"
        ],
        "stressed": [
            "stress relief music",
            "relaxing songs",
            "meditation music",
            "lofi chill beats to relax"
        ],
        "calm": [
            "calm music",
            "peaceful music",
            "relaxing songs",
            "ambient nature sounds study"
        ],
        "focused": [
            "focus music",
            "study music",
            "concentration music",
            "binaural beats deep work"
        ],
        "energetic": [
            "energetic songs",
            "workout music",
            "upbeat songs",
            "motivation EDM workout"
        ]
    }

    # Academic Project & Non-Clinical Disclaimer
    DISCLAIMER = (
        "This software detects facial expressions and suggests music for emotional wellness. "
        "It does NOT provide a medical or clinical mental health diagnosis."
    )
