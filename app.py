"""
Flask REST API for Mental Health Detection and Song Recommendation System.
Provides endpoints for health check, facial emotion recognition,
mood-based music recommendations, and YouTube Data API v3 search.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime, timedelta, timezone
import secrets

from flask import Flask, request, jsonify, session, g
try:
    from flask_cors import CORS
except ImportError:
    print("[Warning] 'flask-cors' is not installed. Using internal CORS fallback headers.")
    class CORS:
        def __init__(self, app, *args, **kwargs):
            @app.after_request
            def add_cors_headers(response):
                origin = request.headers.get("Origin", "*")
                response.headers["Vary"] = "Origin"
                if origin in Config.CORS_ORIGINS:
                    response.headers["Access-Control-Allow-Origin"] = origin
                    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
                    response.headers["Access-Control-Allow-Credentials"] = "true"
                return response
            @app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
            @app.route("/<path:path>", methods=["OPTIONS"])
            def options_handler(path):
                return ("", 204)
from pymongo.errors import DuplicateKeyError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import generate_password_hash

from config import Config
from services.emotion_detection import emotion_service
from services.emotion_history import emotion_history_store
from services.authentication import (
    is_strong_password,
    is_valid_email,
    login_required,
    public_user,
    verify_user_password,
)
from services.recommendation import recommendation_service
from services.youtube import youtube_service
from utils.image_processing import (
    is_allowed_file,
    decode_image_from_bytes,
    decode_base64_image
)

# Initialize Flask application
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
app.config.update(
    SECRET_KEY=Config.FLASK_SECRET_KEY or secrets.token_hex(32),
    SESSION_COOKIE_NAME="moodmate_session",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=Config.SESSION_COOKIE_SECURE,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)

# Enable CORS for cross-origin requests from the HTML/JS frontend
CORS(app, resources={r"/*": {"origins": Config.CORS_ORIGINS}}, supports_credentials=True)


# ============================================================================
# 1. Health Check Endpoint
# ============================================================================
@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint to verify backend status."""
    return jsonify({
        "status": "success",
        "message": "Backend is running",
        "model_status": "loaded" if not emotion_service.is_mock_model else "simulation_fallback",
        "mongodb_status": emotion_history_store.status(),
        "youtube_api_configured": bool(Config.YOUTUBE_API_KEY and Config.YOUTUBE_API_KEY != "YOUR_API_KEY_HERE"),
        "disclaimer": Config.DISCLAIMER
    }), 200


@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "Provide registration details as JSON."}), 400

    name = data.get("name", "")
    email = data.get("email", "")
    password = data.get("password", "")
    confirm_password = data.get("confirm_password", "")

    if not all(isinstance(value, str) for value in (name, email, password, confirm_password)):
        return jsonify({"success": False, "error": "Please check the registration details."}), 400

    name = name.strip()
    email = email.strip().lower()
    if not name or len(name) > 80:
        return jsonify({"success": False, "error": "Enter your name (up to 80 characters)."}), 400
    if not email or not is_valid_email(email):
        return jsonify({"success": False, "error": "Enter a valid email address."}), 400
    if not password:
        return jsonify({"success": False, "error": "Enter a password."}), 400
    if password != confirm_password:
        return jsonify({"success": False, "error": "Passwords do not match."}), 400
    if not is_strong_password(password):
        return jsonify({
            "success": False,
            "error": "Use at least 8 characters with an uppercase letter, a lowercase letter, and a number.",
        }), 400
    if data.get("terms_accepted") is not True:
        return jsonify({"success": False, "error": "Accept the Terms & Conditions to continue."}), 400

    try:
        users = emotion_history_store.users_collection()
        if users.find_one({"email": email}, {"_id": 1}):
            return jsonify({"success": False, "error": "An account with this email already exists."}), 409

        user = {
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password),
            "created_at": datetime.now(timezone.utc),
        }
        result = users.insert_one(user)
        user["_id"] = result.inserted_id
        return jsonify({"success": True, "message": "Account created successfully.", "user": public_user(user)}), 201
    except DuplicateKeyError:
        return jsonify({"success": False, "error": "An account with this email already exists."}), 409
    except Exception as e:
        print(f"[App] Registration failed: {e}")
        return jsonify({"success": False, "error": "Registration is temporarily unavailable."}), 503


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "Provide your email and password."}), 400

    email = data.get("email", "")
    password = data.get("password", "")
    if not isinstance(email, str) or not isinstance(password, str) or not email.strip() or not password:
        return jsonify({"success": False, "error": "Enter your email and password."}), 400

    email = email.strip().lower()
    if not is_valid_email(email):
        return jsonify({"success": False, "error": "Enter a valid email address."}), 400

    try:
        user = emotion_history_store.users_collection().find_one({"email": email})
    except Exception as e:
        print(f"[App] Login lookup failed: {e}")
        return jsonify({"success": False, "error": "Login is temporarily unavailable."}), 503

    if not user or not verify_user_password(user, password):
        return jsonify({"success": False, "error": "Email or password is incorrect."}), 401

    session.clear()
    session["user_id"] = str(user["_id"])
    session.permanent = data.get("remember_me") is True
    return jsonify({"success": True, "user": public_user(user)}), 200


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "You have been logged out."}), 200


@app.route("/api/auth/me", methods=["GET"])
@login_required
def current_user():
    return jsonify({"success": True, "user": public_user(g.current_user)}), 200


# ============================================================================
# 2. Facial Emotion Detection Endpoint
# ============================================================================
@app.route("/api/detect-emotion", methods=["POST"])
@app.route("/predict_emotion", methods=["POST"])  # Alias for compatibility
@login_required
def detect_emotion():
    """
    Receives an image via:
      a) Multipart form-data ('image' file field)
      b) JSON payload with 'image' (Base64 data URI or string)
      
    Pipeline:
      Image -> In-memory decode -> OpenCV face detect & crop ->
      CNN preprocess & inference -> Emotion-to-Mood mapping -> JSON response
    """
    image_bgr = None

    # Case A: Multipart form file upload
    if "image" in request.files:
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Empty file name provided."}), 400

        if not is_allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"Invalid file format. Allowed formats: {', '.join(Config.ALLOWED_EXTENSIONS)}"
            }), 400

        try:
            file_bytes = file.read()
            image_bgr = decode_image_from_bytes(file_bytes)
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to read image file: {str(e)}"}), 400

    # Case B: JSON payload with Base64 image
    elif request.is_json and "image" in request.json:
        b64_str = request.json.get("image", "")
        if not b64_str:
            return jsonify({"success": False, "error": "Empty image string provided."}), 400
        image_bgr = decode_base64_image(b64_str)

    # Neither provided
    else:
        return jsonify({
            "success": False,
            "error": "No image provided. Send a multipart form file or JSON with base64 'image'."
        }), 400

    # Validation: Verify valid OpenCV image decoded
    if image_bgr is None or image_bgr.size == 0:
        return jsonify({"success": False, "error": "Invalid image data. Could not decode image."}), 400

    # Run Emotion Prediction Pipeline
    try:
        detection_result = emotion_service.predict_emotion(image_bgr)
        status_code = 200 if detection_result.get("success") else 422
        if detection_result.get("success"):
            try:
                emotion_history_store.record(detection_result)
            except Exception as e:
                print(f"[App] Could not save emotion history to MongoDB: {e}")
        return jsonify(detection_result), status_code
    except Exception as e:
        print(f"[App] Internal error during emotion detection: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal server error processing image: {str(e)}"
        }), 500


@app.route("/api/history", methods=["GET"])
@login_required
def get_emotion_history():
    """Return recent prediction metadata when MongoDB is configured."""
    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Limit must be an integer."}), 400

    if limit < 1 or limit > 100:
        return jsonify({"success": False, "error": "Limit must be between 1 and 100."}), 400

    try:
        records = emotion_history_store.recent(limit=limit)
        return jsonify({"success": True, "records": records}), 200
    except Exception as e:
        print(f"[App] Could not retrieve emotion history from MongoDB: {e}")
        return jsonify({"success": False, "error": "Emotion history is unavailable."}), 503


# ============================================================================
# 3. Song Recommendations Endpoint
# ============================================================================
@app.route("/api/recommendations", methods=["GET", "POST"])
def get_recommendations():
    """
    Generates music recommendations based on a requested mood and optional language.
    Supports GET /api/recommendations?mood=happy&language=hindi&limit=6
    Supports GET /api/recommendations?emotion=happy&language=hindi&limit=6
    and POST /api/recommendations with JSON {"mood": "happy", "language": "hindi", "limit": 6}
    """
    mood = None
    emotion = None
    language = None
    limit = Config.DEFAULT_MAX_RECOMMENDATIONS

    if request.method == "POST" and request.is_json:
        data = request.get_json() or {}
        mood = data.get("mood")
        emotion = data.get("emotion")
        language = data.get("language")
        limit = int(data.get("limit", Config.DEFAULT_MAX_RECOMMENDATIONS))
    else:
        mood = request.args.get("mood")
        emotion = request.args.get("emotion")
        language = request.args.get("language")
        limit = int(request.args.get("limit", Config.DEFAULT_MAX_RECOMMENDATIONS))

    # Preserve emotion-to-mood mapping if emotion is supplied
    if emotion and not mood:
        mood = Config.EMOTION_TO_MOOD_MAP.get(emotion.lower(), emotion)
    elif not mood:
        mood = "calm"

    try:
        result = recommendation_service.recommend_songs(
            mood=mood,
            max_results=limit,
            language=language,
            emotion=emotion
        )
        return jsonify(result), 200
    except Exception as e:
        print(f"[App] Recommendation error: {e}")
        return jsonify({
            "success": False,
            "error": f"Failed to retrieve recommendations: {str(e)}"
        }), 500


# ============================================================================
# 4. YouTube Search Endpoint
# ============================================================================
@app.route("/api/youtube/search", methods=["GET"])
def search_youtube():
    """
    Direct YouTube Data API v3 search endpoint.
    Example: GET /api/youtube/search?q=happy+music&limit=6
    Only returns YouTube video IDs and metadata for IFrame playback.
    """
    query = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", Config.DEFAULT_MAX_RECOMMENDATIONS))
    mood_hint = request.args.get("mood", "calm")

    if not query:
        return jsonify({
            "success": False,
            "error": "Query parameter 'q' is required."
        }), 400

    try:
        result = youtube_service.search_videos(query=query, max_results=limit, mood_hint=mood_hint)
        return jsonify(result), 200
    except Exception as e:
        print(f"[App] YouTube search error: {e}")
        return jsonify({
            "success": False,
            "error": f"Failed to search YouTube: {str(e)}"
        }), 500


# ============================================================================
# Global Error Handlers (Clean JSON responses)
# ============================================================================
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        "success": False,
        "error": "Image file too large. Maximum allowed size is 10 MB."
    }), 413


@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({
        "success": False,
        "error": "Endpoint not found."
    }), 404


@app.errorhandler(500)
def handle_internal_error(e):
    return jsonify({
        "success": False,
        "error": "An internal server error occurred."
    }), 500


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    print("================================================================")
    print(" 🧠 Mental Health Detection and Song Recommendation Backend API ")
    print("================================================================")
    print(f" -> Server running at: http://{Config.HOST}:{Config.PORT}")
    print(f" -> Health Check: http://127.0.0.1:{Config.PORT}/api/health")
    print(f" -> Emotion API:  http://127.0.0.1:{Config.PORT}/api/detect-emotion")
    print(f" -> Music API:    http://127.0.0.1:{Config.PORT}/api/recommendations")
    print("================================================================")
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
