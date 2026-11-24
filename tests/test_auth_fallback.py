import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app
from services.emotion_history import emotion_history_store


def test_users_collection_falls_back_when_mongodb_is_unreachable():
    emotion_history_store._mongo_available = False
    emotion_history_store._users_collection = None

    unique_id = f"fallback-user-{uuid.uuid4().hex[:8]}"
    unique_email = f"fallback-{uuid.uuid4().hex[:8]}@example.com"
    users = emotion_history_store.users_collection()
    user = users.insert_one({
        "_id": unique_id,
        "name": "Fallback User",
        "email": unique_email,
        "password_hash": "hash"
    })

    assert user.inserted_id == unique_id
    stored = users.find_one({"email": unique_email})
    assert stored is not None
    assert stored["name"] == "Fallback User"


def test_history_store_falls_back_for_recent_records():
    emotion_history_store._mongo_available = False
    emotion_history_store._collection = None

    record = {
        "emotion": "happy",
        "mood": "happy",
        "confidence": 0.91,
        "all_emotions": {"happy": 0.91},
        "model_status": "simulation_fallback",
    }
    emotion_history_store.record(record)

    recent = emotion_history_store.recent(limit=5)
    assert recent
    assert recent[0]["emotion"] == "happy"


def test_login_required_accepts_fallback_user_uuid_ids():
    emotion_history_store._mongo_available = False
    emotion_history_store._users_collection = None

    user_id = f"fallback-user-uuid-{uuid.uuid4().hex[:8]}"
    unique_email = f"uuid-user-{uuid.uuid4().hex[:8]}@example.com"
    users = emotion_history_store.users_collection()
    users.insert_one({
        "_id": user_id,
        "name": "UUID User",
        "email": unique_email,
        "password_hash": "hash",
    })

    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["user_id"] = user_id

        response = client.get("/api/auth/me")
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["success"] is True
        assert payload["user"]["email"] == unique_email
