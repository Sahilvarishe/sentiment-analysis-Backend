"""MongoDB-backed prediction history with a local SQLite fallback.

This keeps the app functioning when the configured Atlas cluster is unreachable,
which is common in restricted network environments or with expired credentials.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from config import Config


class _InsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class FallbackUsersCollection:
    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self._ensure_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    _id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def create_index(self, *args, **kwargs):
        return None

    def find_one(self, query=None, projection=None):
        if not isinstance(query, dict) or not query:
            return None

        if "_id" in query:
            value = str(query["_id"])
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM users WHERE _id = ?", (value,)).fetchone()
        elif "email" in query:
            value = query["email"]
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM users WHERE email = ?", (value,)).fetchone()
        else:
            return None

        if row is None:
            return None

        data = dict(row)
        data["_id"] = str(data["_id"])

        if isinstance(projection, dict):
            include = {key for key, value in projection.items() if value != 0}
            exclude = {key for key, value in projection.items() if value == 0}
            if include:
                data = {key: value for key, value in data.items() if key in include or key == "_id"}
            elif exclude:
                data = {key: value for key, value in data.items() if key not in exclude}

        return data

    def insert_one(self, document):
        payload = dict(document)
        payload["_id"] = str(payload.get("_id") or uuid.uuid4())
        created_at = payload.get("created_at")
        if created_at is not None and not isinstance(created_at, str):
            payload["created_at"] = created_at.isoformat()

        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO users (_id, name, email, password_hash, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        payload["_id"],
                        payload["name"],
                        payload["email"],
                        payload["password_hash"],
                        payload.get("created_at", datetime.now(timezone.utc).isoformat()),
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                raise DuplicateKeyError("email") from exc
        return _InsertResult(payload["_id"])


class FallbackHistoryCollection:
    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self._ensure_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS emotion_history (
                    _id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    emotion TEXT,
                    mood TEXT,
                    confidence REAL,
                    all_emotions TEXT,
                    model_status TEXT
                )
                """
            )
            conn.commit()

    def create_index(self, *args, **kwargs):
        return None

    def insert_one(self, document):
        payload = {"_id": str(document.get("_id") or uuid.uuid4())}
        created_at = document.get("created_at")
        if created_at is not None and not isinstance(created_at, str):
            created_at = created_at.isoformat()
        payload["created_at"] = created_at or datetime.now(timezone.utc).isoformat()
        payload["emotion"] = document.get("emotion")
        payload["mood"] = document.get("mood")
        payload["confidence"] = document.get("confidence")
        payload["all_emotions"] = json.dumps(document.get("all_emotions", {}))
        payload["model_status"] = document.get("model_status")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO emotion_history (_id, created_at, emotion, mood, confidence, all_emotions, model_status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["_id"],
                    payload["created_at"],
                    payload["emotion"],
                    payload["mood"],
                    payload["confidence"],
                    payload["all_emotions"],
                    payload["model_status"],
                ),
            )
            conn.commit()

        return _InsertResult(payload["_id"])

    def find(self, query=None, projection=None):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM emotion_history ORDER BY created_at DESC"
            ).fetchall()
        records = []
        for row in rows:
            record = dict(row)
            record["_id"] = str(record["_id"])
            record["all_emotions"] = json.loads(record["all_emotions"] or "{}")
            records.append(record)
        return records


class EmotionHistoryStore:
    """Lazily connect to MongoDB, but fall back to SQLite when the Atlas URI is unreachable."""

    def __init__(self):
        self._client = None
        self._database = None
        self._collection = None
        self._users_collection = None
        self._fallback_users = None
        self._fallback_history = None
        self._mongo_available = None
        self._fallback_db_dir = Config.DATA_DIR
        self._fallback_db_dir.mkdir(exist_ok=True, parents=True)

    @property
    def configured(self):
        return bool(Config.MONGODB_URI)

    def _get_database(self):
        if not self.configured:
            raise RuntimeError("MongoDB is not configured.")

        if self._database is None:
            self._client = MongoClient(
                Config.MONGODB_URI,
                serverSelectionTimeoutMS=2000,
                connectTimeoutMS=2000,
                socketTimeoutMS=2000,
            )
            self._database = self._client[Config.MONGODB_DATABASE]
        return self._database

    def _mongo_healthy(self):
        if not self.configured:
            return False
        if self._mongo_available is not None:
            return self._mongo_available
        try:
            self._get_database().command("ping")
            self._mongo_available = True
            return True
        except Exception:
            self._mongo_available = False
            return False

    def _get_fallback_users(self):
        if self._fallback_users is None:
            self._fallback_users = FallbackUsersCollection(self._fallback_db_dir / "users.sqlite3")
        return self._fallback_users

    def _get_fallback_history(self):
        if self._fallback_history is None:
            self._fallback_history = FallbackHistoryCollection(self._fallback_db_dir / "emotion_history.sqlite3")
        return self._fallback_history

    def _get_collection(self):
        if self._mongo_healthy():
            if self._collection is None:
                self._collection = self._get_database()[Config.MONGODB_COLLECTION]
                self._collection.create_index([("created_at", -1)])
            return self._collection
        return self._get_fallback_history()

    def users_collection(self):
        if self._mongo_healthy():
            if self._users_collection is None:
                self._users_collection = self._get_database()["users"]
                self._users_collection.create_index("email", unique=True)
            return self._users_collection
        return self._get_fallback_users()

    def status(self):
        if not self.configured:
            return "not_configured"
        if self._mongo_healthy():
            return "connected"
        return "fallback"

    def record(self, result):
        document = {
            "created_at": datetime.now(timezone.utc),
            "emotion": result.get("emotion"),
            "mood": result.get("mood"),
            "confidence": result.get("confidence"),
            "all_emotions": result.get("all_emotions", {}),
            "model_status": result.get("model_status"),
        }
        if self._mongo_healthy():
            self._get_collection().insert_one(document)
            return
        self._get_fallback_history().insert_one(document)

    def recent(self, limit=50):
        if self._mongo_healthy():
            cursor = self._get_collection().find(
                {},
                {"_id": 1, "created_at": 1, "emotion": 1, "mood": 1, "confidence": 1,
                 "all_emotions": 1, "model_status": 1},
            ).sort("created_at", -1).limit(limit)

            records = []
            for document in cursor:
                created_at = document.get("created_at")
                records.append({
                    "id": str(document["_id"]),
                    "timestamp": created_at.isoformat() if created_at else None,
                    "emotion": document.get("emotion"),
                    "mood": document.get("mood"),
                    "confidence": document.get("confidence"),
                    "all_emotions": document.get("all_emotions", {}),
                    "model_status": document.get("model_status"),
                })
            return records

        rows = self._get_fallback_history().find()[:limit]
        records = []
        for document in rows:
            created_at = document.get("created_at")
            records.append({
                "id": document.get("_id"),
                "timestamp": created_at,
                "emotion": document.get("emotion"),
                "mood": document.get("mood"),
                "confidence": document.get("confidence"),
                "all_emotions": document.get("all_emotions", {}),
                "model_status": document.get("model_status"),
            })
        return records


emotion_history_store = EmotionHistoryStore()