"""Helpers for validating credentials and protecting authenticated API routes."""

import re
from functools import wraps

from bson import ObjectId
from flask import g, jsonify, session
from werkzeug.security import check_password_hash

from services.emotion_history import emotion_history_store


EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def is_valid_email(email):
    return bool(EMAIL_PATTERN.fullmatch(email))


def is_strong_password(password):
    return (
        len(password) >= 8
        and any(character.islower() for character in password)
        and any(character.isupper() for character in password)
        and any(character.isdigit() for character in password)
    )


def public_user(user):
    return {
        "id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
    }


def verify_user_password(user, password):
    try:
        return check_password_hash(user.get("password_hash", ""), password)
    except (TypeError, ValueError):
        return False


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"success": False, "error": "Authentication required."}), 401

        normalized_user_id = user_id
        if isinstance(user_id, str):
            candidate = user_id.strip()
            if ObjectId.is_valid(candidate):
                normalized_user_id = ObjectId(candidate)

        try:
            user = emotion_history_store.users_collection().find_one(
                {"_id": normalized_user_id},
                {"password_hash": 0},
            )
        except Exception:
            return jsonify({"success": False, "error": "Authentication is temporarily unavailable."}), 503

        if not user:
            session.clear()
            return jsonify({"success": False, "error": "Your session has expired. Please log in again."}), 401

        g.current_user = user
        return view(*args, **kwargs)

    return wrapped