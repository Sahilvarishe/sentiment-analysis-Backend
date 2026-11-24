import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app
from services.recommendation import recommendation_service
from config import Config


def test_recommendation_service_queries_all_five_languages():
    languages = ["Hindi", "Marathi", "English", "Tamil", "Malayalam"]
    for lang in languages:
        queries = recommendation_service.get_queries_for_mood("happy", language=lang)
        assert any(lang.lower() in q.lower() for q in queries)
        assert any("happy" in q.lower() for q in queries)

    # Test example queries
    marathi_sad = recommendation_service.get_queries_for_mood("sad", language="Marathi")
    assert any("marathi" in q.lower() and "sad" in q.lower() for q in marathi_sad)

    english_calm = recommendation_service.get_queries_for_mood("calm", language="English")
    assert any("english" in q.lower() and "calm" in q.lower() for q in english_calm)

    tamil_energetic = recommendation_service.get_queries_for_mood("energetic", language="Tamil")
    assert any("tamil" in q.lower() and "energetic" in q.lower() for q in tamil_energetic)

    malayalam_relaxing = recommendation_service.get_queries_for_mood("relaxing", language="Malayalam")
    assert any("malayalam" in q.lower() for q in malayalam_relaxing)


def test_recommendation_service_backward_compatibility():
    # Without language, returns original queries
    default_queries = recommendation_service.get_queries_for_mood("calm")
    assert default_queries == Config.MOOD_SEARCH_QUERIES["calm"]


def test_recommendations_endpoint_backward_compatible():
    with app.test_client() as client:
        # 1. No params at all
        res = client.get("/api/recommendations")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["mood"] == "calm"
        assert "recommendations" in data

        # 2. Only mood param
        res = client.get("/api/recommendations?mood=happy&limit=3")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["mood"] == "happy"
        assert len(data["recommendations"]) <= 3


def test_recommendations_endpoint_all_five_languages():
    languages = [
        ("Hindi", "happy"),
        ("Marathi", "sad"),
        ("English", "calm"),
        ("Tamil", "energetic"),
        ("Malayalam", "calm")
    ]
    with app.test_client() as client:
        for lang, mood in languages:
            res = client.get(f"/api/recommendations?mood={mood}&language={lang.lower()}&limit=2")
            assert res.status_code == 200
            data = res.get_json()
            assert data["success"] is True
            assert data["mood"] == mood
            assert data.get("language") == lang
            assert "recommendations" in data
            assert lang.lower() in data.get("query_used", "").lower()


def test_recommendations_endpoint_emotion_mapping():
    with app.test_client() as client:
        # Emotion 'angry' maps to mood 'stressed'
        res = client.get("/api/recommendations?emotion=angry&language=hindi&limit=2")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["mood"] == "stressed"
        assert data.get("language") == "Hindi"
        assert "hindi" in data.get("query_used", "").lower()


def test_recommendations_endpoint_post_json():
    with app.test_client() as client:
        res = client.post("/api/recommendations", json={
            "emotion": "happy",
            "language": "Marathi",
            "limit": 3
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["mood"] == "happy"
        assert data.get("language") == "Marathi"
        assert "marathi" in data.get("query_used", "").lower()
