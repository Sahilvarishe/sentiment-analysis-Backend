"""
Music Recommendation Service for Mental Health & Emotional Wellness.
Translates detected mood states into targeted acoustic music queries,
queries YouTube Data API v3, and returns explainable recommendations.
"""

import random
from typing import Dict, Any, List, Optional
from config import Config
from services.youtube import youtube_service


class RecommendationService:
    """Service to generate explainable mood-based music recommendations."""

    def __init__(self):
        # Configurable mood search queries
        self.queries = Config.MOOD_SEARCH_QUERIES
        self.valid_moods = Config.APPLICATION_MOODS

    def get_queries_for_mood(self, mood: str, language: Optional[str] = None) -> List[str]:
        """Return list of search queries configured for a given mood and optional language."""
        mood_key = (mood or "calm").lower()
        if mood_key == "relaxing":
            mood_key = "calm"

        if not language or not str(language).strip():
            return self.queries.get(mood_key, self.queries.get("calm", ["calm music"]))

        lang = str(language).strip().title()

        # Language-specific query templates per mood
        language_queries = {
            "happy": [
                f"happy {lang} songs",
                f"uplifting {lang} music",
                f"feel good {lang} songs",
                f"cheerful {lang} songs",
            ],
            "sad": [
                f"sad {lang} songs",
                f"comforting {lang} songs",
                f"relaxing emotional {lang} songs",
                f"soothing {lang} songs",
            ],
            "calm": [
                f"calm {lang} music",
                f"relaxing {lang} songs",
                f"peaceful {lang} music",
                f"soothing {lang} songs",
            ],
            "anxious": [
                f"calm {lang} songs",
                f"anxiety relief {lang} music",
                f"relaxing {lang} music",
                f"peaceful {lang} songs",
            ],
            "stressed": [
                f"relaxing {lang} songs",
                f"stress relief {lang} music",
                f"calm {lang} music",
                f"peaceful {lang} songs",
            ],
            "focused": [
                f"focus {lang} music",
                f"study {lang} music",
                f"peaceful {lang} instrumental",
                f"concentration {lang} music",
            ],
            "energetic": [
                f"energetic {lang} songs",
                f"upbeat {lang} songs",
                f"workout {lang} music",
                f"motivation {lang} songs",
            ],
        }

        return language_queries.get(
            mood_key,
            [f"{mood_key} {lang} songs", f"relaxing {lang} music", f"peaceful {lang} songs"]
        )

    def recommend_songs(
        self,
        mood: str = "calm",
        max_results: int = 6,
        language: Optional[str] = None,
        emotion: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate song recommendations for a specific mood and optional language.
        
        Args:
            mood: One of 'happy', 'sad', 'anxious', 'stressed', 'calm', 'focused', 'energetic'
            max_results: Number of songs to return (default: 6)
            language: Optional language (e.g. 'Hindi', 'Marathi', 'English', 'Tamil', 'Malayalam')
            emotion: Optional detected facial emotion (e.g. 'happy', 'sad', 'angry')

        Returns:
            Dictionary containing mood, query used, recommendations list, and metadata.
        """
        # If emotion is provided and mood not explicitly specified, map emotion to mood
        if emotion and (not mood or mood == "calm"):
            mood = Config.EMOTION_TO_MOOD_MAP.get(emotion.lower(), mood or "calm")

        mood_clean = (mood or "calm").lower()
        if mood_clean == "relaxing":
            mood_clean = "calm"

        # Validate mood or default to 'calm'
        if mood_clean not in self.valid_moods:
            if mood_clean in Config.EMOTION_TO_MOOD_MAP:
                mood_clean = Config.EMOTION_TO_MOOD_MAP[mood_clean]
            else:
                print(f"[RecommendationService] Unrecognized mood '{mood_clean}'. Defaulting to 'calm'.")
                mood_clean = "calm"

        # Pick a query to provide fresh song diversity matching mood & language
        available_queries = self.get_queries_for_mood(mood_clean, language=language)
        selected_query = random.choice(available_queries)

        # Call YouTube Data API Service
        search_result = youtube_service.search_videos(
            query=selected_query,
            max_results=max_results,
            mood_hint=mood_clean
        )

        response_data = {
            "success": True,
            "mood": mood_clean,
            "query_used": selected_query,
            "source": search_result.get("source", "youtube_api_v3"),
            "message": search_result.get("message", ""),
            "count": len(search_result.get("videos", [])),
            "recommendations": search_result.get("videos", []),
            "disclaimer": Config.DISCLAIMER
        }

        if language and str(language).strip():
            response_data["language"] = str(language).strip().title()

        return response_data


# Singleton Recommendation Service
recommendation_service = RecommendationService()
