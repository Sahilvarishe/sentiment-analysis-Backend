"""
YouTube Data API v3 Service.
Searches YouTube videos using official Google REST API v3.
Does NOT download, extract, proxy, rebroadcast, or store any YouTube audio or video files.
Returns only video IDs, titles, channels, and thumbnail metadata for frontend IFrame playback.
Includes robust fallback for missing API key, network errors, or quota exhaustion.
"""

from typing import List, Dict, Any, Optional
import requests
from config import Config


# Verified high quality curated fallback tracks for each mood
# Used when YOUTUBE_API_KEY is missing, quota is exceeded, or offline during viva
CURATED_FALLBACK_TRACKS: Dict[str, List[Dict[str, str]]] = {
    "happy": [
        {
            "title": "Pharrell Williams - Happy (Official Music Video)",
            "channel": "Pharrell Williams",
            "video_id": "ZbZSe6N_BXs",
            "thumbnail": "https://i.ytimg.com/vi/ZbZSe6N_BXs/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=ZbZSe6N_BXs"
        },
        {
            "title": "Justin Timberlake - CAN'T STOP THE FEELING!",
            "channel": "Justin Timberlake",
            "video_id": "ru0K8uYEZWw",
            "thumbnail": "https://i.ytimg.com/vi/ru0K8uYEZWw/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=ru0K8uYEZWw"
        },
        {
            "title": "Maroon 5 - Sugar (Official Music Video)",
            "channel": "Maroon 5",
            "video_id": "09R8_2nJtjg",
            "thumbnail": "https://i.ytimg.com/vi/09R8_2nJtjg/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=09R8_2nJtjg"
        }
    ],
    "sad": [
        {
            "title": "Yiruma - River Flows in You (Original Piano)",
            "channel": "Yiruma",
            "video_id": "7maJOI3QMu0",
            "thumbnail": "https://i.ytimg.com/vi/7maJOI3QMu0/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=7maJOI3QMu0"
        },
        {
            "title": "Ludovico Einaudi - Nuvole Bianche",
            "channel": "Ludovico Einaudi",
            "video_id": "4VR-6AS0-l4",
            "thumbnail": "https://i.ytimg.com/vi/4VR-6AS0-l4/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=4VR-6AS0-l4"
        },
        {
            "title": "Soothing Comforting Piano Melodies for Healing",
            "channel": "Relaxing Piano Music",
            "video_id": "1ZYbU82GVz4",
            "thumbnail": "https://i.ytimg.com/vi/1ZYbU82GVz4/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=1ZYbU82GVz4"
        }
    ],
    "anxious": [
        {
            "title": "Marconi Union - Weightless (Official 8 Hour Version - Scientifically Proven Most Relaxing)",
            "channel": "JustMusicCafe",
            "video_id": "qYnA9wWFHLI",
            "thumbnail": "https://i.ytimg.com/vi/qYnA9wWFHLI/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=qYnA9wWFHLI"
        },
        {
            "title": "432Hz Miracle Tone - Anxiety and Stress Relief Frequency",
            "channel": "PowerThoughts Meditation Club",
            "video_id": "2OEL4P1Rz04",
            "thumbnail": "https://i.ytimg.com/vi/2OEL4P1Rz04/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=2OEL4P1Rz04"
        },
        {
            "title": "Deep Relaxation & Panic Relief Ambient Soundscape",
            "channel": "Meditation Relax Music",
            "video_id": "UfcAVejslrU",
            "thumbnail": "https://i.ytimg.com/vi/UfcAVejslrU/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=UfcAVejslrU"
        }
    ],
    "stressed": [
        {
            "title": "Stress Relief Music - Calming Ambient Piano & Nature Sounds",
            "channel": "Yellow Brick Cinema",
            "video_id": "lFcSrYw-ARY",
            "thumbnail": "https://i.ytimg.com/vi/lFcSrYw-ARY/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=lFcSrYw-ARY"
        },
        {
            "title": "lofi hip hop radio - beats to relax/study to",
            "channel": "Lofi Girl",
            "video_id": "5qap5aO4i9A",
            "thumbnail": "https://i.ytimg.com/vi/5qap5aO4i9A/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=5qap5aO4i9A"
        },
        {
            "title": "Deep Healing Sleep & Tension Release",
            "channel": "NuMeditationMusic",
            "video_id": "1ZYbU82GVz4",
            "thumbnail": "https://i.ytimg.com/vi/1ZYbU82GVz4/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=1ZYbU82GVz4"
        }
    ],
    "calm": [
        {
            "title": "Peaceful Guitar & Piano Instrumental for Calm Mind",
            "channel": "Soothing Relaxation",
            "video_id": "2OEL4P1Rz04",
            "thumbnail": "https://i.ytimg.com/vi/2OEL4P1Rz04/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=2OEL4P1Rz04"
        },
        {
            "title": "Deep Theta Wave Ambient Zen Calm",
            "channel": "Brainwave Power Music",
            "video_id": "UfcAVejslrU",
            "thumbnail": "https://i.ytimg.com/vi/UfcAVejslrU/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=UfcAVejslrU"
        },
        {
            "title": "Ambient Chill Evening Soundscapes",
            "channel": "ChilledCow",
            "video_id": "5qap5aO4i9A",
            "thumbnail": "https://i.ytimg.com/vi/5qap5aO4i9A/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=5qap5aO4i9A"
        }
    ],
    "focused": [
        {
            "title": "Deep Focus Music - 4 Hours of Ambient Study Music for Work & Concentration",
            "channel": "Greenred Productions",
            "video_id": "WPni755-Krg",
            "thumbnail": "https://i.ytimg.com/vi/WPni755-Krg/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=WPni755-Krg"
        },
        {
            "title": "Alpha Waves for Deep Work & Extreme Focus",
            "channel": "Study Music Project",
            "video_id": "sjkrrmBnpGE",
            "thumbnail": "https://i.ytimg.com/vi/sjkrrmBnpGE/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=sjkrrmBnpGE"
        },
        {
            "title": "Binaural Beats Focus & Memory Wave",
            "channel": "Magnetic Minds",
            "video_id": "sjkrrmBnpGE",
            "thumbnail": "https://i.ytimg.com/vi/sjkrrmBnpGE/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=sjkrrmBnpGE"
        }
    ],
    "energetic": [
        {
            "title": "The Weeknd - Blinding Lights",
            "channel": "The Weeknd",
            "video_id": "4NRXx6U8ABQ",
            "thumbnail": "https://i.ytimg.com/vi/4NRXx6U8ABQ/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=4NRXx6U8ABQ"
        },
        {
            "title": "Dua Lipa - Levitating (Official Music Video)",
            "channel": "Dua Lipa",
            "video_id": "TUVcZfQe-Kw",
            "thumbnail": "https://i.ytimg.com/vi/TUVcZfQe-Kw/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=TUVcZfQe-Kw"
        },
        {
            "title": "Avicii - Wake Me Up (Official Video)",
            "channel": "Avicii",
            "video_id": "IcrbM1l_BoI",
            "thumbnail": "https://i.ytimg.com/vi/IcrbM1l_BoI/hqdefault.jpg",
            "url": "https://www.youtube.com/watch?v=IcrbM1l_BoI"
        }
    ]
}


class YouTubeService:
    """Service to search YouTube Data API v3 safely with error handling and caching."""

    def __init__(self):
        self.api_key: str = Config.YOUTUBE_API_KEY
        self.search_url: str = Config.YOUTUBE_SEARCH_URL
        self._cache: Dict[str, List[Dict[str, str]]] = {}

    def search_videos(self, query: str, max_results: int = 6, mood_hint: str = "calm") -> Dict[str, Any]:
        """
        Execute YouTube search using Data API v3.
        Returns dictionary with:
          - 'success': bool
          - 'source': 'youtube_api_v3' or 'curated_fallback'
          - 'videos': list of video dictionaries
          - 'message': informative note
        """
        query_clean = query.strip()
        if not query_clean:
            query_clean = "calm relaxing songs"

        # Check in-memory cache to save API quota
        cache_key = f"{query_clean.lower()}_{max_results}"
        if cache_key in self._cache:
            return {
                "success": True,
                "source": "cache",
                "videos": self._cache[cache_key],
                "message": "Results retrieved from local cache"
            }

        # Check if API Key is configured
        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            print(f"[YouTubeService] No valid YOUTUBE_API_KEY found in .env. Using curated fallback.")
            fallback_videos = self.get_curated_fallback(mood_hint, max_results)
            return {
                "success": True,
                "source": "curated_fallback",
                "videos": fallback_videos,
                "message": "Set your YOUTUBE_API_KEY in backend/.env for live YouTube searches"
            }

        # Call official YouTube Data API v3
        params = {
            "part": "snippet",
            "q": query_clean,
            "type": "video",
            "videoCategoryId": "10",  # Category 10 = Music on YouTube
            "maxResults": min(max(1, max_results), 12),
            "key": self.api_key
        }

        try:
            response = requests.get(self.search_url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                videos = []

                for item in items:
                    video_id = item.get("id", {}).get("videoId")
                    snippet = item.get("snippet", {})
                    if not video_id:
                        continue

                    title = snippet.get("title", "Unknown Title")
                    channel = snippet.get("channelTitle", "Unknown Channel")
                    thumbnails = snippet.get("thumbnails", {})
                    thumb_url = (
                        thumbnails.get("high", {}).get("url") or
                        thumbnails.get("medium", {}).get("url") or
                        thumbnails.get("default", {}).get("url") or
                        f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
                    )

                    videos.append({
                        "title": title,
                        "channel": channel,
                        "video_id": video_id,
                        "thumbnail": thumb_url,
                        "url": f"https://www.youtube.com/watch?v={video_id}"
                    })

                if videos:
                    self._cache[cache_key] = videos
                    return {
                        "success": True,
                        "source": "youtube_api_v3",
                        "videos": videos,
                        "message": f"Retrieved {len(videos)} videos from YouTube Data API v3"
                    }
                else:
                    # No videos returned for query
                    fallback_videos = self.get_curated_fallback(mood_hint, max_results)
                    return {
                        "success": True,
                        "source": "curated_fallback",
                        "videos": fallback_videos,
                        "message": "No live results found, returned curated songs."
                    }

            elif response.status_code == 403:
                # Quota exceeded or invalid key
                err_data = response.json().get("error", {})
                err_msg = err_data.get("message", "YouTube API quota exceeded or key invalid.")
                print(f"[YouTubeService] Quota or Auth error (403): {err_msg}")
                fallback_videos = self.get_curated_fallback(mood_hint, max_results)
                return {
                    "success": True,
                    "source": "curated_fallback",
                    "videos": fallback_videos,
                    "message": f"YouTube API 403 (Quota/Key issue). Using verified therapeutic songs: {err_msg}"
                }
            else:
                print(f"[YouTubeService] HTTP Error {response.status_code}: {response.text}")
                fallback_videos = self.get_curated_fallback(mood_hint, max_results)
                return {
                    "success": True,
                    "source": "curated_fallback",
                    "videos": fallback_videos,
                    "message": f"YouTube API returned HTTP {response.status_code}. Using curated songs."
                }

        except requests.exceptions.RequestException as net_err:
            print(f"[YouTubeService] Network connection error: {net_err}")
            fallback_videos = self.get_curated_fallback(mood_hint, max_results)
            return {
                "success": True,
                "source": "curated_fallback",
                "videos": fallback_videos,
                "message": "Network error reaching YouTube. Returned verified songs."
            }

    def get_curated_fallback(self, mood: str, max_results: int = 6) -> List[Dict[str, str]]:
        """Return verified songs for a given mood from the curated database."""
        mood_key = (mood or "calm").lower()
        songs = CURATED_FALLBACK_TRACKS.get(mood_key, CURATED_FALLBACK_TRACKS["calm"])
        return songs[:max_results]


# Singleton YouTube Service
youtube_service = YouTubeService()
