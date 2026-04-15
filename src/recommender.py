from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv
import math

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """
        Recommend top-k songs for a UserProfile.

        This implementation scores each Song using the same per-song logic
        used by the functional API (score_song) by converting dataclasses
        to dictionaries, then returns the top-k Song objects sorted by
        descending score.
        """
        # Convert user profile and songs to dicts compatible with score_song
        user_dict = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }

        scored: List[Tuple[Song, float, List[str]]] = []
        for s in self.songs:
            song_dict = {
                "id": s.id,
                "title": s.title,
                "artist": s.artist,
                "genre": s.genre,
                "mood": s.mood,
                "energy": s.energy,
                "tempo_bpm": s.tempo_bpm,
                "valence": s.valence,
                "danceability": s.danceability,
                "acousticness": s.acousticness,
            }
            score, reasons = score_song(user_dict, song_dict)
            scored.append((s, score, reasons))

        # Sort by score descending and return Song objects in that order
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """
        Return a short, human-readable explanation for why `song` was
        recommended to `user`.
        """
        user_dict = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        song_dict = {
            "id": song.id,
            "title": song.title,
            "artist": song.artist,
            "genre": song.genre,
            "mood": song.mood,
            "energy": song.energy,
            "tempo_bpm": song.tempo_bpm,
            "valence": song.valence,
            "danceability": song.danceability,
            "acousticness": song.acousticness,
        }
        _, reasons = score_song(user_dict, song_dict)
        return "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs: List[Dict] = []
    print(f"Loading songs from {csv_path}...")
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Convert numeric fields
            try:
                song = {
                    "id": int(row.get("id", 0)),
                    "title": row.get("title", "").strip(),
                    "artist": row.get("artist", "").strip(),
                    "genre": row.get("genre", "").strip(),
                    "mood": row.get("mood", "").strip(),
                    "energy": float(row.get("energy", 0.0)),
                    "tempo_bpm": float(row.get("tempo_bpm", 0.0)),
                    "valence": float(row.get("valence", 0.0)),
                    "danceability": float(row.get("danceability", 0.0)),
                    "acousticness": float(row.get("acousticness", 0.0)),
                }
            except Exception:
                # If parsing fails for a row, skip it
                continue
            songs.append(song)

    print(f"Loaded songs: {len(songs)}")
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    # Algorithm recipe (simplified):
    # - +2.0 points for genre match
    # - +1.0 point for mood match
    # - numeric closeness for each numeric feature (energy, valence, danceability, acousticness, tempo)
    # We'll compute raw points and normalize by the maximum possible to return a score in [0,1].

    reasons: List[str] = []
    raw_points = 0.0
    max_points = 0.0

    # Genre and mood
    genre_weight = 2.0
    mood_weight = 1.0
    max_points += genre_weight + mood_weight

    if user_prefs.get("genre") and song.get("genre"):
        if song["genre"].lower() == user_prefs["genre"].lower():
            raw_points += genre_weight
            reasons.append(f"genre match (+{genre_weight:.1f})")

    if user_prefs.get("mood") and song.get("mood"):
        if song["mood"].lower() == user_prefs["mood"].lower():
            raw_points += mood_weight
            reasons.append(f"mood match (+{mood_weight:.1f})")

    # Numeric features: list of (song_key, user_key, sigma, point_weight)
    numeric_features = [
        ("energy", "energy", 0.12, 1.0),
        ("valence", "valence", 0.15, 1.0),
        ("danceability", "danceability", 0.12, 1.0),
        ("acousticness", "acousticness", 0.15, 1.0),
        ("tempo_bpm", "tempo", 12.0, 1.0),
    ]

    for song_key, user_key, sigma, weight in numeric_features:
        if user_prefs.get(user_key) is None:
            # user didn't specify preference for this numeric feature
            continue
        # Ensure song has the key
        if song_key not in song:
            continue

        s_val = float(song[song_key])
        u_val = float(user_prefs[user_key])

        # For tempo_bpm we compare raw BPM using sigma in BPM units
        if song_key == "tempo_bpm":
            diff = s_val - u_val
            score_f = math.exp(- (diff ** 2) / (2 * (sigma ** 2)))
        else:
            # numeric features are expected in 0..1
            diff = s_val - u_val
            score_f = math.exp(- (diff ** 2) / (2 * (sigma ** 2)))

        contrib = score_f * weight
        raw_points += contrib
        max_points += weight
        reasons.append(f"{song_key} closeness (+{contrib:.2f})")

    # Avoid division by zero
    final_score = raw_points / max_points if max_points > 0 else 0.0
    return final_score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    # Score every song using score_song and return top-k sorted by score.
    scored: List[Tuple[Dict, float, List[str]]] = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, reasons))

    # Pythonic sorting: use sorted() to produce a new list sorted by score desc
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

    # Format explanation as a single string for CLI output
    results: List[Tuple[Dict, float, str]] = []
    for song, score, reasons in scored_sorted[:k]:
        explanation = "; ".join(reasons) if reasons else "No strong signals"
        results.append((song, score, explanation))

    return results
