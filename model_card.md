# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

VibeFinder 1.0

---

## 2. Intended Use

This system suggests 3–5 songs from a small local catalog that best match a user's stated taste (genre, mood, and numeric "vibe" targets like energy and tempo). It is designed for classroom exploration, learning, and experimentation, not for production use.

---

## 3. How the Model Works

For each song the model checks categorical matches (genre and mood) and measures how close numeric features are to the user's targets (energy, valence, danceability, acousticness, tempo). Exact genre match gives a large boost, mood gives a smaller boost, and numeric features are scored by how close they are to the target using a smooth Gaussian function. The per-feature contributions are summed and normalized to produce a content score in [0,1]. That score is blended with a small popularity prior using a weight α to produce the final ranking score.

---

## 4. Data

The catalog contains 18 songs in `data/songs.csv`. Genres include pop, lofi, rock, ambient, jazz, synthwave, indie pop, classical, metal, reggae, hip hop, country, blues, electronic, and folk. Moods include happy, chill, intense, contemplative, moody, nostalgic, sultry, euphoric, and more. I added 8 diverse songs to the starter file to increase coverage. The dataset is small and omits rich signals like lyrics, detailed instrumentation, and real user interaction logs.

---

## 5. Strengths

- Transparent and easy to reason about: the reasons list explains which features contributed.
- Works well for clear, narrow tastes (e.g., low-energy acoustic listeners vs high-energy workout listeners).
- Handles cold-start songs: new songs with features can be scored immediately without interaction data.

---

## 6. Limitations and Bias

- Over-prioritizes genre tags: songs that match genre are strongly favored even if mood or numeric features differ.
- Small catalog and handcrafted weights can bias results toward genres that appear more often.
- Lacks user interaction signals (skips, likes), so it cannot learn from behavior or detect context (time of day).
- Numeric features are normalized but derived values (tempo, loudness) may not reflect perceived similarity for all listeners.

---

## 7. Evaluation

I tested several synthetic profiles (e.g., "Evening Chill" with low energy and high acousticness; "Workout" with high energy). For each profile I inspected the top-5 results and checked that genre/mood and numeric closeness appeared in the explanation strings. Tests in `tests/test_recommender.py` validate that the Recommender returns higher-ranked items that match favorite genre and mood in a tiny catalog.

---

## 8. Future Work

- Add interaction signals (plays, skips, likes) to train collaborative filters or to tune weights.
- Add embeddings for genre/mood and use semantic similarity rather than exact matches.
- Implement MMR or slate optimization to improve diversity and reduce artist-repeat.
- Expand dataset and include lyrics/audio embeddings for richer content matching.

---

## 9. Personal Reflection

Building this small recommender clarified how a few numeric features (energy, valence, tempo) combined with simple categorical tags can produce recommendations that feel reasonable. My biggest learning was that the math (Gaussian closeness and weight tuning) matters: small changes in σ or weights change the top results noticeably. AI tools helped speed up code edits and template content, but I verified behavior by running the script and reading explanations. I was surprised how quickly simple algorithms can produce plausible suggestions and how fragile those suggestions are when the dataset is small or biased. Next I would add interaction logs and embeddings to move from handcrafted rules to learned models.
