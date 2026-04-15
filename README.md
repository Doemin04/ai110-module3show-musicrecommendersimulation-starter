# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

This project simulates how a simple content-based music recommender connects a user's taste to song attributes. Real-world systems combine collaborative signals (other users' behavior), content signals (audio + metadata), and contextual priors; this simulation will focus on content signals (the song's numeric "vibe" features and categorical tags) and a small popularity prior that acts as a fallback.

Step 1 — Catalog and extended data
- I inspected `data/songs.csv` (now 18 songs). Available fields are: id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability, acousticness. I added 8 diverse songs (IDs 11–18) covering classical, metal, reggae, hip hop, country, blues, electronic, and folk so the simulator can exercise different textures and moods.
- Suggested additional numeric features (optional for later):
  - instrumentalness (0–1): how much of the track is instrumental (helps separate vocal hip-hop from instrumental ambient)
  - liveness (0–1): how much the track sounds live vs studio (useful for folk/blues/rock)
  - speechiness (0–1): presence of spoken words/rap (useful for hip hop / podcasts)
  - loudness_db (numeric): average loudness in dB (normalize before use)
  - tempo_variability (0–1): how stable the tempo is (helps detect electronic loops vs human-played tempos)

Step 2 — Example UserProfile (taste sketch)
- name: "Evening Chill"
- preferred_genres: {lofi, ambient, folk}
- preferred_moods: {chill, contemplative}
- numeric_preferences:
  - energy: 0.35
  - valence: 0.60
  - danceability: 0.45
  - acousticness: 0.85
  - tempo_bpm: 75   # target BPM (we normalize to dataset tempo range in code)
- α (content vs popularity) = 0.85  # prefer content matching over popularity

Profile critique: this profile is intentionally specific to favor low-energy, acoustic, contemplative tracks. It will clearly differentiate "intense rock" (energy ~0.9, low acousticness) from "chill lofi" (energy ~0.3–0.4, high acousticness) because numeric features like energy and acousticness are strongly different. However, it may be too narrow if the user also likes occasional higher-energy tracks of a preferred mood; add a per-feature tolerance σ_f or allow multiple preferred_genres to increase flexibility.

Step 3 — Finalized Algorithm Recipe (Scoring + Ranking)
Overview: For each song S and user profile P we produce a FinalScore(S|P) in [0,1] and return the top-K by FinalScore. FinalScore blends a content-based score with a small popularity prior.

1) Per-feature closeness (numeric features)
- Normalize numeric features into [0,1] (tempo normalized by dataset min/max or by mapping BPM to perceptual range).
- Use a Gaussian kernel for closeness:
  score_f = exp( - (s_f - u_f)^2 / (2 * σ_f^2) )
  - σ_f controls tolerance. Suggested defaults: σ_energy=0.12, σ_valence=0.15, σ_danceability=0.12, σ_acousticness=0.15, σ_tempo=0.10 (tempo after scaling to 0–1).

2) Categorical matching (genre/mood)
- genre_match = 1.0 if song.genre ∈ user.preferred_genres else 0.0 (or soft similarity if genres are embedded).
- mood_match = 1.0 if any mood tag overlaps else 0.0 (or Jaccard for multi-label moods).

3) Per-song ContentScore (combine features)
- Weighted sum (weights chosen to reflect importance; numeric weights sum to 1 in the numeric block):
  ContentScore = (W_genre * genre_match + W_mood * mood_match + sum_{f in numeric} w_f * score_f) / Z
  where Z is the maximum possible (to keep ContentScore in [0,1]).

Suggested weights (starting defaults):
- W_genre = 0.30   # genre is structural and informative
- W_mood = 0.12    # mood tags helpful but noisier
- Numeric block (total = 0.58): energy 0.18, valence 0.15, danceability 0.10, acousticness 0.10, tempo 0.05

Note: these values reflect the small catalog in `data/songs.csv`; genre is deliberately stronger than a single mood tag because genre often encodes instrumentation and production that numeric features don't fully capture.

4) Popularity prior and final blending
- If PopScore is available (0–1), compute FinalScore = α * ContentScore + (1−α) * PopScore (α from profile or system default e.g., 0.85).

5) Ranking rule and list-level adjustments
- Sort candidates by FinalScore descending.
- Optional re-ranking: apply an MMR-style step to improve diversity:
  pick next item x maximizing λ*FinalScore(x) - (1−λ)*max_{selected y} sim(x,y), with λ≈0.8.
- Apply simple dedup rules (no more than 2 songs by the same artist in top-10).

Practical scoring example (point form for intuition):
- +2.0 points for genre match (equivalent to W_genre=0.30 after normalization)
- +1.0 point for mood match (≈W_mood)
- numeric closeness contributes up to +2.0 points total (split among features, scaled by closeness)
- Normalize total to [0,1] for blending with PopScore.

Step 4 — Data flow (visual)
```mermaid
flowchart TD
  A[User Preferences] --> B[Load CSV Catalog]
  B --> C[For each song: compute per-feature closeness]
  C --> D[Combine with weights -> ContentScore]
  D --> E[Blend with PopScore (α)]
  E --> F[Sort by FinalScore]
  F --> G[Optional: Diversity re-rank / dedup]
  G --> H[Top-K Recommendations]
```

Step 5 — Biases and notes
- This simple content-based system will naturally favor songs that share explicit genre tags and numeric features with the user; it may under-recommend songs that match mood but not genre (risk: over-prioritize genre). Because the numeric features are hand-tuned and the catalog is small, the model may overfit to the dataset's genre/mood distribution. Be explicit about tolerances (σ_f) so users who like variety can be served more diverse results.

📍Checkpoint: the plan defines the extended catalog, a concrete UserProfile example, the per-song Scoring Rule (Gaussian closeness + categorical boosts), the FinalScore blending (α), and the list-level Ranking Rule (sort + optional MMR). With this, we're ready to implement the scorer and a small re-ranker in code.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

