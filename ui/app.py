"""Streamlit user site: pick movies you like, get recommendations.

Run from the repo root:
    .venv/Scripts/python.exe -m streamlit run ui/app.py

Standalone: loads movies.csv + popularity.csv directly, no API process needed.
Scoring mirrors src/api.main (popularity + saved BPR weights + genre overlap).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MOVIES_CSV = BASE_DIR / "data" / "raw" / "movies.csv"
POPULARITY_CSV = BASE_DIR / "data" / "processed" / "popularity.csv"
BPR_VECTORS = BASE_DIR / "artifacts" / "bpr_item_vectors.npy"
BPR_MAPPING = BASE_DIR / "artifacts" / "bpr_movie_to_idx.pkl"


@st.cache_data
def load_catalog():
    """Load movies and popularity table. Returns (movies, pop_scores, item_genre)."""
    movies = pd.read_csv(MOVIES_CSV)
    try:
        pop = pd.read_csv(POPULARITY_CSV)
        pop_scores = dict(zip(pop["movieId"].astype(int), pop["score"].astype(float)))
    except FileNotFoundError:
        pop_scores = {}
    vocab = sorted({g.strip() for gs in movies["genres"].fillna("") for g in str(gs).split("|") if g.strip()})
    g2i = {g: i for i, g in enumerate(vocab)}
    item_genre = {}
    for row in movies.itertuples():
        vec = [0.0] * len(vocab)
        for g in str(row.genres).split("|"):
            g = g.strip()
            if g in g2i:
                vec[g2i[g]] = 1.0
        item_genre[int(row.movieId)] = vec
    return movies, pop_scores, item_genre


@st.cache_data
def load_bpr():
    """Load saved BPR weights (written by train_bpr_hybrid.py). Returns (vectors, mapping) or (None, {})."""
    if not BPR_VECTORS.exists() or not BPR_MAPPING.exists():
        return None, {}
    import pickle

    return np.load(BPR_VECTORS).astype("float32"), pickle.load(open(BPR_MAPPING, "rb"))


def score_candidates(movies, pop_scores, item_genre, liked, seen, top_k, w_pop, w_genre, w_bpr=0.0,
                     bpr_vectors=None, bpr_mapping=None):
    """Blend of normalized popularity, BPR embedding match and genre overlap."""
    dim = len(next(iter(item_genre.values()))) if item_genre else 0
    profile = [0.0] * dim
    n = 0
    for mid in liked:
        vec = item_genre.get(int(mid))
        if vec is not None:
            profile = [p + v for p, v in zip(profile, vec)]
            n += 1
    if n > 0:
        profile = [p / n for p in profile]
    bpr_user = None
    if bpr_vectors is not None and w_bpr > 0:
        idxs = [bpr_mapping[m] for m in liked if m in bpr_mapping]
        if idxs:
            bpr_user = bpr_vectors[np.array(idxs)].mean(axis=0)
    use_bpr = bpr_user is not None
    total = (w_pop + w_genre + (w_bpr if use_bpr else 0.0)) or 1.0
    max_pop = max(pop_scores.values()) if pop_scores else 1.0
    bpr_raw = {}
    if use_bpr:
        vals = []
        for row in movies.itertuples():
            mid = int(row.movieId)
            if mid in seen or mid not in bpr_mapping:
                continue
            v = float(bpr_user @ bpr_vectors[bpr_mapping[mid]])
            bpr_raw[mid] = v
            vals.append(v)
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        span = (hi - lo) if hi > lo else 1.0
        bpr_raw = {m: (v - lo) / span for m, v in bpr_raw.items()}
    scored = []
    for row in movies.itertuples():
        mid = int(row.movieId)
        if mid in seen:
            continue
        pop_norm = pop_scores.get(mid, 0.0) / max_pop
        vec = item_genre.get(mid, [0.0] * dim)
        active = len([x for x in profile if x > 0]) or 1
        genre_score = sum(p * v for p, v in zip(profile, vec)) / active
        score = (w_pop * pop_norm + w_genre * (genre_score if n > 0 else 0.0)) / total
        if use_bpr:
            score += (w_bpr * bpr_raw.get(mid, 0.0)) / total
        scored.append({"movieId": mid, "title": row.title, "genres": row.genres, "score": round(float(score), 4)})
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:top_k]


st.set_page_config(page_title="Movie Recommender", layout="wide")

try:
    movies, pop_scores, item_genre = load_catalog()
    bpr_vectors, bpr_mapping = load_bpr()
except FileNotFoundError as e:
    st.error(f"Catalog file missing: {e}. Run from the repo root.")
    st.stop()

if bpr_vectors is None:
    st.warning("BPR weights not found (artifacts/). Serving popularity + genre only. Run train_bpr_hybrid.py once.")

st.title("Movie Recommender")
st.caption("Pick movies you like and get personal recommendations. New here? Top popular below needs no input.")

query = st.text_input("Search movie", placeholder="e.g. Matrix, Pulp Fiction, Toy Story")
pool = movies
if query.strip():
    pool = movies[movies["title"].str.contains(query.strip(), case=False, na=False)]
pool = pool.head(50)

options = {f"{r.title}  [{r.movieId}]": int(r.movieId) for r in pool.itertuples()}
picked = st.multiselect("Movies you like", list(options.keys()))
liked_ids = [options[p] for p in picked]

col1, col2, col3, col4 = st.columns(4)
with col1:
    top_k = st.slider("How many", 5, 30, 10)
with col2:
    w_pop = st.slider("Popularity weight", 0.0, 1.0, 0.6)
with col3:
    w_bpr = st.slider("BPR weight", 0.0, 1.0, 0.35, disabled=bpr_vectors is None)
with col4:
    w_genre = st.slider("Genre match weight", 0.0, 1.0, 0.4)

if st.button("Recommend", type="primary", disabled=not liked_ids):
    items = score_candidates(movies, pop_scores, item_genre, liked_ids, set(liked_ids),
                             top_k, w_pop, w_genre, w_bpr, bpr_vectors, bpr_mapping)
    st.subheader("Recommended for you")
    for i, d in enumerate(items, 1):
        st.write(f"**{i}. {d['title']}** — `{d['genres']}` — score {d['score']:.3f}")

st.subheader("Top popular right now")
for i, d in enumerate(score_candidates(movies, pop_scores, item_genre, [], set(), 10, 1.0, 0.0), 1):
    st.write(f"**{i}. {d['title']}** — `{d['genres']}`")
