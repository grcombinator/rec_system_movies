"""Streamlit user site: pick movies you like, get recommendations.

Run from the repo root:
    .venv/Scripts/python.exe -m streamlit run ui/app.py

Standalone: loads movies.csv + popularity.csv directly, no API process needed.
Scoring mirrors src/api.main (popularity + saved BPR/HF weights + genre overlap,
weights default to artifacts/ensemble_weights.json).
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
# Aligned row-wise with BPR_MAPPING; zero rows = movie without TMDB embedding.
HF_VECTORS = BASE_DIR / "artifacts" / "hf_item_vectors.npy"
ENSEMBLE_WEIGHTS = BASE_DIR / "artifacts" / "ensemble_weights.json"


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

    with open(BPR_MAPPING, "rb") as f:
        mapping = pickle.load(f)
    return np.load(BPR_VECTORS).astype("float32"), mapping


@st.cache_data
def load_hf():
    """Load saved HF content vectors. Returns matrix or None (zero rows = no embedding)."""
    if not HF_VECTORS.exists():
        return None
    return np.load(HF_VECTORS).astype("float32")


def load_weights():
    """Trained blend weights; fall back to built-ins if the file is missing."""
    import json

    defaults = {"w_pop": 0.35, "w_cf": 0.35, "w_hf": 0.2, "w_g": 0.1}
    if not ENSEMBLE_WEIGHTS.exists():
        return defaults
    try:
        with open(ENSEMBLE_WEIGHTS) as f:
            w = json.load(f)
        return {
            "w_pop": float(w.get("w_pop", defaults["w_pop"])),
            "w_cf": float(w.get("w_cf", defaults["w_cf"])),
            "w_hf": float(w.get("w_hf", defaults["w_hf"])),
            "w_g": float(w.get("w_g", defaults["w_g"])),
        }
    except Exception:
        return defaults


def score_candidates(movies, pop_scores, item_genre, liked, seen, top_k, w_pop, w_genre, w_bpr=0.0,
                     bpr_vectors=None, bpr_mapping=None, w_hf=0.0, hf_vectors=None):
    """Blend of normalized popularity, BPR embedding match, HF content match and genre overlap."""
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
    hf_user = None
    if hf_vectors is not None and w_hf > 0 and bpr_mapping:
        hf_idxs = [
            bpr_mapping[m]
            for m in liked
            if m in bpr_mapping and float(hf_vectors[bpr_mapping[m]] @ hf_vectors[bpr_mapping[m]]) > 1e-9
        ]
        if hf_idxs:
            hf_user = hf_vectors[np.array(hf_idxs)].mean(axis=0)
            hf_norm = float(hf_user @ hf_user)
            hf_user = hf_user / (np.sqrt(hf_norm) + 1e-9) if hf_norm > 1e-9 else None
    use_hf = hf_user is not None
    total = (w_pop + w_genre + (w_bpr if use_bpr else 0.0) + (w_hf if use_hf else 0.0)) or 1.0
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
    hf_raw = {}
    if use_hf:
        vals = []
        for row in movies.itertuples():
            mid = int(row.movieId)
            if mid in seen or mid not in bpr_mapping:
                continue
            item_vec = hf_vectors[bpr_mapping[mid]]
            if float(item_vec @ item_vec) <= 1e-9:
                continue
            v = float(hf_user @ item_vec)
            hf_raw[mid] = v
            vals.append(v)
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        span = (hi - lo) if hi > lo else 1.0
        hf_raw = {m: (v - lo) / span for m, v in hf_raw.items()}
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
        if use_hf:
            score += (w_hf * hf_raw.get(mid, 0.0)) / total
        scored.append({"movieId": mid, "title": row.title, "genres": row.genres, "score": round(float(score), 4)})
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:top_k]


st.set_page_config(page_title="Movie Recommender", layout="wide")

try:
    movies, pop_scores, item_genre = load_catalog()
    bpr_vectors, bpr_mapping = load_bpr()
    hf_vectors = load_hf()
    weights = load_weights()
except FileNotFoundError as e:
    st.error(f"Catalog file missing: {e}. Run from the repo root.")
    st.stop()

if bpr_vectors is None:
    st.warning("BPR weights not found (artifacts/). Serving popularity + genre only. Run train_bpr_hybrid.py once.")
if hf_vectors is None:
    st.warning("HF weights not found (artifacts/hf_item_vectors.npy). Content branch disabled.")

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

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    top_k = st.slider("How many", 5, 30, 10)
with col2:
    w_pop = st.slider("Popularity weight", 0.0, 1.0, weights["w_pop"])
with col3:
    w_bpr = st.slider("BPR weight", 0.0, 1.0, weights["w_cf"], disabled=bpr_vectors is None)
with col4:
    w_hf = st.slider("Content weight", 0.0, 1.0, weights["w_hf"], disabled=hf_vectors is None)
with col5:
    w_genre = st.slider("Genre match weight", 0.0, 1.0, weights["w_g"])

if st.button("Recommend", type="primary", disabled=not liked_ids):
    items = score_candidates(movies, pop_scores, item_genre, liked_ids, set(liked_ids),
                             top_k, w_pop, w_genre, w_bpr, bpr_vectors, bpr_mapping,
                             w_hf, hf_vectors)
    st.subheader("Recommended for you")
    for i, d in enumerate(items, 1):
        st.write(f"**{i}. {d['title']}** — `{d['genres']}` — score {d['score']:.3f}")

st.subheader("Top popular right now")
for i, d in enumerate(score_candidates(movies, pop_scores, item_genre, [], set(), 10, 1.0, 0.0), 1):
    st.write(f"**{i}. {d['title']}** — `{d['genres']}`")
