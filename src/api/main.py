"""
Serving API for the recommender.

Design note (where Postgres fits and where it does not):
- Offline training reads parquet files directly; Postgres is NOT on the
  CSV -> parquet training path (it adds nothing there).
- Postgres IS appropriate here: movie catalog persistence and
  recommendation request logging for the online service.
- Scoring is stateless and lightweight: popularity + genre overlap + the
  saved BPR/HF item vectors (trained offline, loaded from artifacts/).
  Blend weights default to artifacts/ensemble_weights.json.
  No retraining happens at serve time.
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, JSON, MetaData, Table, create_engine, func

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MOVIES_CSV = BASE_DIR / "data" / "raw" / "movies.csv"
POPULARITY_CSV = BASE_DIR / "data" / "processed" / "popularity.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
BPR_VECTORS = ARTIFACTS_DIR / "bpr_item_vectors.npy"
BPR_MAPPING = ARTIFACTS_DIR / "bpr_movie_to_idx.pkl"
HF_VECTORS = ARTIFACTS_DIR / "hf_item_vectors.npy"
ENSEMBLE_WEIGHTS = ARTIFACTS_DIR / "ensemble_weights.json"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:admin@postgres:5432/recsys_db")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_catalog()
    _load_bpr()
    _load_hf()
    _load_ensemble_weights()
    yield


app = FastAPI(title="Hybrid RecSys API", version="0.2.0", lifespan=lifespan)

# --- in-memory serving state (loaded once at startup) ---
import pandas as pd  # local import so the module stays import-safe without pandas

MOVIES: Optional[pd.DataFrame] = None
POP_SCORE: dict = {}
GENRE_VOCAB: list = []
ITEM_GENRE: dict = {}
BPR_ITEM_VECTORS = None
BPR_MOVIE_TO_IDX: dict = {}
# HF content vectors share the row order of BPR_MOVIE_TO_IDX (zero rows = no embedding).
HF_ITEM_VECTORS = None
# Filled from ensemble_weights.json at startup; fall back to these if the file is missing.
ENSEMBLE_DEFAULTS: dict = {"w_pop": 0.35, "w_cf": 0.35, "w_hf": 0.2, "w_g": 0.1}


def _load_catalog() -> None:
    """Load the small serving catalog: movies.csv + precomputed popularity.csv."""
    global MOVIES, POP_SCORE, GENRE_VOCAB, ITEM_GENRE
    if not MOVIES_CSV.exists():
        logging.warning(f"Movies catalog not found at {MOVIES_CSV}; serving in degraded mode.")
        MOVIES = pd.DataFrame(columns=["movieId", "title", "genres"])
        return
    MOVIES = pd.read_csv(MOVIES_CSV)
    if POPULARITY_CSV.exists():
        pop = pd.read_csv(POPULARITY_CSV)
        POP_SCORE = dict(zip(pop["movieId"].astype(int), pop["score"].astype(float)))
    else:
        logging.warning(f"{POPULARITY_CSV} missing; all popularity scores default to 0.")
        POP_SCORE = {}
    vocab = sorted({g.strip() for gs in MOVIES["genres"].fillna("") for g in str(gs).split("|") if g.strip()})
    GENRE_VOCAB = vocab
    g2i = {g: i for i, g in enumerate(vocab)}
    for _, row in MOVIES.iterrows():
        vec = [0.0] * len(vocab)
        for g in str(row["genres"]).split("|"):
            g = g.strip()
            if g in g2i:
                vec[g2i[g]] = 1.0
        ITEM_GENRE[int(row["movieId"])] = vec
    logging.info(f"Catalog loaded: {len(MOVIES)} movies, {len(vocab)} genres.")


def _load_bpr() -> None:
    """Load saved BPR item vectors (written by train_bpr_hybrid.py). Optional."""
    global BPR_ITEM_VECTORS, BPR_MOVIE_TO_IDX
    if not BPR_VECTORS.exists() or not BPR_MAPPING.exists():
        logging.warning("BPR artifacts missing; serving popularity + genre only. Run train_bpr_hybrid.py once.")
        BPR_ITEM_VECTORS, BPR_MOVIE_TO_IDX = None, {}
        return
    import pickle

    import numpy as np

    BPR_ITEM_VECTORS = np.load(BPR_VECTORS).astype("float32")
    with open(BPR_MAPPING, "rb") as f:
        BPR_MOVIE_TO_IDX = pickle.load(f)
    logging.info(f"BPR weights loaded: {BPR_ITEM_VECTORS.shape[0]} items, dim {BPR_ITEM_VECTORS.shape[1]}.")


def _load_hf() -> None:
    """Load saved HF content vectors (written by train_bpr_hybrid.py). Optional."""
    global HF_ITEM_VECTORS
    if not HF_VECTORS.exists():
        logging.warning("HF artifacts missing; serving without content branch. Run train_bpr_hybrid.py once.")
        HF_ITEM_VECTORS = None
        return
    import numpy as np

    HF_ITEM_VECTORS = np.load(HF_VECTORS).astype("float32")
    if BPR_MOVIE_TO_IDX and HF_ITEM_VECTORS.shape[0] != len(BPR_MOVIE_TO_IDX):
        logging.warning(
            f"HF/BPR size mismatch ({HF_ITEM_VECTORS.shape[0]} vs {len(BPR_MOVIE_TO_IDX)}); "
            "HF branch disabled."
        )
        HF_ITEM_VECTORS = None
        return
    logging.info(f"HF weights loaded: {HF_ITEM_VECTORS.shape[0]} items, dim {HF_ITEM_VECTORS.shape[1]}.")


def _load_ensemble_weights() -> None:
    """Read trained blend weights; apply them as API request defaults."""
    global ENSEMBLE_DEFAULTS
    if not ENSEMBLE_WEIGHTS.exists():
        logging.warning("ensemble_weights.json missing; using built-in weight defaults.")
        return
    import json

    try:
        with open(ENSEMBLE_WEIGHTS) as f:
            w = json.load(f)
        ENSEMBLE_DEFAULTS = {
            "w_pop": float(w.get("w_pop", ENSEMBLE_DEFAULTS["w_pop"])),
            "w_cf": float(w.get("w_cf", ENSEMBLE_DEFAULTS["w_cf"])),
            "w_hf": float(w.get("w_hf", ENSEMBLE_DEFAULTS["w_hf"])),
            "w_g": float(w.get("w_g", ENSEMBLE_DEFAULTS["w_g"])),
        }
        # Pydantic v2: update field defaults so /docs shows trained weights.
        for field, key in (("w_pop", "w_pop"), ("w_bpr", "w_cf"), ("w_hf", "w_hf"), ("w_genre", "w_g")):
            if field in RecommendRequest.model_fields:
                RecommendRequest.model_fields[field].default = ENSEMBLE_DEFAULTS[key]
        logging.info(f"Ensemble weights loaded: {ENSEMBLE_DEFAULTS} (cf={w.get('cf')}).")
    except Exception as e:
        logging.warning(f"Could not parse ensemble_weights.json: {e}")


# --- Postgres logging (best-effort: API works even if DB is down) ---
metadata = MetaData()
recommendation_logs = Table(
    "recommendation_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, nullable=True),
    Column("request", JSON, nullable=False),
    Column("response", JSON, nullable=False),
    Column("created_at", DateTime, server_default=func.now()),
)


def _get_engine():
    try:
        return create_engine(DATABASE_URL, pool_pre_ping=True)
    except Exception as e:
        logging.warning(f"Could not create DB engine: {e}")
        return None


def _log_request(user_id: Optional[int], request: dict, response: list) -> None:
    engine = _get_engine()
    if engine is None:
        return
    try:
        with engine.begin() as conn:
            metadata.create_all(conn, tables=[recommendation_logs], checkfirst=True)
            conn.execute(
                recommendation_logs.insert().values(user_id=user_id, request=request, response=response)
            )
    except Exception as e:
        logging.warning(f"DB log skipped (DB unavailable): {e}")


def _ensure_loaded() -> None:
    if MOVIES is None:
        _load_catalog()
    if BPR_ITEM_VECTORS is None and not BPR_MOVIE_TO_IDX:
        _load_bpr()
    if HF_ITEM_VECTORS is None:
        _load_hf()
    _load_ensemble_weights()


class RecommendRequest(BaseModel):
    user_id: Optional[int] = Field(default=None, description="Internal user id, optional for cold start")
    liked_movie_ids: list[int] = Field(default_factory=list, description="Movies the user liked")
    seen_movie_ids: list[int] = Field(default_factory=list, description="Movies to exclude from output")
    top_k: int = Field(default=10, ge=1, le=100)
    w_pop: float = Field(default=0.35, ge=0.0, le=1.0)
    w_bpr: float = Field(default=0.35, ge=0.0, le=1.0, description="BPR term weight (0 disables it)")
    w_hf: float = Field(default=0.2, ge=0.0, le=1.0, description="HF content term weight (0 disables it)")
    w_genre: float = Field(default=0.1, ge=0.0, le=1.0)


class RootResponse(BaseModel):
    service: str = Field(examples=["Hybrid RecSys API"])
    docs: str = Field(examples=["/docs"])
    health: str = Field(examples=["/health"])
    endpoints: list[str]


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    movies_loaded: int = Field(examples=[62423])
    bpr_loaded: bool = Field(examples=[True])
    hf_loaded: bool = Field(examples=[True])
    weights: dict = Field(default_factory=dict)


class MovieResponse(BaseModel):
    movieId: int = Field(examples=[356])
    title: str = Field(examples=["Forrest Gump (1994)"])
    genres: str = Field(examples=["Drama|Romance"])


class RecommendItem(BaseModel):
    movieId: int = Field(examples=[296])
    title: str = Field(examples=["Pulp Fiction (1994)"])
    genres: str = Field(examples=["Comedy|Crime|Drama|Thriller"])
    score: float = Field(examples=[0.85])


def _score_candidates(
    liked: list[int],
    seen: set[int],
    top_k: int,
    w_pop: float,
    w_genre: float,
    w_bpr: float = 0.0,
    w_hf: float = 0.0,
) -> list[dict]:
    assert MOVIES is not None
    # genre profile = mean of liked item vectors
    dim = len(GENRE_VOCAB)
    profile = [0.0] * dim
    n = 0
    for mid in liked:
        vec = ITEM_GENRE.get(int(mid))
        if vec is not None:
            profile = [p + v for p, v in zip(profile, vec)]
            n += 1
    if n > 0:
        profile = [p / n for p in profile]
    # BPR user vector = mean of liked item vectors in the trained embedding space.
    # Items outside the training mapping are skipped; with no mapped likes the
    # BPR term is disabled and remaining weights are renormalized.
    import numpy as np

    bpr_user = None
    if BPR_ITEM_VECTORS is not None and w_bpr > 0:
        idxs = [BPR_MOVIE_TO_IDX[m] for m in liked if m in BPR_MOVIE_TO_IDX]
        if idxs:
            bpr_user = BPR_ITEM_VECTORS[np.array(idxs)].mean(axis=0)
    use_bpr = bpr_user is not None
    # HF content user vector = mean of liked HF item vectors.
    # Zero rows mean "movie without TMDB embedding" and are skipped.
    hf_user = None
    if HF_ITEM_VECTORS is not None and w_hf > 0 and BPR_MOVIE_TO_IDX:
        hf_idxs = [
            BPR_MOVIE_TO_IDX[m]
            for m in liked
            if m in BPR_MOVIE_TO_IDX and float(np.dot(HF_ITEM_VECTORS[BPR_MOVIE_TO_IDX[m]], HF_ITEM_VECTORS[BPR_MOVIE_TO_IDX[m]])) > 1e-9
        ]
        if hf_idxs:
            hf_user = HF_ITEM_VECTORS[np.array(hf_idxs)].mean(axis=0)
            hf_norm = float(np.dot(hf_user, hf_user))
            if hf_norm <= 1e-9:
                hf_user = None
            else:
                hf_user = hf_user / (np.sqrt(hf_norm) + 1e-9)
    use_hf = hf_user is not None
    total = w_pop + w_genre + (w_bpr if use_bpr else 0.0) + (w_hf if use_hf else 0.0)
    total = total if total > 0 else 1.0
    max_pop = max(POP_SCORE.values()) if POP_SCORE else 1.0
    # raw BPR scores for mapped candidates, min-max normalized per request
    bpr_raw: dict = {}
    if use_bpr:
        vals = []
        for _, row in MOVIES.iterrows():
            mid = int(row["movieId"])
            if mid in seen or mid not in BPR_MOVIE_TO_IDX:
                continue
            v = float(bpr_user @ BPR_ITEM_VECTORS[BPR_MOVIE_TO_IDX[mid]])
            bpr_raw[mid] = v
            vals.append(v)
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        span = (hi - lo) if hi > lo else 1.0
        bpr_raw = {m: (v - lo) / span for m, v in bpr_raw.items()}
    # raw HF cosine scores for candidates with embeddings, min-max normalized per request
    hf_raw: dict = {}
    if use_hf:
        vals = []
        for _, row in MOVIES.iterrows():
            mid = int(row["movieId"])
            if mid in seen or mid not in BPR_MOVIE_TO_IDX:
                continue
            item_vec = HF_ITEM_VECTORS[BPR_MOVIE_TO_IDX[mid]]
            if float(item_vec @ item_vec) <= 1e-9:
                continue
            v = float(hf_user @ item_vec)
            hf_raw[mid] = v
            vals.append(v)
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        span = (hi - lo) if hi > lo else 1.0
        hf_raw = {m: (v - lo) / span for m, v in hf_raw.items()}
    scored = []
    for _, row in MOVIES.iterrows():
        mid = int(row["movieId"])
        if mid in seen:
            continue
        pop_norm = POP_SCORE.get(mid, 0.0) / max_pop
        vec = ITEM_GENRE.get(mid, [0.0] * dim)
        genre_score = sum(p * v for p, v in zip(profile, vec)) / (max(len([x for x in profile if x > 0]), 1))
        score = (w_pop * pop_norm + w_genre * (genre_score if n > 0 else 0.0)) / total
        if use_bpr:
            score += (w_bpr * bpr_raw.get(mid, 0.0)) / total
        if use_hf:
            score += (w_hf * hf_raw.get(mid, 0.0)) / total
        scored.append({"movieId": mid, "title": row["title"], "genres": row["genres"], "score": round(float(score), 6)})
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:top_k]


@app.get("/", response_model=RootResponse, summary="Root")
def root() -> RootResponse:
    return RootResponse(
        service="Hybrid RecSys API",
        docs="/docs",
        health="/health",
        endpoints=["GET /health", "GET /movies/{movie_id}", "GET /recommend/popular", "POST /recommend"],
    )


@app.get("/health", response_model=HealthResponse, summary="Health")
def health() -> HealthResponse:
    _ensure_loaded()
    return HealthResponse(
        status="ok",
        movies_loaded=int(len(MOVIES)) if MOVIES is not None else 0,
        bpr_loaded=BPR_ITEM_VECTORS is not None,
        hf_loaded=HF_ITEM_VECTORS is not None,
        weights=dict(ENSEMBLE_DEFAULTS),
    )


@app.get("/movies/{movie_id}", response_model=MovieResponse, summary="Movie")
def get_movie(movie_id: int) -> MovieResponse:
    _ensure_loaded()
    if MOVIES is None or MOVIES.empty:
        raise HTTPException(status_code=503, detail="Catalog not loaded")
    hit = MOVIES[MOVIES["movieId"] == movie_id]
    if hit.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    row = hit.iloc[0]
    return MovieResponse(movieId=int(row["movieId"]), title=row["title"], genres=row["genres"])


@app.get("/recommend/popular", response_model=list[RecommendItem], summary="Popular")
def recommend_popular(top_k: int = 10) -> list[RecommendItem]:
    _ensure_loaded()
    if MOVIES is None:
        raise HTTPException(status_code=503, detail="Catalog not loaded")
    return [RecommendItem(**d) for d in _score_candidates([], set(), min(top_k, 100), w_pop=1.0, w_genre=0.0, w_bpr=0.0, w_hf=0.0)]


@app.post("/recommend", response_model=list[RecommendItem], summary="Recommend")
def recommend(req: RecommendRequest) -> list[RecommendItem]:
    _ensure_loaded()
    if MOVIES is None:
        raise HTTPException(status_code=503, detail="Catalog not loaded")
    seen = set(req.seen_movie_ids) | set(req.liked_movie_ids)
    items = _score_candidates(req.liked_movie_ids, seen, req.top_k, req.w_pop, req.w_genre, req.w_bpr, req.w_hf)
    _log_request(req.user_id, req.model_dump(), items)
    return [RecommendItem(**d) for d in items]
