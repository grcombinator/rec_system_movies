# Hybrid-recsys

Movie recommender on MovieLens ratings with TMDB descriptions. Plain SVD trained on rating prediction turned out weak here (recall@10 0.017, worse than a popularity list at 0.055), so the final model is a BPR two-tower network blended with popularity and content signals. That ensemble reaches recall@10 0.0645 — about 4x the SVD baseline and 18% above the strongest single baseline, with the gap holding up under a paired significance test. Everything is compared under full-ranking evaluation (top-10 out of the whole catalog, seen items excluded), and a FastAPI service serves the model with Postgres request logging.

## Data

Raw dumps go in `data/raw/` (not committed): `ratings.csv`, `movies.csv`, `links.csv`, `tmdb_5000_movies.csv`. Processed files live in `data/processed/`:

- `interactions.parquet` — ratings joined with titles/genres, movies with fewer than 50 ratings removed
- `movie_embeddings.pkl` — MiniLM (`all-MiniLM-L6-v2`) embeddings for matched TMDB movies
- `popularity.csv` — precomputed per-movie counts, used by the API (small enough to commit)

## Setup

```
pip install -e .
docker compose up -d postgres qdrant mlflow
```

Copy `.env.example` to `.env` and adjust if needed. MLflow tracks to `mlflow.db` (sqlite), no server required for the training scripts.

## Pipeline

Run from the repo root. Everything below uses the same 15k-user sample (seed 42) by default so numbers stay comparable; set `SAMPLE_USERS=0` for the full 24.6M-row data (needs 30+ min and plenty of RAM).

1. Baselines:
```
.venv/Scripts/python.exe src/models/evaluate_baselines.py
```
Random, popularity and SVD scored over the full catalog with train items excluded.

2. SVD + genre + popularity ensemble:
```
.venv/Scripts/python.exe src/models/evaluate_ensemble_svd_genre.py
```

3. Final model — BPR two-tower with HF embeddings:
```
.venv/Scripts/python.exe src/models/train_bpr_hybrid.py
```
This is where the main model is trained (`src/models/train_bpr_hybrid.py`): tuned SVD for reference, MiniLM cosine profiles, then the torch BPR two-tower (64-dim, 4 negatives, 6 epochs), then a small weight grid for the final blend. Takes about 7 minutes on CPU for the 15k-user sample; the full 24.6M-row data would take hours, which is why the sample is the default.

Training writes `artifacts/` once: `bpr_item_vectors.npy` (3.2MB), `bpr_movie_to_idx.pkl` and `ensemble_weights.json`. Serving never retrains — the API and the Streamlit site load these files at startup and build the user vector on the fly as the mean of liked item vectors. If `artifacts/` is missing they fall back to popularity + genre. The directory is git-ignored; rerun the script to regenerate it.

## Results

recall@10 / ndcg@10 on the 15k-user sample (paired 95% CI, all gains over baselines significant):

| model | recall@10 | ndcg@10 |
|---|---|---|
| random | 0.0009 | 0.0013 |
| popularity | 0.0546 | 0.0725 |
| SVD (50 factors, MSE) | 0.0168 | 0.0281 |
| SVD + genre (alpha 0.8) | 0.0239 | 0.0369 |
| ensemble (pop + SVD + genre) | 0.0610 | 0.0804 |
| BPR two-tower (torch) | 0.0542 | 0.0696 |
| final (pop + BPR + HF + genre) | 0.0645 | 0.0830 |


What the numbers say: SVD trained on MSE underperforms popularity here; switching the loss to BPR closes the gap (0.0168 -> 0.0542); content features alone are weak but add a few points on top in an ensemble.

## API

```
.venv/Scripts/python.exe -m uvicorn src.api.main:app --port 8000
```

Docs at `http://127.0.0.1:8000/docs`.

- `GET /health` — status and catalog size
- `GET /movies/{movie_id}` — title and genres
- `GET /recommend/popular?top_k=10` — cold-start top
- `POST /recommend` — `{"liked_movie_ids": [356, 318], "seen_movie_ids": [...], "top_k": 10, "w_pop": 0.6, "w_bpr": 0.35, "w_genre": 0.4}`

Scoring is `w_pop * popularity + w_bpr * BPR match + w_genre * genre overlap`, seen items excluded, weights renormalized if BPR artifacts are missing. `GET /health` reports `bpr_loaded` so you can check the weights were picked up. Each request is logged to the `recommendation_logs` table if Postgres is up; otherwise logging is skipped and the request still returns 200.

A Streamlit demo mirrors the API scoring: `.venv/Scripts/python.exe -m streamlit run ui/app.py`.

## Layout

```
src/api/      FastAPI service (serving only, no training logic)
src/models/   eval_utils (full-ranking eval), evaluate_baselines,
              evaluate_ensemble_svd_genre, train_bpr_hybrid, utils (data splits)
data/         raw dumps (ignored) + processed artifacts
docker-compose.yml   postgres, qdrant, mlflow, api
Dockerfile.api       API image
```

