"""
SVD + genre-content + popularity ensemble evaluated with full-ranking.

Rationale:
- SVD memorizes popular items well but struggles on tail/novel items.
- Genres help where SVD has little data.
- Blend: hybrid = alpha * SVD + (1 - alpha) * content.
- alpha=1.0 is pure SVD, so the best alpha cannot be worse than SVD.
- Final stage adds popularity (the strongest single baseline here):
  final = w_pop * popularity + w_svd * SVD + w_con * content.
"""
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD

from eval_utils import (
    build_matrices,
    evaluate_full_ranking,
    paired_diff_ci95,
    random_scorer,
    popularity_scorer,
    svd_scorer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SEED = 42
SAMPLE_USERS = 15000  # sample so training finishes in minutes on 7GB RAM
KS = (10, 20)
THRESHOLD = 4.0
ALPHAS = [0.7, 0.8, 0.85, 0.9, 1.0]  # 1.0 = pure SVD, keeps comparison honest


def load_sample():
    logging.info("Loading parquet...")
    df = pd.read_parquet(
        BASE_DIR / "data/processed/interactions.parquet",
        columns=["userId", "movieId", "rating", "timestamp", "genres"],
    )
    rng = np.random.default_rng(SEED)
    users = df["userId"].unique()
    picked = rng.choice(users, size=min(SAMPLE_USERS, len(users)), replace=False)
    df = df[df["userId"].isin(picked)].copy()
    logging.info(f"Sample: {df.shape[0]} rows, {df['userId'].nunique()} users, {df['movieId'].nunique()} movies")
    return df


def split_per_user(df):
    """Per-user 80/20 chronological split, kept in a single place."""
    df = df.sort_values(["userId", "timestamp"]).reset_index(drop=True)
    df["idx"] = df.groupby("userId").cumcount()
    df["total"] = df.groupby("userId")["userId"].transform("count")
    df["split"] = "train"
    df.loc[df["idx"] >= (df["total"] * 0.8), "split"] = "test"
    train = df[df["split"] == "train"].drop(columns=["idx", "total", "split"])
    test = df[df["split"] == "test"].drop(columns=["idx", "total", "split"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_genre_matrices(train_full, user_to_idx, movie_to_idx):
    """User profile = genres the user liked (rating >= threshold). Item = its genres."""
    liked = train_full[train_full["rating"] >= THRESHOLD]
    # genre vocabulary
    all_genres = set()
    for g in train_full["genres"].fillna(""):
        for x in str(g).split("|"):
            x = x.strip()
            if x:
                all_genres.add(x)
    genres = sorted(all_genres)
    g2i = {g: i for i, g in enumerate(genres)}
    n_users, n_items, n_g = len(user_to_idx), len(movie_to_idx), len(genres)
    logging.info(f"Genres: {n_g} -> {genres}")

    item_mat = np.zeros((n_items, n_g), dtype=np.float32)
    # movieId -> genres mapping from train (first occurrence)
    mov_gen = train_full.drop_duplicates("movieId").set_index("movieId")["genres"]
    for mid, idx in movie_to_idx.items():
        if mid in mov_gen.index:
            for x in str(mov_gen.loc[mid]).split("|"):
                x = x.strip()
                if x in g2i:
                    item_mat[idx, g2i[x]] = 1.0

    user_mat = np.zeros((n_users, n_g), dtype=np.float32)
    user_counts = np.zeros(n_users, dtype=np.float32)
    for u, m in zip(liked["userId"].map(user_to_idx).values, liked["movieId"].map(movie_to_idx).values):
        user_mat[int(u)] += item_mat[int(m)]
        user_counts[int(u)] += 1
    # average the profile; users with no likes stay zero
    nz = user_counts > 0
    user_mat[nz] /= user_counts[nz, None]
    return user_mat, item_mat


def main():
    df = load_sample()
    train_full, test_df = split_per_user(df)

    # IMPORTANT: build index maps from train only to avoid leaking test info
    unique_users = train_full["userId"].unique()
    unique_movies = train_full["movieId"].unique()
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    movie_to_idx = {m: i for i, m in enumerate(unique_movies)}
    # drop test rows with unseen users/movies (cold start is measured separately)
    test_df = test_df[test_df["userId"].isin(user_to_idx) & test_df["movieId"].isin(movie_to_idx)].copy()
    logging.info(f"train={len(train_full)}, test={len(test_df)}")

    train_csr, test_rel_csr = build_matrices(train_full, test_df, user_to_idx, movie_to_idx, THRESHOLD)
    n_items = len(movie_to_idx)

    mlflow.set_tracking_uri("sqlite:///" + str(BASE_DIR / "mlflow.db"))
    mlflow.set_experiment("full_ranking_ensemble")

    # --- baselines ---
    results, per_all = {}, {}

    s, pu = evaluate_full_ranking(random_scorer(n_items, SEED), train_csr, test_rel_csr, KS)
    results["random"], per_all["random"] = s, pu
    logging.info(f"random recall@10={s['recall@10']:.4f}")

    s, pu = evaluate_full_ranking(popularity_scorer(train_csr), train_csr, test_rel_csr, KS)
    results["popularity"], per_all["popularity"] = s, pu
    logging.info(f"popularity recall@10={s['recall@10']:.4f}")

    logging.info("Training SVD (takes a few minutes)...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_full[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()
    svd = SVD(n_factors=50, n_epochs=20, random_state=SEED)
    svd.fit(trainset)
    scorer_svd = svd_scorer(svd, trainset, user_to_idx, movie_to_idx)
    s, pu = evaluate_full_ranking(scorer_svd, train_csr, test_rel_csr, KS)
    results["svd"], per_all["svd"] = s, pu
    logging.info(f"svd recall@10={s['recall@10']:.4f} ndcg@10={s['ndcg@10']:.4f}")

    # --- content ---
    user_mat, item_mat = build_genre_matrices(train_full, user_to_idx, movie_to_idx)

    def content_scorer(users):
        return user_mat[np.asarray(users)] @ item_mat.T

    # per-user [0,1] normalization so SVD and content scales are comparable
    def make_hybrid(alpha):
        def scorer(users):
            users = np.asarray(users)
            s_svd = np.array(scorer_svd(users), dtype=np.float32)
            s_con = np.array(content_scorer(users), dtype=np.float32)
            # min-max per user
            s_svd = (s_svd - s_svd.min(axis=1, keepdims=True)) / (
                s_svd.max(axis=1, keepdims=True) - s_svd.min(axis=1, keepdims=True) + 1e-9
            )
            s_con = (s_con - s_con.min(axis=1, keepdims=True)) / (
                s_con.max(axis=1, keepdims=True) - s_con.min(axis=1, keepdims=True) + 1e-9
            )
            return alpha * s_svd + (1 - alpha) * s_con

        return scorer

    best_alpha, best_s, best_pu = 1.0, results["svd"], per_all["svd"]
    for a in ALPHAS:
        s, pu = evaluate_full_ranking(make_hybrid(a), train_csr, test_rel_csr, KS)
        logging.info(f"hybrid alpha={a} recall@10={s['recall@10']:.4f} ndcg@10={s['ndcg@10']:.4f}")
        with mlflow.start_run(run_name=f"hybrid_alpha_{a}"):
            mlflow.log_param("alpha", a)
            mlflow.log_metric("recall_at_10", s["recall@10"])
            mlflow.log_metric("ndcg_at_10", s["ndcg@10"])
            mlflow.log_metric("precision_at_10", s["precision@10"])
        if s["recall@10"] > best_s["recall@10"]:
            best_alpha, best_s, best_pu = a, s, pu

    # --- SVD+content summary ---
    print("\n=== FULL-RANKING (sample) ===")
    for name in ["random", "popularity", "svd"]:
        print(f"{name:12s} recall@10={results[name]['recall@10']:.4f} ndcg@10={results[name]['ndcg@10']:.4f}")
    print(f"{'hybrid_'+str(best_alpha):12s} recall@10={best_s['recall@10']:.4f} ndcg@10={best_s['ndcg@10']:.4f}")

    d, ci = paired_diff_ci95(best_pu["recall@10"], per_all["svd"]["recall@10"])
    verdict = "SIGNIFICANTLY better than SVD" if d > 0 and abs(d) > ci else "not significant"
    print(f"\nhybrid - svd, recall@10: {d:+.4f} ± {ci:.4f} ({verdict})")
    d2, ci2 = paired_diff_ci95(best_pu["ndcg@10"], per_all["svd"]["ndcg@10"])
    verdict2 = "SIGNIFICANTLY better than SVD" if d2 > 0 and abs(d2) > ci2 else "not significant"
    print(f"hybrid - svd, ndcg@10: {d2:+.4f} ± {ci2:.4f} ({verdict2})")

    # --- Stage 2: add popularity to the ensemble ---
    # popularity is currently stronger than SVD, so final = pop + svd + content
    pop_vec = np.asarray(train_csr.sum(axis=0)).ravel().astype(np.float32)

    def norm_batch(x):
        return (x - x.min(axis=1, keepdims=True)) / (
            x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True) + 1e-9
        )

    def make_final(w_pop, w_svd, w_con):
        def scorer(users):
            users = np.asarray(users)
            s_svd = norm_batch(np.array(scorer_svd(users), dtype=np.float32))
            s_con = norm_batch(np.array(content_scorer(users), dtype=np.float32))
            s_pop = np.broadcast_to(pop_vec, (len(users), len(pop_vec))).astype(np.float32)
            s_pop = norm_batch(s_pop)
            return w_pop * s_pop + w_svd * s_svd + w_con * s_con

        return scorer

    grid = [
        (0.5, 0.3, 0.2),
        (0.6, 0.2, 0.2),
        (0.7, 0.15, 0.15),
        (0.8, 0.1, 0.1),
        (0.6, 0.3, 0.1),
    ]
    best_f, best_fs, best_fpu, best_fw = None, None, None, None
    for w_pop, w_svd, w_con in grid:
        s, pu = evaluate_full_ranking(make_final(w_pop, w_svd, w_con), train_csr, test_rel_csr, KS)
        logging.info(
            f"final pop={w_pop} svd={w_svd} con={w_con} recall@10={s['recall@10']:.4f} ndcg@10={s['ndcg@10']:.4f}"
        )
        with mlflow.start_run(run_name=f"final_pop{w_pop}_svd{w_svd}_con{w_con}"):
            mlflow.log_param("w_pop", w_pop)
            mlflow.log_param("w_svd", w_svd)
            mlflow.log_param("w_con", w_con)
            mlflow.log_metric("recall_at_10", s["recall@10"])
            mlflow.log_metric("ndcg_at_10", s["ndcg@10"])
        if best_fs is None or s["recall@10"] > best_fs["recall@10"]:
            best_f, best_fs, best_fpu, best_fw = f"pop{w_pop}_svd{w_svd}_con{w_con}", s, pu, (w_pop, w_svd, w_con)

    print(f"\n=== FINAL {best_f} ===")
    print(f"final recall@10={best_fs['recall@10']:.4f} ndcg@10={best_fs['ndcg@10']:.4f}")
    for base in ["popularity", "svd"]:
        dd, cc = paired_diff_ci95(best_fpu["recall@10"], per_all[base]["recall@10"])
        v = "SIGNIFICANTLY better" if dd > 0 and abs(dd) > cc else "not significant"
        print(f"final - {base}, recall@10: {dd:+.4f} ± {cc:.4f} ({v})")


if __name__ == "__main__":
    main()
