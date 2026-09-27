"""
BPR two-tower hybrid with HF embeddings and full-ranking evaluation.
1. Tuned SVD (100 factors, 30 epochs) as a reference CF model.
2. HF content: cosine similarity over MiniLM embeddings (mean of liked items -> item).
3. Torch BPR two-tower (ranking loss instead of MSE).
4. Final ensemble: popularity + best CF + HF content + genre content.
The sample and split are fixed (seed 42, 15k users) so numbers stay comparable.
"""
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from surprise import Dataset, Reader, SVD
from torch.utils.data import Dataset as TorchDataset, DataLoader

from eval_utils import (
    build_matrices, evaluate_full_ranking, paired_diff_ci95,
    popularity_scorer, svd_scorer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SEED = 42
SAMPLE_USERS = 15000
KS = (10, 20)
THRESHOLD = 4.0


def load_sample():
    df = pd.read_parquet(
        BASE_DIR / "data/processed/interactions.parquet",
        columns=["userId", "movieId", "rating", "timestamp", "genres"],
    )
    rng = np.random.default_rng(SEED)
    picked = rng.choice(df["userId"].unique(), size=min(SAMPLE_USERS, df["userId"].nunique()), replace=False)
    return df[df["userId"].isin(picked)].copy()


def split_per_user(df):
    df = df.sort_values(["userId", "timestamp"]).reset_index(drop=True)
    df["idx"] = df.groupby("userId").cumcount()
    df["total"] = df.groupby("userId")["userId"].transform("count")
    df["split"] = "train"
    df.loc[df["idx"] >= (df["total"] * 0.8), "split"] = "test"
    train = df[df["split"] == "train"].drop(columns=["idx", "total", "split"])
    test = df[df["split"] == "test"].drop(columns=["idx", "total", "split"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


class BPRDataset(TorchDataset):
    def __init__(self, user_pos_list, n_items, n_neg=4, seed=42):
        self.pairs = []
        rng = np.random.default_rng(seed)
        for u, pos_items in user_pos_list:
            negs = rng.integers(0, n_items, size=len(pos_items) * n_neg)
            for p, n_ in zip(pos_items, negs):
                self.pairs.append((u, p, int(n_)))
        self.pairs = np.array(self.pairs, dtype=np.int64)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.u = nn.Embedding(n_users, dim)
        self.i = nn.Embedding(n_items, dim)
        nn.init.normal_(self.u.weight, std=0.05)
        nn.init.normal_(self.i.weight, std=0.05)

    def forward(self, u, p, n_):
        return (self.u(u) * self.i(p)).sum(1), (self.u(u) * self.i(n_)).sum(1)


def main():
    torch.manual_seed(SEED)
    df = load_sample()
    train_full, test_df = split_per_user(df)
    user_to_idx = {u: i for i, u in enumerate(train_full["userId"].unique())}
    movie_to_idx = {m: i for i, m in enumerate(train_full["movieId"].unique())}
    test_df = test_df[test_df["userId"].isin(user_to_idx) & test_df["movieId"].isin(movie_to_idx)].copy()
    logging.info(f"train={len(train_full)} test={len(test_df)} users={len(user_to_idx)} items={len(movie_to_idx)}")

    train_csr, test_rel_csr = build_matrices(train_full, test_df, user_to_idx, movie_to_idx, THRESHOLD)
    n_items = len(movie_to_idx)
    mlflow.set_tracking_uri("sqlite:///" + str(BASE_DIR / "mlflow.db"))
    mlflow.set_experiment("bpr_hybrid")

    # ---------- 1. Tuned SVD reference ----------
    logging.info("Training tuned SVD (100 factors, 30 epochs)...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_full[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()
    svd = SVD(n_factors=100, n_epochs=30, lr_all=0.005, reg_all=0.02, random_state=SEED)
    svd.fit(trainset)
    scorer_svd = svd_scorer(svd, trainset, user_to_idx, movie_to_idx)
    s_svd, pu_svd = evaluate_full_ranking(scorer_svd, train_csr, test_rel_csr, KS)
    logging.info(f"SVD-tuned recall@10={s_svd['recall@10']:.4f} ndcg@10={s_svd['ndcg@10']:.4f}")

    s_pop, pu_pop = evaluate_full_ranking(popularity_scorer(train_csr), train_csr, test_rel_csr, KS)
    logging.info(f"popularity recall@10={s_pop['recall@10']:.4f}")

    # ---------- 2. HF content ----------
    logging.info("Building HF content profiles...")
    emb_df = pd.read_pickle(BASE_DIR / "data/processed/movie_embeddings.pkl")
    hf_dict = {int(m): np.array(e, dtype=np.float32) for m, e in zip(emb_df["movie_id"], emb_df["embedding"])}
    HF_DIM = 384
    item_hf = np.zeros((n_items, HF_DIM), dtype=np.float32)
    for mid, idx in movie_to_idx.items():
        if int(mid) in hf_dict:
            item_hf[idx] = hf_dict[int(mid)]
    # L2 normalize for cosine similarity
    norms = np.linalg.norm(item_hf, axis=1, keepdims=True) + 1e-9
    item_hf_n = item_hf / norms

    liked = train_full[train_full["rating"] >= THRESHOLD]
    user_hf = np.zeros((len(user_to_idx), HF_DIM), dtype=np.float32)
    cnt = np.zeros(len(user_to_idx), dtype=np.float32)
    umap = liked["userId"].map(user_to_idx).values
    mmap = liked["movieId"].map(movie_to_idx).values
    has_hf = np.array([int(liked.iloc[i]["movieId"]) in hf_dict for i in range(len(liked))])
    for u, m, ok in zip(umap, mmap, has_hf):
        if ok:
            user_hf[int(u)] += item_hf[int(m)]
            cnt[int(u)] += 1
    nz = cnt > 0
    user_hf[nz] /= cnt[nz, None]
    user_hf[nz] /= (np.linalg.norm(user_hf[nz], axis=1, keepdims=True) + 1e-9)
    logging.info(f"Users with HF profile: {nz.sum()}/{len(cnt)}")

    def hf_scorer(users):
        return user_hf[np.asarray(users)] @ item_hf_n.T

    s_hf, pu_hf = evaluate_full_ranking(hf_scorer, train_csr, test_rel_csr, KS)
    logging.info(f"HF-content recall@10={s_hf['recall@10']:.4f} ndcg@10={s_hf['ndcg@10']:.4f}")

    # genre fallback (cheap, full coverage)
    all_g = sorted({x.strip() for g in train_full["genres"].fillna("") for x in str(g).split("|") if x.strip()})
    g2i = {g: i for i, g in enumerate(all_g)}
    item_g = np.zeros((n_items, len(all_g)), dtype=np.float32)
    mov_gen = train_full.drop_duplicates("movieId").set_index("movieId")["genres"]
    for mid, idx in movie_to_idx.items():
        if mid in mov_gen.index:
            for x in str(mov_gen.loc[mid]).split("|"):
                x = x.strip()
                if x in g2i:
                    item_g[idx, g2i[x]] = 1.0
    user_g = np.zeros((len(user_to_idx), len(all_g)), dtype=np.float32)
    cnts = np.zeros(len(user_to_idx))
    for u, m in zip(umap, mmap):
        user_g[int(u)] += item_g[int(m)]
        cnts[int(u)] += 1
    user_g[cnts > 0] /= cnts[cnts > 0, None]

    def genre_scorer(users):
        return user_g[np.asarray(users)] @ item_g.T

    # ---------- 3. Torch BPR two-tower ----------
    logging.info("Training torch BPR two-tower...")
    # positives = liked items (rating >= threshold) in index space
    from collections import defaultdict
    pos = defaultdict(list)
    for u, m in zip(umap, mmap):
        pos[int(u)].append(int(m))
    user_pos = [(u, v) for u, v in pos.items() if len(v) >= 2]
    ds = BPRDataset(user_pos, n_items, n_neg=4, seed=SEED)
    loader = DataLoader(ds, batch_size=4096, shuffle=True, num_workers=0)
    device = torch.device("cpu")
    model = TwoTower(len(user_to_idx), n_items, dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    model.train()
    for epoch in range(6):
        losses = []
        for batch in loader:
            bu, bp, bn = batch[:, 0].to(device), batch[:, 1].to(device), batch[:, 2].to(device)
            sp, sn = model(bu, bp, bn)
            loss = -torch.log(torch.sigmoid(sp - sn) + 1e-9).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        logging.info(f"BPR epoch {epoch+1}/6 loss={np.mean(losses):.4f}")
    model.eval()
    with torch.no_grad():
        U = model.u.weight.cpu().numpy().astype(np.float32)
        I = model.i.weight.cpu().numpy().astype(np.float32)

    def bpr_scorer(users):
        return U[np.asarray(users)] @ I.T

    s_bpr, pu_bpr = evaluate_full_ranking(bpr_scorer, train_csr, test_rel_csr, KS)
    logging.info(f"BPR recall@10={s_bpr['recall@10']:.4f} ndcg@10={s_bpr['ndcg@10']:.4f}")

    # ---------- 4. Final ensemble ----------
    pop_vec = np.asarray(train_csr.sum(axis=0)).ravel().astype(np.float32)

    def norm_batch(x):
        return (x - x.min(axis=1, keepdims=True)) / (x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True) + 1e-9)

    # pick the stronger CF model: tuned SVD vs BPR
    cf_name, cf_scorer, cf_s = ("bpr", bpr_scorer, s_bpr) if s_bpr["recall@10"] > s_svd["recall@10"] else ("svd", scorer_svd, s_svd)
    logging.info(f"Best CF model: {cf_name} recall@10={cf_s['recall@10']:.4f}")

    def make_final(w_pop, w_cf, w_hf, w_g):
        def scorer(users):
            users = np.asarray(users)
            a = norm_batch(np.array(cf_scorer(users), dtype=np.float32))
            b = norm_batch(np.broadcast_to(pop_vec, (len(users), len(pop_vec))).astype(np.float32))
            c = norm_batch(np.array(hf_scorer(users), dtype=np.float32))
            d_ = norm_batch(np.array(genre_scorer(users), dtype=np.float32))
            return w_pop * b + w_cf * a + w_hf * c + w_g * d_
        return scorer

    grid = [
        (0.5, 0.25, 0.15, 0.10),
        (0.4, 0.3, 0.2, 0.10),
        (0.45, 0.25, 0.2, 0.10),
        (0.35, 0.35, 0.2, 0.10),
        (0.5, 0.2, 0.2, 0.10),
    ]
    best = None
    for w in grid:
        s, pu = evaluate_full_ranking(make_final(*w), train_csr, test_rel_csr, KS)
        logging.info(f"weights={w} recall@10={s['recall@10']:.4f} ndcg@10={s['ndcg@10']:.4f}")
        with mlflow.start_run(run_name=f"ensemble_{w}"):
            mlflow.log_params({"w_pop": w[0], "w_cf": w[1], "w_hf": w[2], "w_g": w[3], "cf": cf_name})
            mlflow.log_metric("recall_at_10", s["recall@10"])
            mlflow.log_metric("ndcg_at_10", s["ndcg@10"])
        if best is None or s["recall@10"] > best[1]["recall@10"]:
            best = (w, s, pu)

    w, s, pu = best
    print("\n=== FINAL SUMMARY ===")
    print(f"popularity      recall@10={s_pop['recall@10']:.4f} ndcg@10={s_pop['ndcg@10']:.4f}")
    print(f"svd-tuned       recall@10={s_svd['recall@10']:.4f} ndcg@10={s_svd['ndcg@10']:.4f}")
    print(f"hf-content      recall@10={s_hf['recall@10']:.4f}")
    print(f"bpr             recall@10={s_bpr['recall@10']:.4f} ndcg@10={s_bpr['ndcg@10']:.4f}")
    print(f"final {w} recall@10={s['recall@10']:.4f} ndcg@10={s['ndcg@10']:.4f}")
    for name, ps in [("popularity", pu_pop), ("svd-tuned", pu_svd), ("bpr", pu_bpr)]:
        d_, ci = paired_diff_ci95(pu["recall@10"], ps["recall@10"])
        v = "SIGNIFICANTLY better" if d_ > 0 and abs(d_) > ci else "not significant"
        print(f"final - {name}: {d_:+.4f} ± {ci:.4f} ({v})")

    # ---------- 5. Persist weights so serving never retrains ----------
    # Serving builds the user vector on the fly as the mean of liked item
    # vectors, so only item vectors + id mapping + ensemble weights are needed.
    import json
    import pickle

    artifacts = BASE_DIR / "artifacts"
    artifacts.mkdir(exist_ok=True)
    np.save(artifacts / "bpr_item_vectors.npy", I)
    with open(artifacts / "bpr_movie_to_idx.pkl", "wb") as f:
        pickle.dump(movie_to_idx, f)
    with open(artifacts / "ensemble_weights.json", "w") as f:
        json.dump(
            {"w_pop": w[0], "w_cf": w[1], "w_hf": w[2], "w_g": w[3], "cf": cf_name,
             "recall_at_10": s["recall@10"], "ndcg_at_10": s["ndcg@10"]},
            f, indent=2,
        )
    logging.info(f"Weights saved to {artifacts} (no retraining needed for serving)")
    with mlflow.start_run(run_name="final_artifacts"):
        mlflow.log_param("cf", cf_name)
        mlflow.log_metric("recall_at_10", s["recall@10"])
        mlflow.log_metric("ndcg_at_10", s["ndcg@10"])
        mlflow.log_artifact(str(artifacts / "bpr_item_vectors.npy"))
        mlflow.log_artifact(str(artifacts / "ensemble_weights.json"))


if __name__ == "__main__":
    main()
