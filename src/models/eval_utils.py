"""
Honest full-ranking evaluation of recommender models.

Idea: for each user the model scores the WHOLE catalog,
items seen in train are excluded, top-K is compared
against relevant items from test (rating >= threshold).

Any model plugs in via a "scorer" function:
    scorer(user_indices: np.ndarray[int]) -> np.ndarray[float32] of shape (len(users), n_items)
"""
import numpy as np
import pandas as pd
from scipy import sparse


# --------------------------------------------------------------------------
# Matrix preparation
# --------------------------------------------------------------------------
def _to_binary_csr(df, user_to_idx, movie_to_idx):
    rows = df['userId'].map(user_to_idx)
    cols = df['movieId'].map(movie_to_idx)
    if rows.isna().any() or cols.isna().any():
        raise ValueError('Found userId/movieId missing from user_to_idx/movie_to_idx')
    shape = (len(user_to_idx), len(movie_to_idx))
    m = sparse.csr_matrix(
        (np.ones(len(df), dtype=np.float32), (rows.values.astype(int), cols.values.astype(int))),
        shape=shape,
    )
    m.data[:] = 1.0  # duplicates are summed during construction, reset to binary
    return m


def build_matrices(train_df, test_df, user_to_idx, movie_to_idx, rating_threshold=4.0):
    """
    train_csr    - what the user ALREADY saw in train (any rating) -> excluded from output
    test_rel_csr - what the user rated in test with >= threshold -> this is what we must guess
    """
    train_csr = _to_binary_csr(train_df, user_to_idx, movie_to_idx)
    test_rel = test_df[test_df['rating'] >= rating_threshold]
    test_rel_csr = _to_binary_csr(test_rel, user_to_idx, movie_to_idx)
    return train_csr, test_rel_csr


# --------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------
def evaluate_full_ranking(scorer, train_csr, test_rel_csr, ks=(10, 20), batch_size=512):
    """
    Returns (summary, per_user):
      summary  - dict {'recall@10': mean, ...}
      per_user - dict {'recall@10': np.array per user, ..., 'user_idx': np.array}
    per_user is needed for confidence intervals and paired model comparison.
    """
    n_users, n_items = train_csr.shape
    max_k = max(ks)
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
    ideal_cum = np.cumsum(discounts)  # ideal_cum[j-1] = IDCG for j relevant items

    names = [f'{m}@{k}' for k in ks for m in ('precision', 'recall', 'ndcg', 'hitrate')]
    chunks = {n: [] for n in names}
    user_chunks = []

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        rel_block = test_rel_csr[start:end]
        n_rel = np.asarray(rel_block.sum(axis=1)).ravel()
        valid = n_rel > 0                      # skip users with no relevant items in test
        if not valid.any():
            continue

        users = np.arange(start, end)[valid]
        n_rel = n_rel[valid]
        rel = rel_block[valid].toarray().astype(bool)

        scores = np.array(scorer(users), dtype=np.float32, copy=True)
        assert scores.shape == (len(users), n_items), f'scorer returned {scores.shape}'

        seen = train_csr[users].tocoo()
        scores[seen.row, seen.col] = -np.inf   # never recommend already seen items

        # top-K: argpartition is faster than full sort, then sort only top-K
        part = np.argpartition(-scores, max_k - 1, axis=1)[:, :max_k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        top = np.take_along_axis(part, order, axis=1)

        hits = np.take_along_axis(rel, top, axis=1).astype(np.float32)

        for k in ks:
            h = hits[:, :k]
            n_hit = h.sum(axis=1)
            dcg = (h * discounts[:k]).sum(axis=1)
            idcg = ideal_cum[np.minimum(n_rel, k).astype(int) - 1]
            chunks[f'precision@{k}'].append(n_hit / k)
            chunks[f'recall@{k}'].append(n_hit / n_rel)
            chunks[f'ndcg@{k}'].append(dcg / idcg)
            chunks[f'hitrate@{k}'].append((n_hit > 0).astype(np.float32))
        user_chunks.append(users)

    per_user = {n: np.concatenate(v) for n, v in chunks.items()}
    per_user['user_idx'] = np.concatenate(user_chunks)
    summary = {n: float(per_user[n].mean()) for n in names}
    summary['n_users_evaluated'] = int(len(per_user['user_idx']))
    return summary, per_user


# --------------------------------------------------------------------------
# Statistics: confidence intervals
# --------------------------------------------------------------------------
def mean_ci95(values):
    """Mean and 95% interval half-width (normal approximation, large n)."""
    values = np.asarray(values, dtype=np.float64)
    return values.mean(), 1.96 * values.std(ddof=1) / np.sqrt(len(values))


def paired_diff_ci95(values_a, values_b):
    """
    Difference A - B over the same users + 95% interval.
    If the interval excludes 0, the difference is statistically significant.
    """
    diff = np.asarray(values_a, dtype=np.float64) - np.asarray(values_b, dtype=np.float64)
    return mean_ci95(diff)


# --------------------------------------------------------------------------
# Baseline scorers
# --------------------------------------------------------------------------
def random_scorer(n_items, seed=42):
    rng = np.random.default_rng(seed)

    def scorer(users):
        return rng.random((len(users), n_items), dtype=np.float32)

    return scorer


def popularity_scorer(train_csr):
    """Same list for everyone: most frequently rated movies in train."""
    pop = np.asarray(train_csr.sum(axis=0)).ravel().astype(np.float32)

    def scorer(users):
        return np.broadcast_to(pop, (len(users), len(pop)))

    return scorer


def svd_scorer(svd, trainset, user_to_idx, movie_to_idx):
    """
    Converts a trained surprise.SVD into a matrix scorer.
    Users/items unseen in trainset get zero vectors and biases
    (same behavior as SVD.predict: unknown item -> mu + b_u).
    """
    n_users, n_items = len(user_to_idx), len(movie_to_idx)
    f = svd.pu.shape[1]
    P = np.zeros((n_users, f), dtype=np.float32)
    Q = np.zeros((n_items, f), dtype=np.float32)
    bu = np.zeros(n_users, dtype=np.float32)
    bi = np.zeros(n_items, dtype=np.float32)

    for raw_u, idx in user_to_idx.items():
        try:
            inner = trainset.to_inner_uid(raw_u)
        except ValueError:
            continue
        P[idx], bu[idx] = svd.pu[inner], svd.bu[inner]

    for raw_i, idx in movie_to_idx.items():
        try:
            inner = trainset.to_inner_iid(raw_i)
        except ValueError:
            continue
        Q[idx], bi[idx] = svd.qi[inner], svd.bi[inner]

    mu = np.float32(trainset.global_mean)

    def scorer(users):
        return mu + bu[users, None] + bi[None, :] + P[users] @ Q.T

    return scorer


# --------------------------------------------------------------------------
# Legacy protocol (diagnostics only!)
# --------------------------------------------------------------------------
def legacy_protocol_eval(test_df, scores, k=10, threshold=4.0):
    """
    Reproduces the old scheme: rank ONLY movies rated in test.
    Shows that random/popularity look almost as good as models under this scheme.
    scores - array of scores with the same length as test_df.
    """
    d = pd.DataFrame({
        'userId': test_df['userId'].values,
        'rel': (test_df['rating'].values >= threshold),
        'score': np.asarray(scores),
    })
    d = d.sort_values(['userId', 'score'], ascending=[True, False])
    d['rank'] = d.groupby('userId').cumcount()
    n_rel = d.groupby('userId')['rel'].sum()
    hits = d[d['rank'] < k].groupby('userId')['rel'].sum().reindex(n_rel.index, fill_value=0)
    mask = n_rel > 0
    return float((hits[mask] / k).mean()), float((hits[mask] / n_rel[mask]).mean())
