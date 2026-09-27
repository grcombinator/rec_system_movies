"""
Step 1: honest evaluation of baselines (random, popularity, SVD) with full-ranking.

Run (from the project root, MLflow must be up):
    python src/models/evaluate_baselines.py
"""
import logging
import sys
from pathlib import Path

# Make sibling imports (utils, eval_utils) work both as
# `python src/models/evaluate_baselines.py` and `python -m src.models...`.
# (The IDE "import could not be resolved" hint is the same root cause.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD

from utils import get_data_splits
from eval_utils import (
    build_matrices, evaluate_full_ranking, mean_ci95, paired_diff_ci95,
    random_scorer, popularity_scorer, svd_scorer, legacy_protocol_eval,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_SAVE_DIR = BASE_DIR / 'src' / 'models'

KS = (10, 20)
RATING_THRESHOLD = 4.0
SEED = 42
N_FACTORS, N_EPOCHS = 50, 20


def log_run(name, params, summary, per_user):
    """Log mean metrics and the 95% interval half-width to MLflow."""
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        for metric, value in summary.items():
            key = metric.replace('@', '_at_')
            mlflow.log_metric(key, value)
            if metric in per_user:
                mlflow.log_metric(f'{key}_ci95', mean_ci95(per_user[metric])[1])


def main():
    logging.info('Loading data (SAMPLE_USERS env, default 15000; set 0 for full data)')
    train_full, test_df = get_data_splits(BASE_DIR)
    # Build index maps from train only: no test leakage, always consistent with the sample.
    user_to_idx = {u: i for i, u in enumerate(train_full["userId"].unique())}
    movie_to_idx = {m: i for i, m in enumerate(train_full["movieId"].unique())}
    test_df = test_df[test_df["userId"].isin(user_to_idx) & test_df["movieId"].isin(movie_to_idx)].copy()
    n_items = len(movie_to_idx)

    train_csr, test_rel_csr = build_matrices(
        train_full, test_df, user_to_idx, movie_to_idx, RATING_THRESHOLD)

    density = train_csr.nnz / (train_csr.shape[0] * train_csr.shape[1])
    logging.info(
        f'users={train_csr.shape[0]}, items={n_items}, train_interactions={train_csr.nnz}, '
        f'test_relevant={test_rel_csr.nnz}, density={density:.4%}')

    mlflow.set_tracking_uri('sqlite:///' + str(BASE_DIR / 'mlflow.db'))
    mlflow.set_experiment('recsys_eval_full_ranking')

    results, per_user_all = {}, {}

    # ---------- Random ----------
    logging.info('Evaluating: random')
    s, pu = evaluate_full_ranking(random_scorer(n_items, SEED), train_csr, test_rel_csr, KS)
    results['random'], per_user_all['random'] = s, pu
    log_run('random', {'model': 'random', 'seed': SEED, 'protocol': 'full_ranking'}, s, pu)

    # ---------- Popularity ----------
    logging.info('Evaluating: popularity')
    s, pu = evaluate_full_ranking(popularity_scorer(train_csr), train_csr, test_rel_csr, KS)
    results['popularity'], per_user_all['popularity'] = s, pu
    log_run('popularity', {'model': 'popularity', 'protocol': 'full_ranking'}, s, pu)

    # ---------- SVD (trained on the SAME train_full) ----------
    logging.info('Training SVD (may take a few minutes)')
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_full[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD(n_factors=N_FACTORS, n_epochs=N_EPOCHS, random_state=SEED)
    svd.fit(trainset)

    logging.info('Evaluating: svd')
    scorer = svd_scorer(svd, trainset, user_to_idx, movie_to_idx)
    s, pu = evaluate_full_ranking(scorer, train_csr, test_rel_csr, KS)
    results['svd'], per_user_all['svd'] = s, pu
    log_run('svd', {'model': 'svd', 'n_factors': N_FACTORS, 'n_epochs': N_EPOCHS,
                    'seed': SEED, 'protocol': 'full_ranking'}, s, pu)

    # ---------- Summary table ----------
    rows = []
    for model, s in results.items():
        row = {'model': model}
        for k in KS:
            for m in ('recall', 'ndcg', 'hitrate'):
                name = f'{m}@{k}'
                mean, ci = mean_ci95(per_user_all[model][name])
                row[name] = f'{mean:.4f} ± {ci:.4f}'
        rows.append(row)
    table = pd.DataFrame(rows).set_index('model')
    print('\n=== FULL-RANKING (95% CI) ===')
    print(table.to_string())
    print(f"\nEvaluated users: {results['svd']['n_users_evaluated']}")

    # ---------- Paired comparison: SVD vs popularity ----------
    for metric in ('recall@10', 'ndcg@10'):
        d, ci = paired_diff_ci95(per_user_all['svd'][metric], per_user_all['popularity'][metric])
        verdict = 'significant' if abs(d) > ci else 'NOT significant'
        print(f'SVD - popularity, {metric}: {d:+.4f} ± {ci:.4f} ({verdict})')

    # ---------- Diagnostics: legacy protocol ----------
    print('\n=== LEGACY PROTOCOL (rank only items rated in test) ===')
    rng = np.random.default_rng(SEED)
    item_pop = np.asarray(train_csr.sum(axis=0)).ravel()
    test_movie_idx = test_df['movieId'].map(movie_to_idx).values
    legacy_scores = {
        'random': rng.random(len(test_df)),
        'popularity': item_pop[test_movie_idx],
    }
    for name, sc in legacy_scores.items():
        p, r = legacy_protocol_eval(test_df, sc, k=10, threshold=RATING_THRESHOLD)
        print(f'{name:12s} precision@10={p:.4f} recall@10={r:.4f}')
    print('(reference SVD under this protocol: precision@10=0.5895, recall@10=0.7187)')


if __name__ == '__main__':
    main()
