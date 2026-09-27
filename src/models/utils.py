"""Shared data helpers: per-user chronological train/test splits."""
import os
from pathlib import Path


def get_data_splits(base_dir: Path, sample_users: int | None = None, seed: int = 42):
    """Load parquet with low memory footprint and make a per-user 80/20 time split.

    Only the columns needed for training/eval are loaded (no title/genres strings),
    frames are converted to numpy-backed dtypes to avoid pyarrow take() OOM on sort.
    If sample_users is set (or SAMPLE_USERS env), a fixed random user sample is used
    so baseline evaluation fits into 7-8 GB RAM and stays comparable with ensemble scripts.
    """
    if sample_users is None:
        sample_users = int(os.getenv("SAMPLE_USERS", "15000"))
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(
        base_dir / 'data/processed/interactions.parquet',
        columns=['userId', 'movieId', 'rating', 'timestamp'],
    )
    # numpy-backed dtypes: pyarrow-backed frames OOM in sort_values/take on 24M rows
    df = df.convert_dtypes(dtype_backend="numpy_nullable")
    for col in ("userId", "movieId", "timestamp"):
        df[col] = pd.to_numeric(df[col], downcast="integer")
    df["rating"] = pd.to_numeric(df["rating"], downcast="float")
    if sample_users and df["userId"].nunique() > sample_users:
        rng = np.random.default_rng(seed)
        picked = rng.choice(df["userId"].unique(), size=sample_users, replace=False)
        df = df[df["userId"].isin(picked)].copy()
    df = df.sort_values(['userId', 'timestamp']).reset_index(drop=True)
    df['user_interaction_idx'] = df.groupby('userId').cumcount()
    df['user_total_interactions'] = df.groupby('userId')['userId'].transform('count')
    df['split'] = 'train'
    df.loc[df['user_interaction_idx'] >= (df['user_total_interactions'] * 0.8), 'split'] = 'test'

    train_full = df[df['split'] == 'train'].drop(columns=['user_interaction_idx', 'user_total_interactions', 'split'])
    test_df = df[df['split'] == 'test'].drop(columns=['user_interaction_idx', 'user_total_interactions', 'split'])

    return train_full, test_df
