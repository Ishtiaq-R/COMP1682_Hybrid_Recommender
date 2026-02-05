import numpy as np
import pandas as pd

def train_test_split_per_user(ratings: pd.DataFrame, test_size: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for uid, grp in ratings.groupby("userId"):
        grp = grp.sort_values("timestamp")

        if len(grp) <= test_size:
            train_parts.append(grp)
            continue

        tail = grp.tail(max(test_size * 3, test_size))
        test_idx = rng.choice(tail.index.to_numpy(), size=test_size, replace=False)

        test_parts.append(grp.loc[test_idx])
        train_parts.append(grp.drop(index=test_idx))

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()
    return train_df, test_df
