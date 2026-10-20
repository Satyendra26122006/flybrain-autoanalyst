"""Generate synthetic datasets with known structure and label which tools find real signal."""
import numpy as np
import pandas as pd
from .profiling import profile, feature_vector
from .insights import TASKS, TOOLS


def make_dataset(rng):
    n = int(rng.integers(60, 600)); k = int(rng.integers(2, 7))
    X = rng.normal(size=(n, k))
    if rng.random() < .5:  # inject correlation
        X[:, 1] = X[:, 0] * rng.uniform(.5, 1.5) + rng.normal(scale=.7, size=n)
    df = pd.DataFrame(X, columns=[f"num{i}" for i in range(k)])
    if rng.random() < .5:
        df["grp"] = rng.choice(list("ABC"), n)
        if rng.random() < .5:
            df["num0"] += df["grp"].map({"A": 0, "B": 1.0, "C": 2.0}).values
    if rng.random() < .5:
        df["date"] = pd.date_range("2022-01-01", periods=n)
        if rng.random() < .5:
            df["num0"] += np.linspace(0, rng.uniform(1, 4), n)
    if rng.random() < .3:
        i = rng.choice(n, 3, replace=False); df.loc[i, "num0"] += 40
    if rng.random() < .3:
        df.loc[rng.random(n) < rng.uniform(.05, .4), "num1"] = np.nan
    return df


def build_training_set(n_datasets=300, seed=0):
    """X: feature vectors; Y[i,j]=1 if tool j produced a valid finding on dataset i."""
    rng = np.random.default_rng(seed); X, Y = [], []
    for _ in range(n_datasets):
        df = make_dataset(rng)
        x, y = label_dataset(df)
        X.append(x); Y.append(y)
    return np.stack(X), np.array(Y, dtype=np.float32)


def label_dataset(df):
    """Return the router input and deterministic synthetic-task labels for one dataset."""
    p = profile(df)
    return (
        feature_vector(p),
        np.asarray([float(len(TOOLS[t](df, p)) > 0) for t in TASKS], dtype=np.float32),
    )
