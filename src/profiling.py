"""Dataset profile + fixed-length feature vector (the brain's 'eyes')."""
import numpy as np
import pandas as pd

TARGET_NAMES = {"target", "label", "y", "churn", "class", "outcome"}
N_FEATURES = 10


def profile(df: pd.DataFrame) -> dict:
    num = df.select_dtypes("number").columns.tolist()
    dt = df.select_dtypes("datetime").columns.tolist()
    cat = [c for c in df.columns if c not in num + dt]
    ids = [c for c in df.columns if df[c].nunique() == len(df) and len(df) > 20
           and not pd.api.types.is_float_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])]
    num = [c for c in num if c not in ids]
    cat_useful = [c for c in cat if 2 <= df[c].nunique() <= 20 and c not in ids]
    skew = [abs(df[c].skew()) for c in num if df[c].notna().sum() > 3]
    return {
        "n_rows": len(df), "n_cols": df.shape[1],
        "numeric_columns": num, "categorical_columns": cat_useful,
        "datetime_columns": dt, "id_columns": ids,
        "missing_rate": float(df.isna().mean().mean()),
        "duplicate_rate": float(df.duplicated().mean()),
        "mean_abs_skew": float(np.nan_to_num(np.mean(skew))) if skew else 0.0,
        "has_datetime": bool(dt),
        "has_target": any(c.lower() in TARGET_NAMES for c in df.columns),
    }


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return describe output with mixed categorical top values safe for Arrow tables."""
    summary = df.describe(include="all").transpose()
    if "top" in summary.columns:
        summary["top"] = summary["top"].map(
            lambda value: str(value) if pd.notna(value) else None
        )
    return summary


def feature_vector(p: dict) -> np.ndarray:
    return np.array([
        np.log1p(p["n_rows"]) / 15, np.log1p(p["n_cols"]) / 6,
        len(p["numeric_columns"]) / 20, len(p["categorical_columns"]) / 10,
        float(p["has_datetime"]), float(p["has_target"]),
        p["missing_rate"], p["duplicate_rate"], min(p["mean_abs_skew"], 10) / 10,
        len(p["id_columns"]) / 3,
    ], dtype=np.float32)
