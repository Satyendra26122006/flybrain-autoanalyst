"""Data-quality checks. Flags issues; never deletes data."""
import pandas as pd


def check(df: pd.DataFrame) -> list[dict]:
    issues = []
    dup = int(df.duplicated().sum())
    if dup:
        issues.append({"issue": "duplicate_rows", "detail": f"{dup} duplicate rows", "severity": "medium"})
    for c, r in df.isna().mean().items():
        if r > 0.2:
            issues.append({"issue": "high_missing", "detail": f"'{c}' is {r:.0%} missing", "severity": "high" if r > 0.5 else "medium"})
    for c in df.columns:
        if df[c].nunique(dropna=True) <= 1:
            issues.append({"issue": "constant_column", "detail": f"'{c}' has a single value", "severity": "low"})
    for c in df.select_dtypes("number").columns:
        s = df[c].dropna()
        if len(s) < 8:
            continue
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        n = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum())
        if iqr > 0 and n:
            issues.append({"issue": "extreme_outliers", "detail": f"'{c}': {n} extreme values (3xIQR)", "severity": "low"})
    for c in df.select_dtypes(include=["object", "string"]).columns:
        s = df[c].dropna().astype(str)
        if s.nunique() > 1 and s.str.strip().str.lower().nunique() < s.nunique():
            issues.append({"issue": "inconsistent_categories", "detail": f"'{c}' has values differing only by case/spacing", "severity": "medium"})
    return issues
