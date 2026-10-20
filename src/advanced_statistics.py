"""Selectable statistical tools with explicit tests and multiple-test correction."""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .validation import Finding, bh_adjust

ALPHA = 0.05
MAX_NUMERIC_COLUMNS = 30
MAX_CATEGORICAL_COLUMNS = 8

CORRELATION_METHODS = {
    "Pearson": stats.pearsonr,
    "Spearman": stats.spearmanr,
    "Kendall": stats.kendalltau,
}


def correlations(df: pd.DataFrame, columns: list[str], method: str) -> list[Finding]:
    """Run pairwise correlation tests and adjust p-values within this family."""
    if method not in CORRELATION_METHODS:
        raise ValueError(f"Unsupported correlation method: {method}")
    method_fn = CORRELATION_METHODS[method]
    rows = []
    for left, right in itertools.combinations(columns[:MAX_NUMERIC_COLUMNS], 2):
        pair = df[[left, right]].dropna()
        if len(pair) < 10 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
            continue
        result = method_fn(pair[left], pair[right])
        rows.append((left, right, float(result.statistic), float(result.pvalue), len(pair)))

    adjusted = bh_adjust([row[3] for row in rows])
    findings = []
    for (left, right, coefficient, _pvalue, count), adjusted_p in zip(rows, adjusted):
        if adjusted_p < ALPHA and abs(coefficient) >= 0.3:
            direction = "positive" if coefficient > 0 else "negative"
            findings.append(Finding(
                f"{method.lower()}_correlation",
                f"'{left}' and '{right}' have a {direction} {method.lower()} association.",
                {
                    "method": method,
                    "coefficient": round(coefficient, 4),
                    "n": count,
                    "p_adj": float(adjusted_p),
                },
                "Association does not establish causality. The correlation test evaluates monotonic "
                "association for rank methods and linear association for Pearson.",
                ["Potential confounding and repeated observations are not modeled."],
                f"Inspect a scatter plot of '{left}' and '{right}' and check important segments.",
                float(adjusted_p),
            ))
    return findings


def group_comparisons(
    df: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    method: str,
) -> list[Finding]:
    """Compare numeric outcomes across groups using an explicitly selected test."""
    allowed = {"Welch / ANOVA", "Mann-Whitney / Kruskal-Wallis"}
    if method not in allowed:
        raise ValueError(f"Unsupported group comparison: {method}")

    rows = []
    for category in categorical_columns[:MAX_CATEGORICAL_COLUMNS]:
        for value in numeric_columns[:10]:
            data = df[[category, value]].dropna()
            groups = [
                group[value].to_numpy()
                for _, group in data.groupby(category, observed=True)
                if len(group) >= 5
            ]
            if len(groups) < 2:
                continue

            if method == "Welch / ANOVA":
                if len(groups) == 2:
                    result = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                    test_name = "Welch's t-test"
                else:
                    result = stats.f_oneway(*groups, equal_var=False)
                    test_name = "Welch's one-way ANOVA"
            elif len(groups) == 2:
                result = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                test_name = "Mann-Whitney U"
            else:
                result = stats.kruskal(*groups)
                test_name = "Kruskal-Wallis"

            rows.append((
                category,
                value,
                float(result.statistic),
                float(result.pvalue),
                len(groups),
                sum(map(len, groups)),
                test_name,
            ))

    adjusted = bh_adjust([row[3] for row in rows])
    findings = []
    for row, adjusted_p in zip(rows, adjusted):
        category, value, statistic, _pvalue, group_count, sample_count, test_name = row
        if adjusted_p >= ALPHA:
            continue
        findings.append(Finding(
            "group_comparison",
            f"'{value}' differs across groups of '{category}' ({test_name}).",
            {
                "test": test_name,
                "statistic": round(statistic, 4),
                "groups": group_count,
                "n": sample_count,
                "p_adj": float(adjusted_p),
            },
            "Statistical significance does not show the size or practical importance of a difference.",
            ["Observations are assumed independent; post-hoc pairwise comparisons are not included."],
            f"Review group sizes and distributions for '{value}' by '{category}'.",
            float(adjusted_p),
        ))
    return findings


def categorical_associations(
    df: pd.DataFrame, categorical_columns: list[str]
) -> list[Finding]:
    """Test associations between low-cardinality categorical columns."""
    rows = []
    columns = categorical_columns[:MAX_CATEGORICAL_COLUMNS]
    for left, right in itertools.combinations(columns, 2):
        pair = df[[left, right]].dropna()
        if pair[left].nunique() < 2 or pair[right].nunique() < 2:
            continue
        table = pd.crosstab(pair[left], pair[right])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        effect_chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
        if table.shape == (2, 2):
            statistic, pvalue = stats.fisher_exact(table.to_numpy())
            test_name = "Fisher's exact test"
        else:
            statistic, pvalue, _, _ = stats.chi2_contingency(table)
            test_name = "Chi-square test of independence"
        n = int(table.to_numpy().sum())
        dimension = min(table.shape[0] - 1, table.shape[1] - 1)
        cramers_v = float(np.sqrt(effect_chi2 / (n * dimension))) if n and dimension else 0.0
        rows.append((left, right, float(statistic), float(pvalue), n, test_name, cramers_v))

    adjusted = bh_adjust([row[3] for row in rows])
    findings = []
    for row, adjusted_p in zip(rows, adjusted):
        left, right, statistic, _pvalue, n, test_name, effect_size = row
        if adjusted_p < ALPHA:
            findings.append(Finding(
                "categorical_association",
                f"'{left}' and '{right}' are associated ({test_name}).",
                {
                    "test": test_name,
                    "statistic": round(statistic, 4),
                    "cramers_v": round(effect_size, 4),
                    "n": n,
                    "p_adj": float(adjusted_p),
                },
                "Association does not establish causality. Sparse categories can make chi-square "
                "approximations unreliable.",
                ["Results may be affected by unobserved confounding."],
                "Inspect the contingency table and consider combining justified rare categories.",
                float(adjusted_p),
            ))
    return findings


def normality_checks(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Run Shapiro-Wilk checks on bounded samples and return all tested p-values."""
    rows = []
    for column in numeric_columns[:MAX_NUMERIC_COLUMNS]:
        values = df[column].dropna().to_numpy()
        if len(values) < 3 or np.ptp(values) == 0:
            continue
        sampled = len(values) > 5000
        if sampled:
            rng = np.random.default_rng(0)
            values = rng.choice(values, size=5000, replace=False)
        result = stats.shapiro(values)
        rows.append({
            "column": column,
            "n_tested": len(values),
            "sampled": sampled,
            "shapiro_statistic": float(result.statistic),
            "p_value": float(result.pvalue),
        })
    if rows:
        adjusted = bh_adjust([row["p_value"] for row in rows])
        for row, pvalue in zip(rows, adjusted):
            row["p_adj"] = float(pvalue)
            row["interpretation"] = (
                "Evidence against normality at 0.05 (FDR-adjusted)"
                if pvalue < ALPHA else
                "No evidence against normality at 0.05 (FDR-adjusted)"
            )
    return pd.DataFrame(rows)
