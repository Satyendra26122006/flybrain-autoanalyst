"""Analysis tools. Each returns a list of Finding. Exact math, no LLM."""
import itertools
from scipy import stats
from .validation import Finding, bh_adjust, corr_ci

ALPHA = 0.05
CAUSAL = "Association only; confounders or reverse causation are possible."


def correlations(df, p):
    cols, rows = p["numeric_columns"][:30], []
    for a, b in itertools.combinations(cols, 2):
        s = df[[a, b]].dropna()
        if len(s) >= 10 and s[a].nunique() > 1 and s[b].nunique() > 1:
            r, pv = stats.pearsonr(s[a], s[b]); rows.append((a, b, r, pv, len(s)))
    adj = bh_adjust([r[3] for r in rows])
    out = []
    for (a, b, r, pv, n), pa in zip(rows, adj):
        if pa < ALPHA and abs(r) >= 0.3:
            lo, hi = corr_ci(r, n)
            out.append(Finding("correlation", f"'{a}' and '{b}' are {'positively' if r > 0 else 'negatively'} correlated.",
                {"r": round(float(r), 3), "n": n, "p_adj": float(pa)}, f"95% CI for r: [{lo:.2f}, {hi:.2f}]",
                [CAUSAL, "Pearson assumes a linear relationship."], f"Plot '{a}' vs '{b}' and check by segment.", float(pa)))
    return out


def group_comparisons(df, p):
    rows = []
    for c in p["categorical_columns"][:8]:
        for v in p["numeric_columns"][:10]:
            g = [x[v].dropna().values for _, x in df.groupby(c) if x[v].notna().sum() >= 5]
            if len(g) >= 2:
                h, pv = stats.kruskal(*g); rows.append((c, v, pv, len(g), sum(map(len, g))))
    adj = bh_adjust([r[2] for r in rows]); out = []
    for (c, v, pv, k, n), pa in zip(rows, adj):
        if pa < ALPHA:
            means = df.groupby(c)[v].mean(); hi, lo = means.idxmax(), means.idxmin()
            out.append(Finding("group_comparison", f"'{v}' differs across '{c}' (highest: {hi}, lowest: {lo}).",
                {"groups": k, "n": n, "mean_high": round(float(means.max()), 3), "mean_low": round(float(means.min()), 3), "p_adj": float(pa)},
                "Kruskal-Wallis test, FDR-adjusted p-value.", [CAUSAL, "Check group sizes and mix effects."],
                f"Control for other variables that differ between '{c}' groups.", float(pa)))
    return out


def time_trends(df, p):
    out = []
    if not p["datetime_columns"]:
        return out
    t = p["datetime_columns"][0]
    cols = p["numeric_columns"][:10]
    for v in cols:
        s = df[[t, v]].dropna()
        if len(s) < 10:
            continue
        x = (s[t] - s[t].min()).dt.total_seconds() / 86400
        res = stats.linregress(x, s[v])
        if res.pvalue < ALPHA / max(1, len(cols)) and abs(res.rvalue) >= 0.2:
            out.append(Finding("time_trend", f"'{v}' {'rises' if res.slope > 0 else 'falls'} over time.",
                {"slope_per_day": float(res.slope), "r": round(float(res.rvalue), 3), "n": len(s), "p": float(res.pvalue)},
                f"Slope SE: {res.stderr:.4g} (Bonferroni-corrected threshold).", ["Linear fit ignores seasonality.", CAUSAL],
                "Check seasonality and structural breaks.", float(res.pvalue)))
    return out


def anomalies(df, p):
    out = []
    for v in p["numeric_columns"][:15]:
        s = df[v].dropna()
        if len(s) < 20 or s.std() == 0:
            continue
        mad = (s - s.median()).abs().median() or 1e-9
        z = 0.6745 * (s - s.median()) / mad
        n = int((z.abs() > 5).sum())
        if 0 < n <= 0.05 * len(s):
            out.append(Finding("anomaly", f"'{v}' has {n} unusual values (robust z > 5).",
                {"count": n, "n": len(s), "max_abs_z": round(float(z.abs().max()), 2)}, "Robust z-score (MAD).",
                ["May be valid rare events, not errors."], "Inspect those rows with a domain expert."))
    return out
