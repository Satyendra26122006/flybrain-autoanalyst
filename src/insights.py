"""Run the selected analysis tools."""
from . import statistics as S
from .quality import check

TOOLS = {"correlation": S.correlations, "group_comparison": S.group_comparisons,
         "time_trend": S.time_trends, "anomaly": S.anomalies}
TASKS = list(TOOLS)


def run(df, p, tasks):
    findings = []
    for t in tasks:
        if t in TOOLS:
            findings += TOOLS[t](df, p)
    return findings


def run_all(df, p):
    return {"quality": check(df), "findings": run(df, p, TASKS)}
