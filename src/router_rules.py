"""Version A: deterministic baseline router."""


def select_analyses(p: dict) -> list[str]:
    tasks = []
    if len(p["numeric_columns"]) >= 2:
        tasks.append("correlation")
    if p["categorical_columns"] and p["numeric_columns"]:
        tasks.append("group_comparison")
    if p["has_datetime"] and p["numeric_columns"]:
        tasks.append("time_trend")
    if p["numeric_columns"]:
        tasks.append("anomaly")
    return tasks
