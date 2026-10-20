"""Opt-in, auditable dataframe cleaning operations."""
from collections.abc import Iterable

import pandas as pd


MISSING_STRATEGIES = {"keep", "drop_rows", "impute"}
OUTLIER_METHODS = {"none", "clip_iqr"}


def clean_dataframe(
    df: pd.DataFrame,
    *,
    trim_strings: bool = False,
    normalize_categories: bool = False,
    convert_numeric: bool = False,
    drop_empty_columns: bool = False,
    drop_duplicates: bool = False,
    missing_strategy: str = "keep",
    outlier_method: str = "none",
    drop_columns: Iterable[str] = (),
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Return a cleaned copy and a log of operations that changed the data."""
    if missing_strategy not in MISSING_STRATEGIES:
        raise ValueError(f"Unknown missing-value strategy: {missing_strategy}")
    if outlier_method not in OUTLIER_METHODS:
        raise ValueError(f"Unknown outlier method: {outlier_method}")

    result = df.copy()
    changes: list[dict[str, object]] = []

    selected_drops = list(drop_columns)
    unknown = [column for column in selected_drops if column not in result.columns]
    if unknown:
        raise ValueError(f"Cannot drop unknown columns: {unknown}")
    if selected_drops:
        result = result.drop(columns=selected_drops)
        changes.append({
            "operation": "Drop selected columns",
            "affected": len(selected_drops),
            "details": ", ".join(map(str, selected_drops)),
        })
        if result.shape[1] == 0:
            raise ValueError("Cleaning options cannot remove every column.")

    if trim_strings:
        trimmed = 0
        for column in result.select_dtypes(include=["object", "string"]).columns:
            original = result[column]
            cleaned = original.map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            trimmed += int((original.fillna("").astype(str) != cleaned.fillna("").astype(str)).sum())
            result[column] = cleaned
        if trimmed:
            changes.append({
                "operation": "Trim string whitespace",
                "affected": trimmed,
                "details": "Changed string cells",
            })

    if normalize_categories:
        normalized = 0
        for column in result.select_dtypes(include=["object", "string"]).columns:
            values = result[column]
            canonical: dict[str, str] = {}
            for value in values.dropna():
                key = " ".join(str(value).split()).casefold()
                canonical.setdefault(key, " ".join(str(value).split()))
            cleaned = values.map(
                lambda value: canonical.get(" ".join(str(value).split()).casefold(), value)
                if isinstance(value, str) else value
            )
            normalized += int(
                (values.fillna("").astype(str) != cleaned.fillna("").astype(str)).sum()
            )
            result[column] = cleaned
        if normalized:
            changes.append({
                "operation": "Normalize category labels",
                "affected": normalized,
                "details": "Collapsed case/whitespace variants to first-seen labels",
            })

    if convert_numeric:
        for column in result.select_dtypes(include=["object", "string"]).columns:
            original = result[column]
            non_missing = int(original.notna().sum())
            if not non_missing:
                continue
            converted = pd.to_numeric(original, errors="coerce")
            valid = int(converted.notna().sum())
            if valid / non_missing >= 0.9:
                coerced = non_missing - valid
                result[column] = converted
                changes.append({
                    "operation": "Convert text to numeric",
                    "affected": valid,
                    "details": f"{column} converted; {coerced} non-empty values became missing",
                })

    if drop_empty_columns:
        empty_columns = result.columns[result.isna().all()].tolist()
        if empty_columns:
            result = result.drop(columns=empty_columns)
            changes.append({
                "operation": "Drop all-missing columns",
                "affected": len(empty_columns),
                "details": ", ".join(map(str, empty_columns)),
            })
            if result.shape[1] == 0:
                raise ValueError("All columns are empty; cannot remove every column.")

    if drop_duplicates:
        duplicate_count = int(result.duplicated().sum())
        if duplicate_count:
            result = result.drop_duplicates().reset_index(drop=True)
            changes.append({
                "operation": "Remove duplicate rows",
                "affected": duplicate_count,
                "details": "Kept the first occurrence",
            })

    if missing_strategy == "drop_rows":
        before = len(result)
        result = result.dropna().reset_index(drop=True)
        removed = before - len(result)
        if removed:
            changes.append({
                "operation": "Drop rows with missing values",
                "affected": removed,
                "details": f"Removed {removed} rows with at least one missing cell",
            })
    elif missing_strategy == "impute":
        filled = 0
        for column in result.columns:
            missing = int(result[column].isna().sum())
            if not missing:
                continue
            if pd.api.types.is_numeric_dtype(result[column]):
                fill_value = result[column].median()
            elif not pd.api.types.is_datetime64_any_dtype(result[column]):
                modes = result[column].mode(dropna=True)
                fill_value = modes.iloc[0] if not modes.empty else None
            else:
                continue
            if fill_value is not None and not pd.isna(fill_value):
                result[column] = result[column].fillna(fill_value)
                filled += missing
        if filled:
            changes.append({
                "operation": "Impute missing values",
                "affected": filled,
                "details": "Numeric median; categorical mode",
            })

    if outlier_method == "clip_iqr":
        capped = 0
        for column in result.select_dtypes(include="number").columns:
            values = result[column].dropna()
            if len(values) < 4:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            spread = q3 - q1
            if spread <= 0:
                continue
            lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
            outside = result[column].notna() & (
                (result[column] < lower) | (result[column] > upper)
            )
            count = int(outside.sum())
            if count:
                result[column] = result[column].clip(lower=lower, upper=upper)
                capped += count
        if capped:
            changes.append({
                "operation": "Cap numeric outliers",
                "affected": capped,
                "details": "Values clipped to 1.5 x IQR bounds",
            })

    if len(df) and result.empty:
        raise ValueError("These cleaning options would remove every row. Change the selected options.")
    return result, changes


def validate_numeric_range(
    df: pd.DataFrame,
    column: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, object] | None:
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Range validation requires an existing numeric column: {column}")
    if minimum is None and maximum is None:
        return None
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("Minimum allowed value cannot exceed maximum allowed value.")
    values = df[column]
    invalid = pd.Series(False, index=df.index)
    if minimum is not None:
        invalid |= values.notna() & (values < minimum)
    if maximum is not None:
        invalid |= values.notna() & (values > maximum)
    count = int(invalid.sum())
    if not count:
        return None
    bounds = f"{minimum if minimum is not None else '-infinity'} to {maximum if maximum is not None else 'infinity'}"
    return {
        "issue": "numeric_range_validation",
        "detail": f"'{column}': {count} values outside configured range ({bounds})",
        "severity": "medium",
    }
