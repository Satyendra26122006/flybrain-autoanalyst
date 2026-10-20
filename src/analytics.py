"""Bounded, reproducible ML, forecasting, KPI, text, and SQL analysis helpers."""
from __future__ import annotations

import re
import itertools
import sqlite3
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, inspect, text
from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MAX_MODEL_ROWS = 100_000
MAX_FEATURES = 100

CLASSIFIERS = {
    "Logistic regression": LogisticRegression(max_iter=1000),
    "Random forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient boosting": GradientBoostingClassifier(random_state=42),
}
REGRESSORS = {
    "Linear regression": LinearRegression(),
    "Random forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient boosting": GradientBoostingRegressor(random_state=42),
}


def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=50)),
            ]),
            categorical,
        ))
    if not transformers:
        raise ValueError("No usable features remain after removing missing values.")
    return ColumnTransformer(transformers, remainder="drop")


@dataclass
class ModelResult:
    task: str
    model_name: str
    metrics: dict
    predictions: pd.DataFrame
    importance: pd.DataFrame
    warnings: list[str]


def fit_supervised(
    df: pd.DataFrame,
    *,
    target: str,
    features: list[str],
    task: str,
    model_name: str,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> ModelResult:
    """Train/evaluate a selected model with a held-out test split and CV."""
    if task not in {"classification", "regression"}:
        raise ValueError("Task must be classification or regression.")
    models = CLASSIFIERS if task == "classification" else REGRESSORS
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    if target not in df.columns or target in features:
        raise ValueError("Choose a target column that is not also a feature.")
    features = list(dict.fromkeys(column for column in features if column != target))
    if not features or len(features) > MAX_FEATURES:
        raise ValueError(f"Choose between 1 and {MAX_FEATURES} feature columns.")
    data = df[[*features, target]].dropna(subset=[target])
    warnings = []
    if len(data) > MAX_MODEL_ROWS:
        data = data.sample(MAX_MODEL_ROWS, random_state=seed)
        warnings.append(f"Training used a reproducible sample of {MAX_MODEL_ROWS:,} rows.")

    X, y = data[features].copy(), data[target]
    for column in X.select_dtypes(include="datetime").columns:
        values = pd.to_datetime(X[column], errors="coerce")
        X[f"{column}__year"] = values.dt.year
        X[f"{column}__month"] = values.dt.month
        X[f"{column}__day"] = values.dt.day
        X[f"{column}__weekday"] = values.dt.dayofweek
        X = X.drop(columns=column)
        warnings.append(f"Expanded datetime feature '{column}' into calendar components.")
    features = X.columns.tolist()
    if task == "classification":
        if y.nunique() < 2:
            raise ValueError("Classification requires at least two target classes.")
        if y.value_counts().min() < 2:
            raise ValueError("Each target class needs at least two rows for a stratified split.")
        class_count = int(y.nunique())
        test_count = int(np.ceil(len(y) * test_fraction))
        train_count = len(y) - test_count
        stratify = (
            y if y.value_counts().min() >= 2
            and test_count >= class_count and train_count >= class_count
            else None
        )
        if stratify is None:
            warnings.append("Class stratification was not feasible; the split is random.")
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        if len(y) < 10:
            raise ValueError("Regression requires at least 10 rows with numeric target values.")
        stratify = None
    if len(y) < 20:
        raise ValueError("Train/test evaluation requires at least 20 usable rows.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=seed,
        stratify=stratify,
    )
    preprocessor = _preprocessor(X_train)
    estimator = clone(models[model_name])
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)

    if task == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "classes": int(y.nunique()),
            "test_rows": int(len(y_test)),
            "classification_report": classification_report(
                y_test, predicted, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
        }
        if y.nunique() == 2 and hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    else:
        metrics = {
            "mae": float(mean_absolute_error(y_test, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
            "r_squared": float(r2_score(y_test, predicted)),
            "test_rows": int(len(y_test)),
        }

    cv_count = min(5, int(y_train.value_counts().min())) if task == "classification" else 5
    if cv_count >= 2 and len(y_train) >= cv_count * 2:
        scoring = "accuracy" if task == "classification" else "neg_mean_absolute_error"
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv_count, scoring=scoring, n_jobs=1
        )
        cv_values = scores if task == "classification" else -scores
        metrics["cross_validation_score_mean"] = float(np.mean(cv_values))
        metrics["cross_validation_score_std"] = float(np.std(cv_values))
        metrics["cross_validation_metric"] = "accuracy" if task == "classification" else "MAE"
    else:
        warnings.append("Cross-validation skipped because the training split is too small.")

    importance = pd.DataFrame()
    if len(X_test) >= 10:
        scoring = "accuracy" if task == "classification" else "r2"
        result = permutation_importance(
            pipeline, X_test, y_test, n_repeats=5, random_state=seed,
            scoring=scoring, n_jobs=1,
        )
        importance = pd.DataFrame({
            "feature": features,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False)
    return ModelResult(
        task,
        model_name,
        metrics,
        pd.DataFrame({"actual": y_test, "predicted": predicted}, index=y_test.index),
        importance,
        warnings,
    )


def cluster_and_reduce(
    df: pd.DataFrame,
    features: list[str],
    *,
    algorithm: str = "K-means",
    clusters: int = 3,
    eps: float = 0.5,
    min_samples: int = 5,
) -> tuple[pd.DataFrame, dict]:
    numeric_features = df[features].select_dtypes(include="number")
    if numeric_features.shape[1] < 2:
        raise ValueError("Clustering and PCA need at least two numeric features.")
    data = numeric_features.iloc[:MAX_MODEL_ROWS]
    matrix = SimpleImputer(strategy="median").fit_transform(data)
    scaled = StandardScaler().fit_transform(matrix)
    if algorithm == "K-means":
        labels = KMeans(n_clusters=clusters, random_state=42, n_init=10).fit_predict(scaled)
    elif algorithm == "DBSCAN":
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(scaled)
    result = pd.DataFrame({
        "row_index": data.index,
        "cluster": labels,
        "PC1": coordinates[:, 0],
        "PC2": coordinates[:, 1],
    })
    return result, {
        "rows": len(result),
        "clusters_found": len(set(labels) - {-1}),
        "noise_rows": int(np.sum(labels == -1)),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }


def isolation_anomalies(
    df: pd.DataFrame, features: list[str], contamination: float = 0.03
) -> pd.DataFrame:
    numeric = df[features].select_dtypes(include="number")
    if numeric.shape[1] < 1:
        raise ValueError("Isolation Forest requires at least one numeric feature.")
    if not 0 < contamination <= 0.2:
        raise ValueError("Contamination must be greater than 0 and at most 0.2.")
    data = numeric.iloc[:MAX_MODEL_ROWS]
    matrix = SimpleImputer(strategy="median").fit_transform(data)
    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    labels = model.fit_predict(matrix)
    return pd.DataFrame({
        "row_index": data.index,
        "anomaly": labels == -1,
        "anomaly_score": -model.score_samples(matrix),
    }).sort_values("anomaly_score", ascending=False)


def bootstrap_mean_difference(
    first: pd.Series,
    second: pd.Series,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> dict:
    a = pd.to_numeric(first, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(second, errors="coerce").dropna().to_numpy()
    if min(len(a), len(b)) < 2:
        raise ValueError("Each group must contain at least two numeric observations.")
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations)
    for index in range(iterations):
        differences[index] = (
            rng.choice(a, len(a), replace=True).mean()
            - rng.choice(b, len(b), replace=True).mean()
        )
    return {
        "mean_difference": float(a.mean() - b.mean()),
        "ci_95": [float(x) for x in np.quantile(differences, [0.025, 0.975])],
        "n_first": int(len(a)),
        "n_second": int(len(b)),
        "iterations": iterations,
        "method": "Nonparametric bootstrap percentile interval",
    }


def summarize_business_metrics(
    df: pd.DataFrame,
    *,
    revenue: str | None = None,
    cost: str | None = None,
    date: str | None = None,
    category: str | None = None,
    target: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary: dict[str, float] = {}
    detail = pd.DataFrame()
    if revenue:
        revenue_values = pd.to_numeric(df[revenue], errors="coerce")
        summary["revenue_total"] = float(revenue_values.sum())
        summary["revenue_mean"] = float(revenue_values.mean())
    if cost:
        cost_values = pd.to_numeric(df[cost], errors="coerce")
        summary["cost_total"] = float(cost_values.sum())
    if revenue and cost:
        profit = revenue_values - cost_values
        summary["profit_total"] = float(profit.sum())
        revenue_total = float(revenue_values.sum())
        summary["margin_percent"] = (
            float(100 * profit.sum() / revenue_total) if revenue_total else float("nan")
        )
    if category and revenue:
        detail = (
            pd.DataFrame({category: df[category], "revenue": revenue_values})
            .groupby(category, dropna=False)["revenue"]
            .agg(["sum", "mean", "count"])
            .sort_values("sum", ascending=False)
        )
    if date and revenue:
        date_values = pd.to_datetime(df[date], errors="coerce")
        valid = pd.DataFrame({"date": date_values, "revenue": revenue_values}).dropna()
        if not valid.empty:
            period_revenue = valid.set_index("date")["revenue"].resample("ME").sum()
            detail.attrs["period_revenue"] = period_revenue
            if len(period_revenue) >= 2:
                previous, current = period_revenue.iloc[-2], period_revenue.iloc[-1]
                summary["latest_period_growth_percent"] = (
                    float(100 * (current - previous) / abs(previous))
                    if previous else float("nan")
                )
    if target and revenue:
                target_values = pd.to_numeric(df[target], errors="coerce")
                actual_total = float(revenue_values.sum())
                target_total = float(target_values.sum())
                summary["target_total"] = target_total
                summary["target_attainment_percent"] = (
                    float(100 * actual_total / target_total) if target_total else float("nan")
                )
    return pd.DataFrame([summary]) if summary else pd.DataFrame(), detail


def text_term_frequency(series: pd.Series, top_n: int = 30) -> pd.DataFrame:
    text = series.dropna().astype(str).str.lower()
    words = Counter()
    for value in text:
        words.update(re.findall(r"\b[a-z][a-z0-9']{2,}\b", value))
    stop_words = {
        "the", "and", "for", "that", "with", "this", "from", "are", "was",
        "were", "have", "has", "had", "not", "but", "you", "your", "our",
        "they", "their", "its", "can", "will", "all", "any", "one", "too",
    }
    return pd.DataFrame(
        [(word, count) for word, count in words.most_common() if word not in stop_words][:top_n],
        columns=["term", "frequency"],
    )


def read_only_sqlite_query(source, query: str, limit: int = 10_000) -> pd.DataFrame:
    """Execute one SELECT/CTE query against an uploaded SQLite file in memory."""
    normalized = query.strip()
    if not re.match(r"^(select|with)\b", normalized, flags=re.IGNORECASE):
        raise ValueError("Only a single read-only SELECT or WITH query is allowed.")
    if ";" in normalized.rstrip(";"):
        raise ValueError("Only one SQL statement may be run at a time.")
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(source.getvalue())
        connection.execute("PRAGMA query_only = ON")
        result = pd.read_sql_query(
            f"SELECT * FROM ({normalized.rstrip(';')}) LIMIT ?",
            connection,
            params=(limit,),
        )
    except sqlite3.Error as error:
        raise ValueError(f"SQL query failed: {error}") from error
    finally:
        connection.close()
    return result


def validate_database_url(database_url: str) -> str:
    if not re.match(r"^(postgresql(\+psycopg)?|mysql(\+pymysql)?)://", database_url):
        raise ValueError("Only PostgreSQL and MySQL SQLAlchemy URLs are supported.")
    return database_url


def remote_database_tables(database_url: str) -> list[str]:
    engine = create_engine(validate_database_url(database_url), pool_pre_ping=True)
    try:
        return inspect(engine).get_table_names()
    finally:
        engine.dispose()


def read_only_remote_query(
    database_url: str,
    query: str,
    *,
    limit: int = 10_000,
) -> pd.DataFrame:
    """Run a single SELECT/CTE with a read-only transaction on PostgreSQL/MySQL."""
    normalized = query.strip()
    if not re.match(r"^(select|with)\b", normalized, flags=re.IGNORECASE):
        raise ValueError("Only a single read-only SELECT or WITH query is allowed.")
    if ";" in normalized.rstrip(";"):
        raise ValueError("Only one SQL statement may be run at a time.")
    url = validate_database_url(database_url)
    engine = create_engine(url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                if url.startswith("postgresql"):
                    connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                else:
                    connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                result = pd.read_sql_query(
                    text(f"SELECT * FROM ({normalized.rstrip(';')}) AS analyst_result LIMIT :limit"),
                    connection,
                    params={"limit": limit},
                )
                transaction.rollback()
                return result
            except Exception:
                transaction.rollback()
                raise
    finally:
        engine.dispose()


def permutation_mean_difference(
    first: pd.Series,
    second: pd.Series,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> dict:
    a = pd.to_numeric(first, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(second, errors="coerce").dropna().to_numpy()
    if min(len(a), len(b)) < 2:
        raise ValueError("Each group must contain at least two numeric observations.")
    observed = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(iterations):
        shuffled = rng.permutation(pooled)
        difference = shuffled[:len(a)].mean() - shuffled[len(a):].mean()
        extreme += abs(difference) >= abs(observed)
    return {
        "mean_difference": observed,
        "p_value": float((extreme + 1) / (iterations + 1)),
        "n_first": int(len(a)),
        "n_second": int(len(b)),
        "iterations": iterations,
        "method": "Two-sided permutation test",
    }


def cohort_retention(
    df: pd.DataFrame,
    customer_column: str,
    date_column: str,
) -> pd.DataFrame:
    data = df[[customer_column, date_column]].copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna().drop_duplicates()
    if data.empty:
        raise ValueError("No valid customer/date observations for cohort analysis.")
    data["activity_period"] = data[date_column].dt.to_period("M")
    data["cohort_period"] = data.groupby(customer_column)["activity_period"].transform("min")
    offsets = (
        (data["activity_period"].dt.year - data["cohort_period"].dt.year) * 12
        + data["activity_period"].dt.month
        - data["cohort_period"].dt.month
    )
    data["period_number"] = offsets
    counts = data.groupby(["cohort_period", "period_number"])[customer_column].nunique()
    matrix = counts.unstack("period_number")
    if 0 not in matrix:
        raise ValueError("No first-period customer cohort was produced.")
    return matrix.div(matrix[0], axis=0).fillna(0)


def market_basket_pairs(
    series: pd.Series,
    *,
    delimiter: str = ",",
    minimum_support: float = 0.01,
    top_n: int = 100,
) -> pd.DataFrame:
    transactions = [
        set(item.strip() for item in str(value).split(delimiter) if item.strip())
        for value in series.dropna()
    ]
    if not transactions:
        raise ValueError("No transaction rows were provided.")
    if not 0 < minimum_support <= 1:
        raise ValueError("Minimum support must be in the interval (0, 1].")
    item_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for items in transactions:
        item_counts.update(items)
        pair_counts.update(itertools.combinations(sorted(items), 2))
    rows = []
    transaction_count = len(transactions)
    for (left, right), count in pair_counts.items():
        support = count / transaction_count
        if support >= minimum_support:
            rows.append({
                "item_a": left,
                "item_b": right,
                "support": support,
                "confidence_a_to_b": count / item_counts[left],
                "confidence_b_to_a": count / item_counts[right],
                "lift": support / ((item_counts[left] / transaction_count)
                                   * (item_counts[right] / transaction_count)),
                "transactions": transaction_count,
            })
    return pd.DataFrame(rows).sort_values("lift", ascending=False).head(top_n) if rows else pd.DataFrame(
        columns=[
            "item_a", "item_b", "support", "confidence_a_to_b",
            "confidence_b_to_a", "lift", "transactions",
        ]
    )
