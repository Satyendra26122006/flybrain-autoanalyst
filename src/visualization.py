"""Interactive Plotly charts for exploring uploaded datasets."""
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

MAX_SCATTER_POINTS = 5_000


def distribution_figure(df: pd.DataFrame, column: str, bins: int = 30):
    return px.histogram(
        df,
        x=column,
        nbins=bins,
        marginal="box",
        title=f"Distribution of {column}",
        labels={column: column},
    )


def scatter_figure(
    df: pd.DataFrame, x: str, y: str, color: str | None = None
):
    columns = list(dict.fromkeys([x, y] + ([color] if color else [])))
    data = df[columns].dropna(subset=[x, y])
    if len(data) > MAX_SCATTER_POINTS:
        data = data.sample(MAX_SCATTER_POINTS, random_state=0)
    return px.scatter(
        data,
        x=x,
        y=y,
        color=color,
        title=f"{y} vs {x}",
        hover_data=[column for column in columns if column not in (x, y)],
    )


def correlation_figure(df: pd.DataFrame, columns: list[str]):
    data = df[columns].dropna(axis=1, how="all")
    data = data.loc[:, data.nunique(dropna=True) > 1].iloc[:, :20]
    if data.shape[1] < 2:
        return None
    corr = data.corr()
    return px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title="Pearson correlation matrix (up to 20 columns)",
    )


def category_figure(df: pd.DataFrame, column: str):
    counts = (
        df[column]
        .astype("string")
        .fillna("(missing)")
        .value_counts()
        .rename_axis("category")
        .reset_index(name="count")
    )
    if len(counts) > 20:
        counts = pd.concat(
            [
                counts.iloc[:19],
                pd.DataFrame([{"category": "Other", "count": counts.iloc[19:]["count"].sum()}]),
            ],
            ignore_index=True,
        )
    return px.bar(
        counts,
        x="category",
        y="count",
        title=f"Counts for {column} (top 20 categories)",
        labels={"category": column},
    )


def time_series_figure(df: pd.DataFrame, date_column: str, value_column: str):
    data = df[[date_column, value_column]].dropna().sort_values(date_column)
    if len(data) > MAX_SCATTER_POINTS:
        step = math.ceil(len(data) / MAX_SCATTER_POINTS)
        data = data.iloc[::step]
    return px.line(
        data,
        x=date_column,
        y=value_column,
        title=f"{value_column} over time",
    )


def missingness_figure(df: pd.DataFrame):
    missing = (
        df.isna()
        .mean()
        .mul(100)
        .rename("missing_percent")
        .rename_axis("column")
        .reset_index()
        .sort_values("missing_percent", ascending=False)
    )
    missing = missing[missing["missing_percent"] > 0]
    if missing.empty:
        return None
    return px.bar(
        missing,
        x="column",
        y="missing_percent",
        title="Missing values by column",
        labels={"missing_percent": "Missing (%)", "column": "Column"},
    )


def boxplot_figure(df: pd.DataFrame, columns: list[str]):
    selected = columns[:20]
    sample = df[selected]
    if len(sample) > MAX_SCATTER_POINTS:
        sample = sample.sample(MAX_SCATTER_POINTS, random_state=0)
    data = sample.melt(var_name="column", value_name="value").dropna()
    if data.empty:
        return None
    return px.box(
        data,
        x="column",
        y="value",
        points=False,
        title="Numeric spread and potential outliers",
    )


def pairplot_figure(df: pd.DataFrame, columns: list[str]):
    selected = columns[:5]
    data = df[selected].dropna()
    if len(data) > MAX_SCATTER_POINTS:
        data = data.sample(MAX_SCATTER_POINTS, random_state=0)
    if len(selected) < 2 or data.empty:
        return None
    return px.scatter_matrix(
        data,
        dimensions=selected,
        title="Pairwise numeric relationships",
    )


def grouped_boxplot_figure(df: pd.DataFrame, category: str, value: str):
    data = df[[category, value]].dropna()
    if data.empty:
        return None
    counts = data[category].value_counts()
    selected = counts.head(20).index
    data = data[data[category].isin(selected)]
    data[category] = data[category].astype(str)
    return px.box(
        data,
        x=category,
        y=value,
        points="outliers",
        title=f"{value} distribution by {category} (top 20 groups)",
    )


def violin_figure(df: pd.DataFrame, column: str, category: str | None = None):
    columns = [column] + ([category] if category else [])
    data = df[columns].dropna()
    if len(data) > MAX_SCATTER_POINTS:
        data = data.sample(MAX_SCATTER_POINTS, random_state=0)
    return px.violin(
        data,
        y=column,
        x=category,
        box=True,
        points=False,
        title=f"Density and distribution of {column}",
    )


def geographic_figure(
    df: pd.DataFrame,
    latitude: str,
    longitude: str,
    color: str | None = None,
):
    columns = [latitude, longitude] + ([color] if color else [])
    data = df[columns].copy()
    data[latitude] = pd.to_numeric(data[latitude], errors="coerce")
    data[longitude] = pd.to_numeric(data[longitude], errors="coerce")
    data = data.dropna(subset=[latitude, longitude])
    data = data[
        data[latitude].between(-90, 90) & data[longitude].between(-180, 180)
    ]
    if len(data) > MAX_SCATTER_POINTS:
        data = data.sample(MAX_SCATTER_POINTS, random_state=0)
    return px.scatter_geo(
        data,
        lat=latitude,
        lon=longitude,
        color=color,
        title="Geographic records",
        projection="natural earth",
    )


def pareto_figure(df: pd.DataFrame, category: str, value: str):
    ranked = (
        df.groupby(category, dropna=False)[value]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .rename("value")
        .reset_index()
    )
    if ranked.empty:
        return None
    total = ranked["value"].sum()
    ranked["cumulative_percent"] = (
        ranked["value"].cumsum() / total * 100 if total else 0
    )
    return px.bar(
        ranked,
        x=category,
        y="value",
        title=f"Pareto ranking: {value} by {category}",
        hover_data=["cumulative_percent"],
    )


def forecast_figure(series: pd.Series, forecast: pd.DataFrame, holdout: pd.DataFrame):
    history = pd.DataFrame({"date": series.index, "value": series.to_numpy(), "kind": "history"})
    test = holdout.reset_index(names="date").rename(columns={
        "actual": "value",
        "holdout_forecast": "holdout forecast",
    })
    test_long = test.melt(
        id_vars="date",
        value_vars=["value", "holdout forecast"],
        var_name="kind",
        value_name="series_value",
    ).rename(columns={"series_value": "value"})
    future = forecast.rename(columns={"forecast": "value"}).assign(kind="future forecast")
    plot_data = pd.concat(
        [history, test_long, future[["date", "value", "kind"]]],
        ignore_index=True,
    )
    return px.line(
        plot_data,
        x="date",
        y="value",
        color="kind",
        title="History, holdout backtest, and forecast",
    )


def residual_figure(predictions: pd.DataFrame):
    data = predictions.copy()
    data["residual"] = data["actual"] - data["predicted"]
    return px.scatter(
        data,
        x="predicted",
        y="residual",
        title="Regression residuals vs predicted",
        labels={"predicted": "Predicted value", "residual": "Residual"},
    ).add_hline(y=0, line_dash="dash")


def cluster_figure(result: pd.DataFrame):
    return px.scatter(
        result,
        x="PC1",
        y="PC2",
        color=result["cluster"].astype(str),
        hover_data=["row_index"],
        title="Clusters projected onto the first two principal components",
    )


def correlation_network_figure(df: pd.DataFrame, columns: list[str], threshold: float = 0.7):
    data = df[columns[:20]].dropna()
    data = data.loc[:, data.nunique(dropna=True) > 1]
    if data.shape[1] < 2:
        return None
    correlations = data.corr(method="spearman")
    graph = nx.Graph()
    graph.add_nodes_from(correlations.columns)
    for left in correlations.columns:
        for right in correlations.columns:
            coefficient = correlations.loc[left, right]
            if left < right and abs(coefficient) >= threshold:
                graph.add_edge(left, right, weight=float(coefficient))
    if not graph.edges:
        return None
    positions = nx.spring_layout(graph, seed=42)
    edge_x, edge_y = [], []
    for left, right in graph.edges:
        x0, y0 = positions[left]
        x1, y1 = positions[right]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    node_x = [positions[node][0] for node in graph.nodes]
    node_y = [positions[node][1] for node in graph.nodes]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line={"width": 1, "color": "#9AA4B2"},
        hoverinfo="none",
        name="Associations",
    ))
    figure.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(graph.nodes),
        textposition="top center",
        marker={"size": 16, "color": "#3157A4"},
        hovertemplate="%{text}<extra></extra>",
        name="Features",
    ))
    figure.update_layout(
        title=f"Strong rank-correlation network (|rho| ≥ {threshold:.2f})",
        showlegend=False,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure
