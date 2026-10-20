import numpy as np, pandas as pd, torch
from src import advanced_statistics, analytics, brain_router, cleaning, ingestion, profiling, insights, reporting, time_series, visualization
from src.validation import bh_adjust
from src.router_rules import select_analyses
from src.connectome import build_tensors, random_connectome, Connectome, shuffled, subset
from src.profiling import N_FEATURES
from src.insights import TASKS


def sample():
    rng = np.random.default_rng(0); n = 200
    a = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": a * .8 + rng.normal(scale=.5, size=n), "g": rng.choice(list("xy"), n),
                         "d": pd.date_range("2023-01-01", periods=n)})


def test_bh_monotone():
    assert (bh_adjust([0.01, 0.04, 0.03]) >= np.array([0.01, 0.04, 0.03]) - 1e-12).all()


def test_pipeline_finds_correlation():
    df = sample(); p = profiling.profile(df)
    f = insights.run(df, p, select_analyses(p))
    assert any(x.tool == "correlation" for x in f)


def test_gradients_reach_all_params():
    m = Connectome(random_connectome(100, 800), N_FEATURES, len(TASKS))
    m(torch.rand(N_FEATURES)).sum().backward()
    assert m.log_w.grad is not None and m.bias.grad is not None and m.gain.grad is not None


def test_brain_trainer_accumulates_minibatch_gradients():
    graph = random_connectome(30, 120)
    features = np.zeros((2, N_FEATURES), dtype=np.float32)
    labels = np.zeros((2, len(TASKS)), dtype=np.float32)
    model = brain_router.train(
        graph,
        features,
        labels,
        epochs=1,
        steps=1,
        staged=False,
        batch_size=2,
    )
    assert model(torch.as_tensor(features[0])).shape == (len(TASKS),)


def test_shuffle_and_subset():
    t = random_connectome(100, 800)
    assert shuffled(t)["src"].shape == t["src"].shape
    assert subset(t, torch.arange(50))["n"] == 50


def test_build_tensors_includes_isolated_neurons(tmp_path):
    connections_path = tmp_path / "connections.csv.gz"
    neurons_path = tmp_path / "neurons.csv.gz"
    pd.DataFrame({
        "pre_root_id": [100, 200],
        "post_root_id": [200, 100],
        "syn_count": [4, 6],
        "nt_type": ["ACH", "GABA"],
    }).to_csv(connections_path, index=False, compression="gzip")
    pd.DataFrame({"root_id": [100, 200, 300]}).to_csv(
        neurons_path, index=False, compression="gzip"
    )

    graph = build_tensors(connections_path, min_syn=1, neurons_path=neurons_path)
    assert graph["n"] == 3
    assert graph["root_ids"].tolist() == [100, 200, 300]
    assert len(graph["src"]) == 2
    assert graph["includes_neuron_inventory"] is True

    pd.DataFrame({"root_id": [100, 300]}).to_csv(
        neurons_path, index=False, compression="gzip"
    )
    with np.testing.assert_raises_regex(ValueError, "missing 1 connection endpoint"):
        build_tensors(connections_path, min_syn=1, neurons_path=neurons_path)


def test_brain_checkpoint_round_trip(tmp_path):
    t = random_connectome(30, 100)
    model = Connectome(t, N_FEATURES, len(TASKS))
    path = tmp_path / "brain.pt"
    brain_router.save_checkpoint(model, path)
    loaded = brain_router.load_checkpoint(path)
    x = torch.rand(N_FEATURES)
    torch.testing.assert_close(loaded(x), model(x))


def test_visualization_figures_handle_common_data():
    df = sample()
    assert visualization.distribution_figure(df, "a").data
    assert visualization.scatter_figure(df, "a", "b", "g").data
    assert visualization.correlation_figure(df, ["a", "b"]) is not None
    assert visualization.category_figure(df, "g").data
    assert visualization.time_series_figure(df, "d", "a").data


def test_correlation_figure_skips_constant_columns():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
    assert visualization.correlation_figure(df, ["a", "b"]) is None


def test_scatter_sampling_is_bounded():
    df = pd.DataFrame({"x": np.arange(6000), "y": np.arange(6000)})
    figure = visualization.scatter_figure(df, "x", "y")
    assert len(figure.data[0].x) == visualization.MAX_SCATTER_POINTS


def test_cleaning_is_opt_in_and_auditable():
    df = pd.DataFrame({
        "number": [1.0, np.nan, 3.0, 100.0, 100.0],
        "category": [" A ", "B", "B", "C", "C"],
        "text_number": ["1", "2", "3", "4", "4"],
        "empty": [np.nan] * 5,
    })
    cleaned, changes = cleaning.clean_dataframe(
        df,
        trim_strings=True,
        convert_numeric=True,
        drop_empty_columns=True,
        drop_duplicates=True,
        missing_strategy="impute",
        outlier_method="clip_iqr",
    )
    assert cleaned["category"].iloc[0] == "A"
    assert pd.api.types.is_numeric_dtype(cleaned["text_number"])
    assert cleaned["number"].isna().sum() == 0
    assert cleaned["number"].max() < 100
    assert "empty" not in cleaned
    assert len(cleaned) == 4
    assert len(changes) >= 5
    assert pd.isna(df.loc[1, "number"])


def test_cleaning_rejects_empty_result_and_unknown_columns():
    df = pd.DataFrame({"value": [np.nan, np.nan]})
    with np.testing.assert_raises(ValueError):
        cleaning.clean_dataframe(df, missing_strategy="drop_rows")
    with np.testing.assert_raises(ValueError):
        cleaning.clean_dataframe(df, drop_columns=["value"])
    with np.testing.assert_raises(ValueError):
        cleaning.clean_dataframe(df, drop_columns=["missing"])


def test_extra_eda_figures():
    df = sample()
    assert visualization.missingness_figure(df) is None
    df.loc[0, "a"] = np.nan
    assert visualization.missingness_figure(df).data
    assert visualization.boxplot_figure(df, ["a", "b"]).data
    assert visualization.pairplot_figure(df, ["a", "b"]).data
    assert visualization.grouped_boxplot_figure(df, "g", "a").data


def test_boxplot_sampling_is_bounded():
    df = pd.DataFrame({"x": np.arange(6000), "y": np.arange(6000)})
    figure = visualization.boxplot_figure(df, ["x", "y"])
    assert sum(len(trace.y) for trace in figure.data) <= visualization.MAX_SCATTER_POINTS * 2


def test_pdf_report_is_valid_and_contains_analysis():
    df = sample()
    profile = profiling.profile(df)
    findings = insights.run(df, profile, select_analyses(profile))
    pdf = reporting.to_pdf(
        "sample.csv",
        profile,
        insights.check(df),
        findings,
        df=df,
        cleaning_log=[{"operation": "Trim text", "affected": 2, "details": "Sample change"}],
        analysis_tables={"Normality checks": pd.DataFrame({"column": ["a"], "p_adj": [0.2]})},
        analysis_methods=["pearson", "spearman"],
    )
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 1000


def test_json_and_jsonl_ingestion():
    import io

    json_df = ingestion.load(io.StringIO('[{"value": 1}, {"value": 2}]'), "sample.json")
    jsonl_df = ingestion.load(
        io.StringIO('{"value": 1}\n{"value": 2}\n'), "sample.jsonl"
    )
    assert json_df["value"].tolist() == [1, 2]
    assert jsonl_df["value"].tolist() == [1, 2]


def test_sqlite_database_table_ingestion():
    import io
    import sqlite3

    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE sample_data (value INTEGER, label TEXT)")
    connection.executemany(
        "INSERT INTO sample_data VALUES (?, ?)",
        [(1, "a"), (2, "b")],
    )
    database_bytes = connection.serialize()
    connection.close()
    source = io.BytesIO(database_bytes)
    tables = ingestion.sqlite_tables(source)
    result = ingestion.load_sqlite_table(source, tables[0])
    assert tables == ["sample_data"]
    assert result.to_dict("records") == [
        {"value": 1, "label": "a"},
        {"value": 2, "label": "b"},
    ]
    with np.testing.assert_raises(ValueError):
        ingestion.load_sqlite_table(source, "unknown")


def test_selectable_statistical_test_families():
    rng = np.random.default_rng(5)
    n = 180
    group = np.repeat(["control", "treatment"], n // 2)
    metric = np.r_[rng.normal(0, 1, n // 2), rng.normal(1.5, 1, n // 2)]
    df = pd.DataFrame({
        "metric": metric,
        "rank_signal": metric + rng.normal(0, 0.2, n),
        "group": group,
        "segment": np.where(np.arange(n) < n // 2, "A", "B"),
    })
    assert advanced_statistics.correlations(df, ["metric", "rank_signal"], "Spearman")
    assert advanced_statistics.correlations(df, ["metric", "rank_signal"], "Kendall")
    assert advanced_statistics.group_comparisons(
        df, ["group"], ["metric"], "Welch / ANOVA"
    )
    assert advanced_statistics.categorical_associations(df, ["group", "segment"])
    normality = advanced_statistics.normality_checks(df, ["metric"])
    assert normality.loc[0, "n_tested"] == n


def test_supervised_models_and_unsupervised_tools():
    rng = np.random.default_rng(21)
    n = 120
    feature = rng.normal(size=n)
    df = pd.DataFrame({
        "feature": feature,
        "category": rng.choice(["a", "b"], size=n),
        "class": np.where(feature > 0, "positive", "negative"),
        "value": 2 * feature + rng.normal(0, 0.2, n),
    })
    classification = analytics.fit_supervised(
        df,
        target="class",
        features=["feature", "category"],
        task="classification",
        model_name="Logistic regression",
    )
    assert 0 <= classification.metrics["accuracy"] <= 1
    assert not classification.importance.empty
    regression = analytics.fit_supervised(
        df,
        target="value",
        features=["feature"],
        task="regression",
        model_name="Linear regression",
    )
    assert regression.metrics["rmse"] < 1
    clusters, cluster_metrics = analytics.cluster_and_reduce(
        df, ["feature", "value"], clusters=2
    )
    assert len(clusters) == len(df)
    assert cluster_metrics["clusters_found"] == 2
    anomalies = analytics.isolation_anomalies(df, ["feature"], contamination=0.05)
    assert anomalies["anomaly"].sum() >= 1


def test_resampling_business_cohort_basket_and_readonly_sqlite():
    import io
    import sqlite3

    first = pd.Series([1, 2, 3, 4, 5])
    second = pd.Series([2, 3, 4, 5, 6])
    bootstrap = analytics.bootstrap_mean_difference(first, second, iterations=200)
    permutation = analytics.permutation_mean_difference(first, second, iterations=200)
    assert bootstrap["ci_95"][0] <= bootstrap["mean_difference"] <= bootstrap["ci_95"][1]
    assert 0 < permutation["p_value"] <= 1

    df = pd.DataFrame({
        "customer": ["a", "a", "b", "b", "c"],
        "date": pd.to_datetime([
            "2024-01-02", "2024-02-02", "2024-01-08", "2024-03-01", "2024-02-09"
        ]),
        "revenue": [10, 12, 4, 7, 8],
        "cost": [4, 5, 1, 3, 2],
        "product": ["tea,coffee", "tea", "coffee,biscuit", "coffee", "tea,biscuit"],
    })
    summary, category_summary = analytics.summarize_business_metrics(
        df, revenue="revenue", cost="cost", date="date", category="product"
    )
    assert summary.loc[0, "profit_total"] == 26
    assert not category_summary.empty
    assert not category_summary.attrs["period_revenue"].empty
    retention = analytics.cohort_retention(df, "customer", "date")
    assert retention.iloc[:, 0].eq(1).all()
    pairs = analytics.market_basket_pairs(df["product"], minimum_support=0.1)
    assert not pairs.empty

    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE example (value INTEGER)")
    connection.execute("INSERT INTO example VALUES (5)")
    source = io.BytesIO(connection.serialize())
    connection.close()
    queried = analytics.read_only_sqlite_query(source, "SELECT value FROM example")
    assert queried["value"].tolist() == [5]
    with np.testing.assert_raises(ValueError):
        analytics.read_only_sqlite_query(source, "DELETE FROM example")


def test_time_series_forecast_has_backtest_and_intervals():
    dates = pd.date_range("2021-01-31", periods=36, freq="ME")
    df = pd.DataFrame({
        "date": dates,
        "value": np.linspace(10, 40, len(dates)) + np.sin(np.arange(len(dates))),
    })
    series = time_series.prepare_series(df, "date", "value", "ME", "mean")
    future, metrics, holdout = time_series.forecast(
        series, periods=4, frequency="ME"
    )
    assert len(future) == 4
    assert metrics["holdout_periods"] >= 2
    assert {"lower_approx_95", "upper_approx_95"}.issubset(future.columns)
    assert len(holdout) == metrics["holdout_periods"]
    with np.testing.assert_raises_regex(ValueError, "three full seasonal cycles"):
        time_series.forecast(series.iloc[:24], periods=4, seasonal_periods=12)
    with np.testing.assert_raises_regex(ValueError, "only by exponential smoothing"):
        time_series.forecast(
            series, periods=4, model_name="ARIMA (1,1,1)", seasonal_periods=12
        )


def test_additional_eda_figures():
    df = pd.DataFrame({
        "lat": [40.7, 34.0, 41.8],
        "lon": [-74.0, -118.2, -87.6],
        "group": ["east", "west", "midwest"],
        "value": [10, 20, 15],
    })
    assert visualization.violin_figure(df, "value", "group").data
    assert visualization.geographic_figure(df, "lat", "lon", "group").data
    assert visualization.pareto_figure(df, "group", "value").data
    forecast = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=2),
        "forecast": [2.0, 3.0],
        "lower_approx_95": [1.0, 2.0],
        "upper_approx_95": [3.0, 4.0],
    })
    series = pd.Series(
        [1.0, 1.5, 2.0],
        index=pd.date_range("2024-10-01", periods=3, freq="ME"),
    )
    holdout = pd.DataFrame({
        "actual": [1.8],
        "holdout_forecast": [1.7],
    }, index=pd.DatetimeIndex(["2024-12-31"]))
    assert visualization.forecast_figure(series, forecast, holdout).data
    predictions = pd.DataFrame({"actual": [1, 2], "predicted": [1.2, 1.8]})
    assert visualization.residual_figure(predictions).data
