import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import os
from sqlalchemy.exc import SQLAlchemyError
from io import BytesIO
from pathlib import Path

from src import (
    advanced_statistics,
    analytics,
    brain_router,
    cleaning,
    ingestion,
    profiling,
    insights,
    reporting,
    time_series,
    visualization,
)
from src.router_rules import select_analyses

st.set_page_config(
    page_title="FlyBrain AutoAnalyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
CHECKPOINT = Path("checkpoints/brain_router.pt")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

st.markdown(
    """
    <style>
    .block-container {max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stHeader"] {background: transparent;}
    [data-testid="stMetric"] {
        background: color-mix(in srgb, var(--secondary-background-color) 72%, transparent);
        border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetricLabel"] p {font-size: .82rem; opacity: .78;}
    [data-testid="stMetricValue"] {font-size: 1.65rem;}
    .stTabs [data-baseweb="tab-list"] {gap: .4rem; border-bottom: 1px solid rgba(128,128,128,.2);}
    .stTabs [data-baseweb="tab"] {padding: .75rem 1rem; border-radius: 10px 10px 0 0;}
    div[data-testid="stPlotlyChart"] {border-radius: 12px; overflow: hidden;}
    .hero-kicker {font-size: .8rem; font-weight: 700; letter-spacing: .12em;
        text-transform: uppercase; color: #4c8bf5; margin-bottom: .35rem;}
    .hero-copy {font-size: 1.06rem; opacity: .78; max-width: 780px; margin-top: -.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-kicker">FLYBRAIN · DATA WORKBENCH</div>', unsafe_allow_html=True)
st.title("FlyBrain AutoAnalyst")
st.markdown(
    '<p class="hero-copy">Explore, clean, visualize, and analyze your data in one workspace. '
    'Your original upload stays unchanged.</p>',
    unsafe_allow_html=True,
)


def render_chart(figure):
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False, "responsive": True, "displayModeBar": "hover"},
    )


@st.cache_resource
def load_brain(path):
    return brain_router.load_checkpoint(path)


brain = load_brain(str(CHECKPOINT)) if CHECKPOINT.is_file() else None
with st.sidebar:
    st.subheader("Workspace")
    st.caption("Upload a dataset to begin. Files are analyzed in this app session.")
    if brain is None:
        st.info("Router: rule-based baseline")
        st.caption("Build a FlyWire checkpoint to enable connectome routing.")
    else:
        st.success("Router: trained connectome")
    st.divider()
    st.markdown("**Supported files**")
    st.caption("CSV · Excel (.xlsx) · JSON · JSONL · Parquet · SQLite")
    st.caption("Maximum upload size: 50 MB")
    st.markdown("**Workflow**")
    st.caption(    "1. Review and clean\n2. Explore and analyze\n3. Model and forecast\n4. Export your results")
    st.divider()
    st.link_button("View source on GitHub", "https://github.com/Satyendra26122006/flybrain-autoanalyst")

f = st.file_uploader(
    "Upload a dataset",
    type=["csv", "xlsx", "json", "jsonl", "parquet", "db", "sqlite", "sqlite3"],
    help="CSV, Excel (.xlsx), JSON, JSON Lines, Parquet, and SQLite databases are supported up to 50 MB.",
)
if f:
    if f.size > MAX_UPLOAD_BYTES:
        st.error("This file exceeds the 50 MB upload limit. Choose a smaller file.")
        st.stop()
    try:
        if Path(f.name).suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            tables = ingestion.sqlite_tables(f)
            if not tables:
                st.error("No tables were found in this SQLite database.")
                st.stop()
            selected_table = st.selectbox("SQLite table to analyze", tables)
            original_df = ingestion.load_sqlite_table(f, selected_table)
        else:
            original_df = ingestion.load(f)
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError, ImportError, sqlite3.Error) as error:
        st.error(f"Could not read this file: {error}")
        st.stop()
    if original_df.empty or original_df.shape[1] == 0:
        st.error("This file contains no data rows or columns to analyze.")
        st.stop()

    source_description = (
        f"{f.name} · table `{selected_table}`"
        if Path(f.name).suffix.lower() in {".db", ".sqlite", ".sqlite3"}
        else f.name
    )
    st.caption(f"Loaded **{source_description}** · {len(original_df):,} rows · {original_df.shape[1]:,} columns")
    overview_tab, clean_tab, explore_tab, insights_tab, ml_tab, forecast_tab, business_tab, sql_text_tab, report_tab = st.tabs(
        ["Overview", "Clean data", "EDA & Visualize", "Insights", "ML Lab", "Forecast", "Business", "SQL & Text", "Report"]
    )

    with clean_tab:
        st.caption("Cleaning is opt-in and applied to a working copy. The uploaded file remains unchanged.")
        left, right = st.columns(2)
        with left:
            trim_strings = st.checkbox("Trim whitespace in text cells")
            normalize_categories = st.checkbox("Normalize case/whitespace category variants")
            convert_numeric = st.checkbox("Convert numeric-looking text columns")
            drop_empty = st.checkbox("Drop columns that are entirely missing")
            drop_duplicates = st.checkbox("Remove duplicate rows")
        with right:
            missing_strategy = st.selectbox(
                "Missing values",
                ["keep", "impute", "drop_rows"],
                format_func={
                    "keep": "Keep as-is",
                    "impute": "Fill numeric median / category mode",
                    "drop_rows": "Drop rows containing missing values",
                }.get,
            )
            outlier_method = st.selectbox(
                "Numeric outliers",
                ["none", "clip_iqr"],
                format_func={
                    "none": "Keep as-is",
                    "clip_iqr": "Cap to 1.5 x IQR bounds",
                }.get,
            )
            columns_to_drop = st.multiselect(
                "Columns to remove (optional)", list(original_df.columns)
            )
        try:
            df, cleaning_log = cleaning.clean_dataframe(
                original_df,
                trim_strings=trim_strings,
                normalize_categories=normalize_categories,
                convert_numeric=convert_numeric,
                drop_empty_columns=drop_empty,
                drop_duplicates=drop_duplicates,
                missing_strategy=missing_strategy,
                outlier_method=outlier_method,
                drop_columns=columns_to_drop,
            )
        except ValueError as error:
            st.error(str(error))
            df, cleaning_log = original_df.copy(), []

        if cleaning_log:
            st.success(
                f"Cleaning changed the working data: {len(original_df):,} → {len(df):,} rows, "
                f"{original_df.shape[1]:,} → {df.shape[1]:,} columns."
            )
            st.dataframe(pd.DataFrame(cleaning_log), hide_index=True, use_container_width=True)
        else:
            st.info("No data changes from the selected cleaning options.")
        st.subheader("Cleaned preview")
        st.dataframe(df.head(500), use_container_width=True, height=420)
        cleaned_name = f"{Path(f.name).stem}_cleaned.csv"
        st.download_button(
            "Download cleaned CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=cleaned_name,
            mime="text/csv",
            key="download_cleaned",
        )

    p = profiling.profile(df)

    with overview_tab:
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Rows", f"{p['n_rows']:,}")
        metric2.metric("Columns", f"{p['n_cols']:,}")
        metric3.metric("Missing cells", f"{p['missing_rate']:.1%}")
        metric4.metric("Duplicate rows", f"{df.duplicated().sum():,}")
        st.caption(f"Approximate in-memory size: {df.memory_usage(index=True, deep=True).sum() / (1024 ** 2):,.1f} MB")
        st.subheader("Data preview")
        st.dataframe(df.head(500), use_container_width=True, height=420)
        st.download_button(
            "Download original uploaded data as CSV",
            original_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(f.name).stem}.csv",
            mime="text/csv",
            key="download_original",
        )

        st.subheader("Column profile")
        column_profile = pd.DataFrame({
            "column": df.columns,
            "type": [str(dtype) for dtype in df.dtypes],
            "missing": df.isna().sum().values,
            "missing_rate": df.isna().mean().values,
            "unique_values": [df[c].nunique(dropna=True) for c in df.columns],
        })
        st.dataframe(column_profile, hide_index=True, use_container_width=True)
        key_columns = st.multiselect(
            "Optional key columns for duplicate-key detection",
            list(df.columns),
            help="Select one or more columns that should uniquely identify a record.",
            key="duplicate_key_columns",
        )
        duplicate_keys = (
            int(df.duplicated(subset=key_columns).sum()) if key_columns else None
        )
        if key_columns:
            st.caption(
                f"Repeated key rows: **{duplicate_keys:,}** "
                f"using {', '.join(map(str, key_columns))}. "
                "Rows are only flagged; none are removed automatically."
            )
        st.subheader("Descriptive statistics")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
        with st.expander("Full dataset profile"):
            st.json({k: v for k, v in p.items() if not isinstance(v, list)})

    with explore_tab:
        st.caption("Exploratory charts use the cleaned working data; downloads preserve the original separately.")
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        datetime_columns = df.select_dtypes(include="datetime").columns.tolist()

        missing_figure = visualization.missingness_figure(df)
        if missing_figure is not None:
            st.subheader("Missingness")
            render_chart(missing_figure)
        else:
            st.success("No missing values in the cleaned working data.")

        if numeric_columns:
            st.subheader("Outlier overview")
            outlier_figure = visualization.boxplot_figure(df, numeric_columns)
            if outlier_figure is not None:
                render_chart(outlier_figure)

            st.subheader("Numeric distribution")
            dist_col, bins_col = st.columns([3, 1])
            with dist_col:
                dist_column = st.selectbox("Column", numeric_columns, key="distribution_column")
            with bins_col:
                bins = st.slider("Histogram bins", min_value=5, max_value=100, value=30)
            render_chart(visualization.distribution_figure(df, dist_column, bins))
            with st.expander("Density / violin view"):
                violin_group = st.selectbox(
                    "Optional group", ["(none)", *categorical_columns], key="violin_group"
                )
                render_chart(visualization.violin_figure(
                    df,
                    dist_column,
                    None if violin_group == "(none)" else violin_group,
                ))

            st.subheader("Scatter plot")
            x_col, y_col, color_col = st.columns(3)
            with x_col:
                scatter_x = st.selectbox("X axis", numeric_columns, key="scatter_x")
            with y_col:
                scatter_y = st.selectbox(
                    "Y axis", numeric_columns,
                    index=1 if len(numeric_columns) > 1 else 0,
                    key="scatter_y",
                )
            with color_col:
                color_options = ["(no grouping)", *categorical_columns]
                scatter_color = st.selectbox("Color by", color_options, key="scatter_color")
            render_chart(
                visualization.scatter_figure(
                    df, scatter_x, scatter_y,
                    None if scatter_color == "(no grouping)" else scatter_color,
                )
            )
            if len(df) > 5000:
                st.caption("Scatter plot uses a reproducible 5,000-row sample for responsiveness.")

            corr_figure = visualization.correlation_figure(df, numeric_columns)
            if corr_figure is not None:
                st.subheader("Numeric correlations")
                render_chart(corr_figure)
            else:
                st.info("Correlation heatmap needs at least two varying numeric columns.")

            if len(numeric_columns) >= 2:
                st.subheader("Pairwise feature exploration")
                pair_columns = st.multiselect(
                    "Choose up to 5 numeric columns",
                    numeric_columns,
                    default=numeric_columns[:min(4, len(numeric_columns))],
                    max_selections=5,
                    key="pair_columns",
                )
                pair_figure = visualization.pairplot_figure(df, pair_columns)
                if pair_figure is not None:
                    render_chart(pair_figure)
                network_figure = visualization.correlation_network_figure(
                    df, numeric_columns
                )
                if network_figure is not None:
                    with st.expander("Strong-association network"):
                        render_chart(network_figure)

            if numeric_columns:
                with st.expander("Numeric distribution profile"):
                    numeric_profile = df[numeric_columns].agg(
                        ["count", "mean", "median", "std", "var", "min", "max", "skew", "kurt"]
                    ).transpose()
                    st.dataframe(numeric_profile, use_container_width=True)
        else:
            st.info("No numeric columns are available for numeric visualizations.")

        if categorical_columns:
            st.subheader("Category counts")
            category_column = st.selectbox(
                "Category column", categorical_columns, key="category_column"
            )
            render_chart(visualization.category_figure(df, category_column))
            if numeric_columns:
                grouped_value = st.selectbox(
                    "Compare numeric measure across categories",
                    numeric_columns,
                    key="grouped_value",
                )
                grouped_figure = visualization.grouped_boxplot_figure(
                    df, category_column, grouped_value
                )
                if grouped_figure is not None:
                    render_chart(grouped_figure)

            if numeric_columns:
                with st.expander("Group summary and pivot table"):
                    aggregation = st.selectbox(
                        "Aggregation",
                        ["mean", "median", "sum", "count"],
                        key="group_aggregation",
                    )
                    grouped = df.groupby(category_column, dropna=False)[grouped_value]
                    summary = grouped.agg(aggregation).sort_values(ascending=False)
                    st.dataframe(
                        summary.rename(f"{grouped_value}_{aggregation}").to_frame(),
                        use_container_width=True,
                    )
                    if len(categorical_columns) >= 2:
                        second_category = st.selectbox(
                            "Cross-tabulate with",
                            [column for column in categorical_columns if column != category_column],
                            key="second_category",
                        )
                        st.dataframe(
                            pd.crosstab(df[category_column], df[second_category], margins=True),
                            use_container_width=True,
                        )

        if datetime_columns and numeric_columns:
            st.subheader("Values over time")
            time_x, time_y = st.columns(2)
            with time_x:
                time_column = st.selectbox("Date/time column", datetime_columns)
            with time_y:
                time_value = st.selectbox("Measure", numeric_columns)
            render_chart(visualization.time_series_figure(df, time_column, time_value))
        elif datetime_columns:
            st.info("Datetime columns found, but a numeric measure is needed for a time-series chart.")

        latitude_options = [
            column for column in numeric_columns
            if str(column).lower() in {"lat", "latitude"}
        ]
        longitude_options = [
            column for column in numeric_columns
            if str(column).lower() in {"lon", "lng", "longitude"}
        ]
        if latitude_options and longitude_options:
            with st.expander("Geographic view"):
                geo_lat, geo_lon, geo_color = st.columns(3)
                with geo_lat:
                    latitude_column = st.selectbox("Latitude", latitude_options)
                with geo_lon:
                    longitude_column = st.selectbox("Longitude", longitude_options)
                with geo_color:
                    color_column = st.selectbox(
                        "Color by", ["(none)", *categorical_columns], key="geo_color"
                    )
                render_chart(visualization.geographic_figure(
                    df,
                    latitude_column,
                    longitude_column,
                    None if color_column == "(none)" else color_column,
                ))

    dataset_signature = f"{getattr(f, 'file_id', f.name)}_{len(cleaning_log)}"
    model_state_key = f"analysis_model_{dataset_signature}"
    forecast_state_key = f"analysis_forecast_{dataset_signature}"
    cluster_state_key = f"analysis_clusters_{dataset_signature}"
    anomaly_state_key = f"analysis_anomalies_{dataset_signature}"
    cohort_state_key = f"analysis_cohorts_{dataset_signature}"
    basket_state_key = f"analysis_basket_{dataset_signature}"
    sql_state_key = f"analysis_sql_{dataset_signature}"
    resampling_state_key = f"analysis_resampling_{dataset_signature}"

    with ml_tab:
        st.subheader("Supervised prediction")
        st.caption(
            "Choose an outcome and features. Evaluation uses a held-out split plus cross-validation "
            "when there is enough data; results are predictive associations, not causal effects."
        )
        target_column = st.selectbox("Target / outcome", list(df.columns), key="ml_target")
        task_type = st.selectbox("Prediction task", ["classification", "regression"])
        feature_options = [column for column in df.columns if column != target_column]
        default_features = [
            column for column in feature_options
            if not pd.api.types.is_datetime64_any_dtype(df[column])
            and df[column].nunique(dropna=True) < max(100, len(df) * 0.5)
        ][:20]
        feature_columns = st.multiselect(
            "Input features",
            feature_options,
            default=default_features,
            max_selections=analytics.MAX_FEATURES,
            help="High-cardinality and identifier columns can cause leakage; exclude them unless justified.",
        )
        estimator_options = (
            list(analytics.CLASSIFIERS) if task_type == "classification"
            else list(analytics.REGRESSORS)
        )
        estimator_name = st.selectbox("Estimator", estimator_options)
        test_fraction = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
        if st.button("Train and evaluate", type="primary", key="run_supervised"):
            try:
                model_result = analytics.fit_supervised(
                    df,
                    target=target_column,
                    features=feature_columns,
                    task=task_type,
                    model_name=estimator_name,
                    test_fraction=test_fraction,
                )
                st.session_state[model_state_key] = model_result
            except (ValueError, TypeError) as error:
                st.error(f"Model evaluation could not run: {error}")
        model_result = st.session_state.get(model_state_key)
        if model_result is not None:
            st.json(model_result.metrics)
            for warning in model_result.warnings:
                st.warning(warning)
            if not model_result.importance.empty:
                st.subheader("Permutation feature importance")
                st.dataframe(model_result.importance, hide_index=True, use_container_width=True)
            st.subheader("Held-out predictions")
            st.dataframe(model_result.predictions.head(1000), use_container_width=True)
            st.download_button(
                "Download held-out predictions",
                model_result.predictions.to_csv(index=True).encode("utf-8"),
                file_name="held_out_predictions.csv",
                mime="text/csv",
            )
            if model_result.task == "regression":
                render_chart(visualization.residual_figure(model_result.predictions))

        st.divider()
        st.subheader("Unsupervised learning")
        cluster_features = st.multiselect(
            "Numeric clustering features",
            df.select_dtypes(include="number").columns.tolist(),
            max_selections=analytics.MAX_FEATURES,
            key="cluster_features",
        )
        cluster_algorithm = st.selectbox("Clustering algorithm", ["K-means", "DBSCAN"])
        cluster_count = st.slider("K-means clusters", 2, 12, 3)
        if st.button("Run clustering and PCA", key="run_clustering"):
            try:
                cluster_result, cluster_summary = analytics.cluster_and_reduce(
                    df,
                    cluster_features,
                    algorithm=cluster_algorithm,
                    clusters=cluster_count,
                )
                st.session_state[cluster_state_key] = (cluster_result, cluster_summary)
            except ValueError as error:
                st.error(f"Clustering could not run: {error}")
        if cluster_state_key in st.session_state:
            cluster_result, cluster_summary = st.session_state[cluster_state_key]
            st.json(cluster_summary)
            render_chart(visualization.cluster_figure(cluster_result))
            st.download_button(
                "Download cluster assignments",
                cluster_result.to_csv(index=False).encode("utf-8"),
                file_name="cluster_assignments.csv",
                mime="text/csv",
            )

        st.subheader("Isolation Forest anomaly review")
        anomaly_features = st.multiselect(
            "Numeric anomaly features",
            df.select_dtypes(include="number").columns.tolist(),
            key="anomaly_features",
        )
        contamination = st.slider("Expected anomaly fraction", 0.01, 0.2, 0.03, 0.01)
        if st.button("Score anomalies", key="run_isolation_forest"):
            try:
                st.session_state[anomaly_state_key] = analytics.isolation_anomalies(
                    df, anomaly_features, contamination
                )
            except ValueError as error:
                st.error(f"Anomaly scoring could not run: {error}")
        if anomaly_state_key in st.session_state:
            anomalies = st.session_state[anomaly_state_key]
            st.dataframe(anomalies.head(100), hide_index=True, use_container_width=True)
            st.caption("Anomaly scores prioritize review; they do not establish fraud.")

    with forecast_tab:
        st.subheader("Forecast a time series")
        if not datetime_columns or not numeric_columns:
            st.info("Forecasting requires at least one datetime and one numeric column.")
        else:
            forecast_date, forecast_value = st.columns(2)
            with forecast_date:
                forecast_date_column = st.selectbox("Date column", datetime_columns, key="forecast_date")
            with forecast_value:
                forecast_value_column = st.selectbox("Measure", numeric_columns, key="forecast_value")
            forecast_frequency = st.selectbox(
                "Aggregation frequency",
                ["D", "W", "ME", "QE", "YE"],
                format_func={"D": "Daily", "W": "Weekly", "ME": "Monthly", "QE": "Quarterly", "YE": "Yearly"}.get,
            )
            forecast_aggregation = st.selectbox("Aggregate values using", ["sum", "mean", "median"])
            forecast_model_name = st.selectbox(
                "Forecast model",
                ["Exponential smoothing", "ARIMA (1,1,1)"],
            )
            horizon, season = st.columns(2)
            with horizon:
                forecast_horizon = st.number_input("Periods to forecast", 1, 120, 6)
            with season:
                if forecast_model_name == "Exponential smoothing":
                    seasonal_periods_input = st.number_input(
                        "Season length (0 = none)", min_value=0, max_value=365, value=0,
                        help="For monthly yearly seasonality use 12; for weekly yearly seasonality use 52.",
                    )
                else:
                    seasonal_periods_input = 0
                    st.caption("Seasonal terms are not enabled for this ARIMA model.")
            if st.button("Run forecast and backtest", type="primary", key="run_forecast"):
                try:
                    series = time_series.prepare_series(
                        df, forecast_date_column, forecast_value_column,
                        forecast_frequency, forecast_aggregation,
                    )
                    forecast_frame, forecast_metrics, holdout = time_series.forecast(
                        series,
                        periods=int(forecast_horizon),
                        model_name=forecast_model_name,
                        seasonal_periods=int(seasonal_periods_input) or None,
                        frequency=forecast_frequency,
                    )
                    st.session_state[forecast_state_key] = (
                        series, forecast_frame, forecast_metrics, holdout
                    )
                except (ValueError, TypeError) as error:
                    st.error(f"Forecast could not run: {error}")
            forecast_result = st.session_state.get(forecast_state_key)
            if forecast_result is not None:
                series, forecast_frame, forecast_metrics, holdout = forecast_result
                st.json(forecast_metrics)
                if forecast_metrics.get("adf_p_value") is not None:
                    st.caption(f"Stationarity check: {forecast_metrics['stationarity_note']}")
                render_chart(visualization.forecast_figure(series, forecast_frame, holdout))
                st.dataframe(forecast_frame, hide_index=True, use_container_width=True)
                st.download_button(
                    "Download forecast",
                    forecast_frame.to_csv(index=False).encode("utf-8"),
                    file_name="time_series_forecast.csv",
                    mime="text/csv",
                )

    with business_tab:
        st.subheader("Business metrics")
        st.caption("Map your own columns to metrics; no sales, cost, or churn semantics are guessed.")
        revenue_options = [None, *numeric_columns]
        revenue_column = st.selectbox("Revenue / value column", revenue_options, key="business_revenue")
        cost_column = st.selectbox("Cost column (optional)", revenue_options, key="business_cost")
        target_column = st.selectbox("Target column (optional)", revenue_options, key="business_target")
        date_options = [None, *datetime_columns]
        business_date = st.selectbox("Date column (optional)", date_options, key="business_date")
        category_options = [None, *categorical_columns]
        business_category = st.selectbox("Product / segment (optional)", category_options, key="business_category")
        kpi_summary, category_summary = analytics.summarize_business_metrics(
            df,
            revenue=revenue_column,
            cost=cost_column,
            date=business_date,
            category=business_category,
            target=target_column,
        )
        if kpi_summary.empty:
            st.info("Select a numeric revenue/value column to calculate business metrics.")
        else:
            st.dataframe(kpi_summary.transpose().rename(columns={0: "value"}), use_container_width=True)
            if not category_summary.empty:
                st.subheader("Category / product performance")
                st.dataframe(category_summary, use_container_width=True)
                if revenue_column:
                    render_chart(visualization.pareto_figure(df, business_category, revenue_column))
            if business_date and revenue_column:
                period_series = category_summary.attrs.get("period_revenue")
                if period_series is not None:
                    render_chart(visualization.time_series_figure(
                        period_series.rename_axis(business_date).reset_index(),
                        business_date,
                        "revenue",
                    ))

        st.subheader("Customer cohort retention")
        if categorical_columns and datetime_columns:
            customer_column = st.selectbox("Customer identifier", df.columns.tolist(), key="cohort_customer")
            cohort_date = st.selectbox("Cohort activity date", datetime_columns, key="cohort_date")
            if st.button("Calculate cohort retention", key="run_cohort"):
                try:
                    st.session_state[cohort_state_key] = analytics.cohort_retention(
                        df, customer_column, cohort_date
                    )
                except ValueError as error:
                    st.error(f"Cohort analysis could not run: {error}")
            if cohort_state_key in st.session_state:
                st.dataframe(
                    st.session_state[cohort_state_key].style.format("{:.0%}"),
                    use_container_width=True,
                )
        else:
            st.caption("Cohort analysis requires a customer identifier and date column.")

        st.subheader("Market basket pairs")
        basket_column = st.selectbox(
            "Transaction items column (comma-separated items per row)",
            [None, *df.select_dtypes(include=["object", "string"]).columns.tolist()],
            key="basket_column",
        )
        basket_support = st.slider("Minimum item-pair support", 0.01, 0.5, 0.05, 0.01)
        if basket_column and st.button("Find item associations", key="run_basket"):
            try:
                st.session_state[basket_state_key] = analytics.market_basket_pairs(
                    df[basket_column], minimum_support=basket_support
                )
            except ValueError as error:
                st.error(f"Basket analysis could not run: {error}")
        if basket_state_key in st.session_state:
            st.dataframe(st.session_state[basket_state_key], hide_index=True, use_container_width=True)

    with insights_tab:
        tasks = brain_router.select_analyses(brain, p) if brain is not None else select_analyses(p)
        st.caption(
            f"Router: {'trained connectome' if brain is not None else 'rule-based baseline'}"
            f" · Core tools: {', '.join(tasks) if tasks else 'none selected'}"
        )
        st.markdown("**Additional statistical tests**")
        add_on_labels = {
            "Spearman correlation": "spearman",
            "Kendall correlation": "kendall",
            "Group comparisons (Welch / nonparametric)": "group_comparison",
            "Categorical association (Fisher / chi-square)": "categorical",
            "Normality checks (Shapiro-Wilk)": "normality",
        }
        selected_add_ons = st.multiselect(
            "Choose additional tests",
            list(add_on_labels),
            default=list(add_on_labels),
            help="P-values are adjusted with Benjamini-Hochberg within each test family.",
        )
        group_method = st.selectbox(
            "Group comparison method",
            ["Welch / ANOVA", "Mann-Whitney / Kruskal-Wallis"],
            help="The nonparametric option uses Mann-Whitney for 2 groups and Kruskal-Wallis for 3+.",
        )
        numeric_columns = p["numeric_columns"]
        categorical_columns = p["categorical_columns"]
        selected_tools = [add_on_labels[label] for label in selected_add_ons]
        core_tasks = [task for task in tasks if not (
            task == "group_comparison" and "group_comparison" in selected_tools
        )]
        findings = insights.run(df, p, core_tasks)
        if "spearman" in [add_on_labels[label] for label in selected_add_ons]:
            findings += advanced_statistics.correlations(df, numeric_columns, "Spearman")
        if "kendall" in [add_on_labels[label] for label in selected_add_ons]:
            findings += advanced_statistics.correlations(df, numeric_columns, "Kendall")
        if "group_comparison" in selected_tools:
            findings += advanced_statistics.group_comparisons(
                df,
                categorical_columns,
                numeric_columns,
                group_method,
            )
        if "categorical" in [add_on_labels[label] for label in selected_add_ons]:
            findings += advanced_statistics.categorical_associations(df, categorical_columns)
        normality_results = (
            advanced_statistics.normality_checks(df, numeric_columns)
            if "normality" in selected_tools else pd.DataFrame()
        )
        quality = insights.check(df)
        if numeric_columns:
            with st.expander("Custom numeric range validation"):
                validation_column = st.selectbox(
                    "Column to validate", numeric_columns, key="validation_column"
                )
                lower_enabled, upper_enabled = st.columns(2)
                with lower_enabled:
                    has_lower_bound = st.checkbox("Set minimum", key="has_lower_bound")
                    lower_bound = st.number_input(
                        "Minimum allowed",
                        value=0.0,
                        key="lower_bound",
                        disabled=not has_lower_bound,
                    )
                with upper_enabled:
                    has_upper_bound = st.checkbox("Set maximum", key="has_upper_bound")
                    upper_bound = st.number_input(
                        "Maximum allowed",
                        value=0.0,
                        key="upper_bound",
                        disabled=not has_upper_bound,
                    )
                try:
                    range_issue = cleaning.validate_numeric_range(
                        df,
                        validation_column,
                        minimum=lower_bound if has_lower_bound else None,
                        maximum=upper_bound if has_upper_bound else None,
                    )
                    if range_issue:
                        quality.append(range_issue)
                        st.warning(range_issue["detail"])
                    elif has_lower_bound or has_upper_bound:
                        st.success("No values violate the configured range.")
                    else:
                        st.caption("Select at least one bound to run the check.")
                except ValueError as error:
                    st.error(f"Range validation failed: {error}")
        if key_columns and duplicate_keys:
            quality.append({
                "issue": "duplicate_key",
                "detail": (
                    f"{duplicate_keys} repeated key rows across "
                    f"{', '.join(map(str, key_columns))}"
                ),
                "severity": "medium",
            })
        out = {
            "quality": quality,
            "findings": findings,
            "normality": normality_results,
        }

        quality_col, findings_col, tools_col = st.columns(3)
        quality_col.metric("Quality flags", len(out["quality"]))
        findings_col.metric("Statistical findings", len(out["findings"]))
        tools_col.metric("Analysis methods", len(core_tasks) + len(selected_tools))
        st.subheader("Data quality")
        if out["quality"]:
            st.dataframe(pd.DataFrame(out["quality"]), hide_index=True, use_container_width=True)
        else:
            st.success("No data-quality issues were flagged.")
        st.subheader("Findings")
        for index, x in enumerate(out["findings"], start=1):
            with st.expander(f"{index}. {x.claim} · {x.tool}"):
                st.write("**Evidence:**", x.evidence); st.write("**Uncertainty:**", x.uncertainty)
                st.write("**Limitations:**", x.limitations); st.write("**Next step:**", x.next_step)
        if not normality_results.empty:
            st.subheader("Normality checks")
            st.caption(
                "Shapiro-Wilk tests use Benjamini-Hochberg-adjusted p-values; "
                "failure to reject normality does not prove a normal distribution."
            )
            st.dataframe(normality_results, hide_index=True, use_container_width=True)
        if not out["findings"]:
            st.info("No statistically solid findings.")
        if categorical_columns and numeric_columns:
            with st.expander("Two-group bootstrap and permutation test"):
                test_category = st.selectbox(
                    "Grouping column", categorical_columns, key="resampling_category"
                )
                test_groups = df[test_category].dropna().unique().tolist()
                if len(test_groups) >= 2:
                    first_group, second_group = st.columns(2)
                    with first_group:
                        first_value = st.selectbox(
                            "First group", test_groups, key="resampling_first_group"
                        )
                    with second_group:
                        second_options = [value for value in test_groups if value != first_value]
                        second_value = st.selectbox(
                            "Second group", second_options, key="resampling_second_group"
                        )
                    test_measure = st.selectbox(
                        "Numeric outcome", numeric_columns, key="resampling_measure"
                    )
                    iterations = st.select_slider(
                        "Resampling iterations", [500, 1000, 2000, 5000], value=2000
                    )
                    if st.button("Run two-group tests", key="run_resampling"):
                        first_data = df.loc[df[test_category] == first_value, test_measure]
                        second_data = df.loc[df[test_category] == second_value, test_measure]
                        try:
                            st.session_state[resampling_state_key] = {
                                "bootstrap": analytics.bootstrap_mean_difference(
                                    first_data, second_data, iterations=iterations
                                ),
                                "permutation": analytics.permutation_mean_difference(
                                    first_data, second_data, iterations=iterations
                                ),
                            }
                        except ValueError as error:
                            st.error(f"Resampling tests could not run: {error}")
                    if resampling_state_key in st.session_state:
                        st.json(st.session_state[resampling_state_key])
                        st.caption(
                            "The bootstrap percentile interval and permutation p-value compare means; "
                            "validity depends on independent observations and the sampling design."
                        )
                else:
                    st.caption("At least two groups are required.")

    with sql_text_tab:
        st.subheader("Read-only SQL")
        remote_database_url = os.environ.get("DATABASE_URL")
        if remote_database_url:
            st.caption(
                "Remote database URL is read from the DATABASE_URL environment variable. "
                "Use a dedicated read-only account."
            )
            try:
                remote_tables = analytics.remote_database_tables(remote_database_url)
                if remote_tables:
                    st.write("Available remote tables:", ", ".join(remote_tables[:100]))
                remote_query = st.text_area(
                    "Remote SQL SELECT / CTE",
                    value="SELECT 1",
                    key="remote_sql_query",
                    height=120,
                    help="One read-only query per run; use least-privilege database credentials.",
                )
                if st.button("Run remote read-only query", key="run_remote_sql"):
                    try:
                        st.session_state["remote_sql_result"] = analytics.read_only_remote_query(
                            remote_database_url,
                            remote_query,
                            limit=analytics.MAX_MODEL_ROWS,
                        )
                    except (ValueError, SQLAlchemyError) as error:
                        st.error(f"Remote SQL failed: {error}")
                if "remote_sql_result" in st.session_state:
                    st.dataframe(
                        st.session_state["remote_sql_result"].head(1000),
                        use_container_width=True,
                    )
            except (ValueError, SQLAlchemyError) as error:
                st.error(f"Remote database connection failed: {error}")
        else:
            st.caption(
                "Optional PostgreSQL/MySQL connection: set DATABASE_URL in the environment. "
                "Use a read-only database account; never put credentials in uploaded files."
            )
        if Path(f.name).suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            st.caption(
                f"Queries run only against the uploaded SQLite file, in memory, with query-only mode "
                f"and a {analytics.MAX_MODEL_ROWS:,}-row result limit."
            )
            default_query = f'SELECT * FROM "{selected_table}" LIMIT 100'
            sql_query = st.text_area(
                "SQL SELECT or WITH query",
                value=default_query,
                height=140,
                help="Only one read-only SELECT/CTE query is allowed; writes and schema changes are blocked.",
            )
            if st.button("Run read-only query", key="run_sql"):
                try:
                    query_result = analytics.read_only_sqlite_query(
                        f, sql_query, limit=analytics.MAX_MODEL_ROWS
                    )
                    st.session_state[sql_state_key] = query_result
                except ValueError as error:
                    st.error(str(error))
            if sql_state_key in st.session_state:
                st.dataframe(st.session_state[sql_state_key].head(1000), use_container_width=True)
                st.download_button(
                    "Download query result",
                    st.session_state[sql_state_key].to_csv(index=False).encode("utf-8"),
                    file_name="sql_query_result.csv",
                    mime="text/csv",
                )
        else:
            st.info("Upload a SQLite database file to run read-only SQL against its tables.")

        st.divider()
        st.subheader("Basic text exploration")
        text_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if text_columns:
            text_column = st.selectbox("Text column", text_columns, key="text_column")
            terms = analytics.text_term_frequency(df[text_column])
            if not terms.empty:
                st.dataframe(terms, hide_index=True, use_container_width=True)
                render_chart(
                    px.bar(
                        terms.head(20),
                        x="frequency",
                        y="term",
                        orientation="h",
                        title=f"Frequent terms in {text_column}",
                    )
                )
            else:
                st.info("No terms were found in this column.")
            st.caption(
                "This is frequency analysis, not validated sentiment analysis. Text is treated as data, "
                "never executed as instructions."
            )
        else:
            st.info("No text columns are available for word-frequency analysis.")

    with report_tab:
        st.subheader("Download your analysis")
        st.caption(
            "The PDF includes dataset metrics, data-quality flags, a cleaning audit trail, "
            "summary charts, findings, and interpretation cautions."
        )
        report_col, data_col = st.columns(2)
        analysis_methods = list(dict.fromkeys(
            [*core_tasks, *[label for label in selected_add_ons]]
        ))
        resampling_result = st.session_state.get(resampling_state_key)
        if resampling_result:
            analysis_methods.extend(["two-group bootstrap", "permutation test"])
        model_result = st.session_state.get(model_state_key)
        forecast_result = st.session_state.get(forecast_state_key)
        report_tables = {"Normality checks (Shapiro-Wilk)": normality_results}
        if resampling_result:
            report_tables["Bootstrap and permutation tests"] = pd.DataFrame([
                {"method": name, **values}
                for name, values in resampling_result.items()
            ])
        if model_result is not None:
            report_tables["Predictive model metrics"] = pd.DataFrame([
                {"metric": name, "value": value}
                for name, value in model_result.metrics.items()
                if isinstance(value, (int, float, str))
            ])
        if forecast_result is not None:
            report_tables["Forecast backtest metrics"] = pd.DataFrame([
                {"metric": name, "value": value}
                for name, value in forecast_result[2].items()
                if isinstance(value, (int, float, str))
            ])
        with report_col:
            pdf = reporting.to_pdf(
                source_description,
                p,
                out["quality"],
                out["findings"],
                df=df,
                cleaning_log=cleaning_log,
                analysis_tables=report_tables,
                analysis_methods=analysis_methods,
            )
            st.download_button(
                "Download PDF report",
                pdf,
                file_name=f"{Path(f.name).stem}_analysis_report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
            html_report = reporting.to_html(source_description, p, out["quality"], out["findings"])
            st.download_button(
                "Download HTML report",
                html_report,
                file_name=f"{Path(f.name).stem}_analysis_report.html",
                mime="text/html",
                use_container_width=True,
            )
        with data_col:
            st.download_button(
                "Download cleaned data (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"{Path(f.name).stem}_cleaned.csv",
                mime="text/csv",
                use_container_width=True,
            )
            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False)
            st.download_button(
                "Download cleaned data (Excel)",
                excel_buffer.getvalue(),
                file_name=f"{Path(f.name).stem}_cleaned.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
else:
    st.write("")
    with st.container(border=True):
        st.subheader("Start with a dataset")
        st.write(
            "Upload a CSV, Excel, JSON, Parquet, or SQLite file above to get an instant profile, "
            "cleaning preview, interactive EDA, statistical checks, and downloadable results."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**01 · Understand**")
            st.caption("Review row counts, column types, missing values, and summary statistics.")
        with col2:
            st.markdown("**02 · Prepare**")
            st.caption("Apply optional cleaning to a working copy and inspect every change.")
        with col3:
            st.markdown("**03 · Explore**")
            st.caption("Use charts and statistical findings to investigate patterns in your data.")
