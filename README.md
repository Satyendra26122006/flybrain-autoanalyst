# AutoAnalyst

Copyright (c) 2026 Satyendra26122006. All rights reserved. No permission is granted to use, copy, modify, or
distribute this project without prior written permission. See [LICENSE](./LICENSE).

An interactive data-analysis workbench for loading, cleaning, exploring, modeling, and reporting on datasets.
Transparent data-profile rules suggest core checks, and Python tools perform the calculations and attach evidence,
uncertainty, and caveats.

## Run
```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
streamlit run app.py
```

## Analysis, cleaning, and EDA
The app accepts CSV, Excel, JSON, JSONL, Parquet, and SQLite uploads up to 50 MB. It reports dimensions,
column types, unique values, missingness, duplicate rows/selected keys, approximate memory, and descriptive
statistics. Cleaning is opt-in, previewed, and audited: trim and normalize category text, convert numeric-looking
text, drop empty columns/duplicates/selected columns, impute or drop missing rows, and cap numeric outliers.
Custom numeric range checks flag invalid values without silently changing data.

EDA includes distribution profiles, histograms, violin plots, box plots, scatter plots, correlation heatmaps and
networks, geographic maps where latitude/longitude columns are present, category counts, group summaries,
pivots, crosstabs, and time-series charts. The app has selectable Shapiro-Wilk normality, Pearson/Spearman/Kendall
correlations, Welch t-tests/ANOVA or Mann-Whitney/Kruskal-Wallis, Fisher/chi-square categorical tests,
bootstrap intervals, permutation tests, and Benjamini-Hochberg corrections. It also includes supervised
classification/regression with held-out evaluation, cross-validation and permutation importance; K-means/DBSCAN,
PCA and Isolation Forest; Exponential Smoothing/ARIMA forecasting with a chronological holdout and stationarity
diagnostic; mapped revenue/cost/profit/margin/target KPIs; customer cohort retention; Pareto summaries; and
market-basket item-pair support, confidence and lift. All computations run in Python and the app labels
assumptions and limitations.

Uploaded SQLite files support table selection and read-only SQL. Optional PostgreSQL/MySQL connections use
`DATABASE_URL` from the process environment, not a UI credential field. Use a dedicated read-only database user.
Only one SELECT/CTE query is accepted, and result rows are capped. Do not use administrator credentials.

Reports include downloadable PDF and HTML, cleaning/quality summaries, analysis methods, and findings; cleaned
data exports to CSV and Excel. Forecast and predictive outputs are evaluations, not guarantees. Correlation does
not imply causation; the app does not infer business definitions or make unsupported conclusions.

### Important scope and safety limits
Core analysis suggestions use transparent rules based on the dataset profile. Advanced options are explicit
deterministic Python functions invoked from the UI. There is no generic LLM that executes instructions found in
uploaded data. Large-file processing is currently in-memory, with a 50 MB upload cap and bounded samples for
selected models/charts; this is not an incremental streaming ETL system. The current version does not implement
survival analysis, generalized causal inference, production-grade fraud detection, calibrated seasonal prediction
intervals, rolling-origin hyperparameter search, arbitrary geospatial operations, a full sentiment model,
production ETL/lineage orchestration, or Monte Carlo optimization. Use domain experts for validation and review
outputs before acting on them. Uploaded cell contents are treated as data, not as instructions to execute.

## Layout
`src/` ingestion, profiling, quality, statistics, validation, insights, transparent analysis-selection rules,
cleaning, visualization, modeling, forecasting, and reporting.
