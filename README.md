# FlyBrain AutoAnalyst

Automated data analyst. A fully trainable fly-brain connectome network (every neuron and synapse learnable, wiring fixed)
chooses which analyses to run; exact Python code runs them; a validator attaches evidence, uncertainty and caveats.

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
The connectome currently routes only its four original analysis classes; it does not itself learn or select every
newly integrated tool. Advanced options are explicit deterministic Python functions invoked from the UI. There is
no generic LLM that executes instructions found in uploaded data. Large-file processing is currently in-memory,
with a 50 MB upload cap and bounded samples for selected models/charts; this is not an incremental streaming ETL
system. The current version does not implement survival analysis, generalized causal inference, production-grade
fraud detection, calibrated seasonal prediction intervals, rolling-origin hyperparameter search, arbitrary
geospatial operations, a full sentiment model, production ETL/lineage orchestration, or Monte Carlo optimization.
Use domain experts for validation and review outputs before acting on them. Uploaded cell contents are treated
as data, not as instructions to execute.

## Brain
The connectome routes datasets to the existing, exact Python analysis tools; it does not replace their computations.
To train it on the full FlyWire graph:

1. Download the FlyWire Codex connections CSV/CSV.GZ and put it at `data/raw/connections.csv.gz`. Record the release
   and citation. The dataset is not included in this repository.
2. Train and save the full-graph router:
   `python train_brain.py`
   This keeps edges with at least one synapse by default. Use `--connections`, `--min-syn`, `--epochs`, and
   `--device` to override defaults. For a GPU, install the matching PyTorch build before training; a full graph can be
   memory- and compute-intensive, especially on CPU.
3. Start the app with `streamlit run app.py`. It loads `checkpoints/brain_router.pt` automatically. If no checkpoint
   exists, it clearly reports that it is using the rule-based baseline instead.

To build tensors separately, run:
`python -c "from src.connectome import build_tensors as b; b('data/raw/connections.csv.gz', min_syn=1, out_path='data/processed/tensors.pt')"`

Compare routers (no FlyWire data needed for the small synthetic-graph dry run):
`python experiment.py` or `python experiment.py data/processed/tensors.pt`

`experiment.py` trains the real connectome and a degree-preserving shuffled control on the same synthetic task.

## Layout
`src/` ingestion, profiling, quality, statistics (tools), validation, insights, router_rules (A), brain_router (C), connectome, synthetic, reporting.

## Hosting
Deploy the public repository on [Streamlit Community Cloud](https://share.streamlit.io/):

1. Sign in with the GitHub account that can access this repository.
2. Choose **Create app**, then select `Satyendra26122006/flybrain-autoanalyst`, branch `main`, and file path `app.py`.
3. Deploy. The app installs its Python dependencies from `requirements.txt`.

The web app runs the complete available analyst toolkit. Until the FlyWire connections export has been supplied and
the router trained, it reports and uses the rule-based analysis router. The full-graph checkpoint and raw data are
excluded from Git; do not commit uploaded datasets, credentials, or private connection strings. To enable a trained
connectome on a hosted deployment, provide the checkpoint through the hosting platform's secure storage and update
the app's checkpoint path accordingly.
