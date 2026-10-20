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

1. Download both `connections.csv.gz` and `neurons.csv.gz` from FlyWire Codex snapshot **783 (October 2023)** into
   `data/raw/`. The official public files are [connections](https://storage.googleapis.com/flywire-data/codex/data/fafb/783/connections.csv.gz)
   and [neurons](https://storage.googleapis.com/flywire-data/codex/data/fafb/783/neurons.csv.gz). Do not commit the
   source data to this repository.
2. Train and save the full-graph router, supplying the complete neuron inventory so neurons with no listed edges
   are included:
   `python train_brain.py`
   This uses every inventory neuron and retains edges with at least one synapse by default. Use `--connections`,
   `--neurons`, `--min-syn`, `--datasets`, `--epochs`, `--steps`, `--batch-size`, and `--device` to override defaults.
   Training targets are generated from synthetic analyst tasks; they are not biological ground truth. CPU training
   may take many hours with the default training set; a CUDA GPU is recommended.
3. Start the app with `streamlit run app.py`. It loads `checkpoints/brain_router.pt` automatically. If no checkpoint
   exists, it clearly reports that it is using the rule-based baseline instead.

The snapshot's neuron inventory contains 139,255 rows. The connection export contains 3,869,878 rows before
aggregation. Cite the underlying connectome as: Dorkenwald et al. (2024), “Neuronal wiring diagram of an adult
brain,” *Nature* 634, 124–138. https://doi.org/10.1038/s41586-024-07558-y. Codex describes snapshot 783 as
October 2023.

To build tensors separately, run:
`python -c "from src.connectome import build_tensors as b; b('data/raw/connections.csv.gz', min_syn=1, out_path='data/processed/tensors.pt', neurons_path='data/raw/neurons.csv.gz')"`

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

The web app runs the complete available analyst toolkit. Until a validated trained FlyWire checkpoint is supplied,
it reports and uses the rule-based analysis router. The app can load a checkpoint from local storage or download a
verified checkpoint from an HTTPS URL. For Streamlit Community Cloud, put `BRAIN_CHECKPOINT_URL` and
`BRAIN_CHECKPOINT_SHA256` in the app's **Manage app → Settings → Secrets**. The app verifies the SHA-256 digest
before loading the checkpoint. Keep raw data and checkpoint files out of Git; do not commit uploaded datasets,
credentials, or private connection strings.
