"""Reproducible univariate time-series diagnostics and held-out forecasts."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def prepare_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    frequency: str,
    aggregation: str,
) -> pd.Series:
    if aggregation not in {"sum", "mean", "median"}:
        raise ValueError("Aggregation must be sum, mean, or median.")
    values = pd.to_numeric(df[value_column], errors="coerce")
    dates = pd.to_datetime(df[date_column], errors="coerce")
    valid = pd.DataFrame({"date": dates, "value": values}).dropna()
    if valid.empty:
        raise ValueError("No valid date/value rows are available.")
    series = valid.set_index("date")["value"].sort_index()
    series = series.resample(frequency).agg(aggregation).dropna()
    if len(series) < 10:
        raise ValueError("Time-series analysis requires at least 10 non-empty periods.")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("The aggregated time series contains non-finite values.")
    return series


def forecast(
    series: pd.Series,
    *,
    periods: int = 6,
    model_name: str = "Exponential smoothing",
    seasonal_periods: int | None = None,
    frequency: str | None = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Use a final holdout to evaluate, then fit all history for future forecasts."""
    if periods < 1 or periods > 120:
        raise ValueError("Forecast horizon must be between 1 and 120 periods.")
    if len(series) < 10:
        raise ValueError("Forecasting requires at least 10 observations.")
    if model_name not in {"Exponential smoothing", "ARIMA (1,1,1)"}:
        raise ValueError(f"Unsupported forecast model: {model_name}")
    if seasonal_periods is not None and model_name != "Exponential smoothing":
        raise ValueError("Seasonality is supported only by exponential smoothing.")
    if seasonal_periods is not None and seasonal_periods > 1 and len(series) < seasonal_periods * 3:
        raise ValueError("Seasonal modeling requires at least three full seasonal cycles.")

    holdout_size = max(2, min(periods, int(round(len(series) * 0.2))))
    train, test = series.iloc[:-holdout_size], series.iloc[-holdout_size:]
    model, _ = _fit(train, model_name, seasonal_periods)
    holdout_prediction = pd.Series(
        np.asarray(model.forecast(len(test))),
        index=test.index,
        name="forecast",
    )
    metrics = {
        "mae": float(mean_absolute_error(test, holdout_prediction)),
        "rmse": float(np.sqrt(mean_squared_error(test, holdout_prediction))),
        "holdout_periods": len(test),
        "model": model_name,
        "validation": "Single chronological holdout; not rolling-origin validation.",
    }
    if len(test) > 1 and np.abs(test.to_numpy()).sum() > 0:
        nonzero = test.to_numpy() != 0
        if nonzero.any():
            metrics["mape_percent"] = float(
                np.mean(np.abs((test.to_numpy()[nonzero] - holdout_prediction.to_numpy()[nonzero])
                               / test.to_numpy()[nonzero])) * 100
            )

    full_model, full_residuals = _fit(series, model_name, seasonal_periods)
    inferred_frequency = frequency or series.index.freq or pd.infer_freq(series.index)
    if inferred_frequency is None:
        inferred_frequency = series.index[-1] - series.index[-2]
    future_index = pd.date_range(
        start=series.index[-1],
        periods=periods + 1,
        freq=inferred_frequency,
    )[1:]
    future_values = np.asarray(full_model.forecast(periods))
    residual_scale = float(np.nanstd(full_residuals, ddof=1)) if len(full_residuals) > 1 else 0.0
    future = pd.DataFrame({
        "date": future_index,
        "forecast": future_values,
        "lower_approx_95": future_values - 1.96 * residual_scale,
        "upper_approx_95": future_values + 1.96 * residual_scale,
        "interval_note": "Approximate constant-residual interval; not a calibrated prediction interval.",
    })

    diagnostics = {}
    if len(series) >= 10 and series.nunique() > 1:
        result = adfuller(
            series.to_numpy(),
            maxlag=0,
            autolag=None,
            regression="c",
            result_object=True,
        )
        diagnostics = {
            "adf_statistic": float(result.statistic),
            "adf_p_value": float(result.pvalue),
            "stationarity_note": (
                "Reject unit-root null at 0.05."
                if result.pvalue < 0.05 else
                "Insufficient evidence to reject unit-root null at 0.05."
            ),
            "adf_method": "Dickey-Fuller test with zero augmentation lags and a constant.",
            "autocorrelation": [
                float(value)
                for value in acf(series.to_numpy(), nlags=min(20, len(series) // 3), fft=True)
            ],
        }
    return future, metrics | diagnostics, pd.DataFrame({
        "actual": test,
        "holdout_forecast": holdout_prediction,
    })


def _fit(series: pd.Series, model_name: str, seasonal_periods: int | None):
    if model_name == "ARIMA (1,1,1)":
        fitted = ARIMA(series, order=(1, 1, 1)).fit()
        residuals = np.asarray(fitted.resid)
        return fitted, residuals
    seasonal = (
        "add" if seasonal_periods and seasonal_periods > 1 else None
    )
    fitted = ExponentialSmoothing(
        series,
        trend="add" if len(series) >= 4 else None,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods if seasonal else None,
        initialization_method="estimated",
    ).fit(optimized=True)
    return fitted, np.asarray(fitted.resid)
