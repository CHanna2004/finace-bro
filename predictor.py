"""
ML-based commodity futures price predictor.

Uses a Random Forest trained on technical features from historical data
combined with the current price to generate a point forecast and
directional confidence for each forecast horizon.
"""

import logging
import math
import warnings
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error

from scraper import compute_features

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


FORECAST_HORIZONS = [1, 5, 10, 20, 30]   # trading days ahead


def _build_targets(feat: pd.DataFrame, horizon: int) -> pd.Series:
    """Future log-return over `horizon` days."""
    return (feat["close"].shift(-horizon) / feat["close"]).apply(
        lambda x: math.log(x) if not math.isnan(float(x)) else float("nan")
    ).dropna()


def _feature_cols(feat: pd.DataFrame) -> List[str]:
    return [c for c in feat.columns if c != "close"]


def train_model(
    feat: pd.DataFrame,
    horizon: int,
    model_type: str = "rf",
) -> tuple:
    """
    Train a model to predict the log-return `horizon` days ahead.

    Returns (pipeline, cv_score, feature_importance_series).
    """
    y = _build_targets(feat, horizon)
    X = feat.loc[y.index, _feature_cols(feat)]

    if len(X) < 60:
        raise ValueError("Not enough data to train (need ≥ 60 rows).")

    if model_type == "gb":
        base = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
    else:
        base = RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )

    pipe = Pipeline([("scaler", StandardScaler()), ("model", base)])

    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(pipe, X, y, cv=tscv, scoring="r2", n_jobs=-1)
    cv_r2 = float(scores.mean())

    pipe.fit(X, y)

    # Feature importances
    importances = pipe.named_steps["model"].feature_importances_
    imp_series = pd.Series(importances, index=_feature_cols(feat), name=f"h{horizon}").sort_values(ascending=False)

    logger.info("Horizon %dd — CV R²=%.3f", horizon, cv_r2)
    return pipe, cv_r2, imp_series


def predict_futures(
    historical_df: pd.DataFrame,
    current_price: float,
    horizons: list[int] = FORECAST_HORIZONS,
    model_type: str = "rf",
    current_features: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Train models for each forecast horizon and predict future prices.

    Parameters
    ----------
    historical_df   : DataFrame with OHLCV columns (from scraper.py)
    current_price   : current market price (from current_data.py)
    horizons        : list of trading-day horizons to forecast
    model_type      : "rf" (Random Forest) or "gb" (Gradient Boosting)
    current_features: optional dict overriding the latest feature row

    Returns
    -------
    DataFrame indexed by horizon (days), with columns:
        predicted_price, log_return_pred, direction, confidence, cv_r2
    """
    feat = compute_features(historical_df)
    fcols = _feature_cols(feat)
    latest = feat[fcols].iloc[-1:]  # most recent feature row

    if current_features:
        for k, v in current_features.items():
            if k in latest.columns:
                latest[k] = v

    results = []
    for h in horizons:
        try:
            pipe, cv_r2, _ = train_model(feat, h, model_type=model_type)
            log_ret_pred = float(pipe.predict(latest)[0])
            pred_price   = current_price * math.exp(log_ret_pred)
            direction    = "UP" if log_ret_pred > 0 else "DOWN"
            # Translate |log_ret| to a rough confidence (bounded 50–99%)
            confidence   = min(99.0, 50.0 + abs(log_ret_pred) * 1000)

            results.append({
                "horizon_days":    h,
                "predicted_price": round(pred_price, 4),
                "log_return_pred": round(log_ret_pred, 6),
                "change_pct":      round(log_ret_pred * 100, 3),
                "direction":       direction,
                "confidence":      round(confidence, 1),
                "cv_r2":           round(cv_r2, 4),
                "model":           model_type.upper(),
            })
        except Exception as exc:
            logger.warning("Skipping horizon %dd: %s", h, exc)

    return pd.DataFrame(results).set_index("horizon_days") if results else pd.DataFrame()


def ensemble_predict(
    historical_df: pd.DataFrame,
    current_price: float,
    horizons: list[int] = FORECAST_HORIZONS,
) -> pd.DataFrame:
    """
    Run both RF and GB models and return the averaged prediction.
    """
    rf_preds = predict_futures(historical_df, current_price, horizons, model_type="rf")
    gb_preds = predict_futures(historical_df, current_price, horizons, model_type="gb")

    if rf_preds.empty:
        return gb_preds
    if gb_preds.empty:
        return rf_preds

    ensemble = rf_preds.copy()
    ensemble["predicted_price"] = (
        rf_preds["predicted_price"] + gb_preds["predicted_price"]
    ) / 2
    ensemble["log_return_pred"] = (
        rf_preds["log_return_pred"] + gb_preds["log_return_pred"]
    ) / 2
    ensemble["change_pct"] = (rf_preds["change_pct"] + gb_preds["change_pct"]) / 2
    ensemble["cv_r2"]      = (rf_preds["cv_r2"] + gb_preds["cv_r2"]) / 2
    ensemble["model"]      = "ENSEMBLE"
    ensemble["direction"]  = ensemble["log_return_pred"].apply(
        lambda x: "UP" if x > 0 else "DOWN"
    )
    ensemble["confidence"] = ensemble["log_return_pred"].apply(
        lambda x: min(99.0, 50.0 + abs(x) * 1000)
    ).round(1)

    return ensemble
