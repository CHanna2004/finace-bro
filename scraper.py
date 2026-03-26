"""
Historical commodity data scraper using yfinance.
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import pandas as pd

from config import CACHE_DIR, DEFAULT_HISTORY_YEARS, COMMODITIES

logger = logging.getLogger(__name__)


def _cache_path(ticker: str, years: int) -> Path:
    key = hashlib.md5(f"{ticker}_{years}".encode()).hexdigest()[:8]
    Path(CACHE_DIR).mkdir(exist_ok=True)
    return Path(CACHE_DIR) / f"{ticker.replace('=', '_')}_{years}y_{key}.parquet"


def fetch_historical(
    ticker: str,
    years: int = DEFAULT_HISTORY_YEARS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV historical data for a ticker.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    indexed by Date.
    """
    cache_file = _cache_path(ticker, years)

    if not force_refresh and cache_file.exists():
        age_minutes = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 60
        if age_minutes < 60 * 24:  # cache historical data for 24 hours
            logger.info("Loading historical data from cache: %s", cache_file)
            return pd.read_parquet(cache_file)

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)

    logger.info("Fetching %d years of historical data for %s...", years, ticker)
    raw = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No historical data returned for ticker '{ticker}'")

    # Flatten MultiIndex columns if present (yfinance >=0.2 quirk)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.dropna(subset=["Close"], inplace=True)

    df.to_parquet(cache_file)
    logger.info("Saved %d rows to cache.", len(df))
    return df


def get_commodity_history(
    commodity: str,
    years: int = DEFAULT_HISTORY_YEARS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch historical data for a named commodity (from config.COMMODITIES).
    """
    if commodity not in COMMODITIES:
        raise ValueError(
            f"Unknown commodity '{commodity}'. "
            f"Available: {', '.join(COMMODITIES.keys())}"
        )
    ticker = COMMODITIES[commodity]["ticker"]
    return fetch_historical(ticker, years=years, force_refresh=force_refresh)


def compute_returns(df: pd.DataFrame) -> pd.Series:
    """Compute daily log returns from Close prices."""
    return pd.Series(
        (df["Close"] / df["Close"].shift(1)).apply(lambda x: x if pd.isna(x) else __import__('math').log(x)),
        name="log_return",
    ).dropna()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features used for ML prediction:
      - Log returns
      - Rolling volatility (5, 10, 20-day)
      - Rolling mean return (5, 10, 20-day)
      - Price momentum (rate of change)
      - High-Low range ratio
    """
    close = df["Close"]
    log_ret = (close / close.shift(1)).apply(
        lambda x: __import__('math').log(x) if not __import__('math').isnan(float(x)) else float('nan')
    )

    feat = pd.DataFrame(index=df.index)
    feat["log_return"] = log_ret
    feat["hl_range"] = (df["High"] - df["Low"]) / close
    feat["vol_5"]  = log_ret.rolling(5).std()
    feat["vol_10"] = log_ret.rolling(10).std()
    feat["vol_20"] = log_ret.rolling(20).std()
    feat["ma_5"]   = log_ret.rolling(5).mean()
    feat["ma_10"]  = log_ret.rolling(10).mean()
    feat["ma_20"]  = log_ret.rolling(20).mean()
    feat["roc_5"]  = close.pct_change(5)
    feat["roc_10"] = close.pct_change(10)
    feat["roc_20"] = close.pct_change(20)
    feat["close"]  = close

    return feat.dropna()
