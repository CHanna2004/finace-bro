"""
Fetch current / real-time commodity prices and market data.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yfinance as yf
import pandas as pd

from config import CACHE_DIR, CACHE_TTL_MINUTES, COMMODITIES

logger = logging.getLogger(__name__)


def _live_cache_path(ticker: str) -> Path:
    Path(CACHE_DIR).mkdir(exist_ok=True)
    return Path(CACHE_DIR) / f"live_{ticker.replace('=', '_')}.json"


def fetch_current_price(
    ticker: str,
    force_refresh: bool = False,
) -> dict:
    """
    Fetch the current price info for a ticker.

    Returns a dict with keys:
        ticker, price, previous_close, change, change_pct,
        day_high, day_low, volume, timestamp
    """
    cache_file = _live_cache_path(ticker)

    if not force_refresh and cache_file.exists():
        age_minutes = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 60
        if age_minutes < CACHE_TTL_MINUTES:
            logger.info("Using cached live data for %s (%.1f min old)", ticker, age_minutes)
            with open(cache_file) as f:
                return json.load(f)

    logger.info("Fetching live data for %s...", ticker)
    tkr = yf.Ticker(ticker)
    info = tkr.info

    # yfinance sometimes returns different key names depending on asset type
    price = (
        info.get("regularMarketPrice")
        or info.get("currentPrice")
        or info.get("navPrice")
        or info.get("ask")
        or None
    )

    if price is None:
        # Fallback: fetch the last 2 days of intraday data
        hist = tkr.history(period="2d", interval="1m")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
        else:
            raise ValueError(f"Could not fetch current price for '{ticker}'")

    prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose") or price
    change = price - prev_close
    change_pct = (change / prev_close) * 100 if prev_close else 0.0

    result = {
        "ticker":       ticker,
        "price":        round(float(price), 4),
        "previous_close": round(float(prev_close), 4),
        "change":       round(float(change), 4),
        "change_pct":   round(float(change_pct), 4),
        "day_high":     round(float(info.get("regularMarketDayHigh") or info.get("dayHigh") or price), 4),
        "day_low":      round(float(info.get("regularMarketDayLow")  or info.get("dayLow")  or price), 4),
        "volume":       int(info.get("regularMarketVolume") or info.get("volume") or 0),
        "timestamp":    datetime.utcnow().isoformat() + "Z",
    }

    with open(cache_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def fetch_current_commodity(
    commodity: str,
    force_refresh: bool = False,
) -> dict:
    """
    Fetch current price for a named commodity, adding metadata from config.
    """
    if commodity not in COMMODITIES:
        raise ValueError(
            f"Unknown commodity '{commodity}'. "
            f"Available: {', '.join(COMMODITIES.keys())}"
        )
    meta = COMMODITIES[commodity]
    data = fetch_current_price(meta["ticker"], force_refresh=force_refresh)
    data["commodity"] = commodity
    data["name"]      = meta["name"]
    data["unit"]      = meta["unit"]
    return data


def fetch_multiple_current(
    commodities: list[str],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch current prices for multiple commodities and return as a DataFrame.
    """
    rows = []
    for c in commodities:
        try:
            rows.append(fetch_current_commodity(c, force_refresh=force_refresh))
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", c, exc)
    return pd.DataFrame(rows).set_index("commodity") if rows else pd.DataFrame()
