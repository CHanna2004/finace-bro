"""
Synthetic market data generator for offline / demo mode.

Produces realistic OHLCV history and a "current" price using
a seeded GBM so results are reproducible.
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import COMMODITIES

# Approximate real-world base prices (USD) as of early 2025
BASE_PRICES = {
    "gold":        2300.0,
    "silver":        27.0,
    "crude_oil":     80.0,
    "brent_oil":     84.0,
    "natural_gas":    2.0,
    "corn":         450.0,
    "wheat":        560.0,
    "soybeans":    1050.0,
    "copper":         4.2,
    "platinum":     960.0,
    "palladium":    960.0,
    "cotton":        80.0,
    "sugar":         20.0,
    "coffee":       200.0,
    "cocoa":       8500.0,
}

# Approximate annual volatilities
ANNUAL_VOL = {
    "gold":        0.15,
    "silver":      0.28,
    "crude_oil":   0.35,
    "brent_oil":   0.33,
    "natural_gas": 0.55,
    "corn":        0.25,
    "wheat":       0.27,
    "soybeans":    0.22,
    "copper":      0.28,
    "platinum":    0.22,
    "palladium":   0.40,
    "cotton":      0.22,
    "sugar":       0.30,
    "coffee":      0.35,
    "cocoa":       0.40,
}


def generate_history(commodity: str, years: int = 5, seed: int = 0) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame mimicking `years` of daily trading data.
    """
    rng = np.random.default_rng(seed + hash(commodity) % 1000)

    annual_mu  = 0.05   # 5% drift
    annual_sig = ANNUAL_VOL.get(commodity, 0.25)
    dt         = 1 / 252
    n_days     = years * 252

    base = BASE_PRICES.get(commodity, 100.0)
    log_rets = rng.normal(
        (annual_mu - 0.5 * annual_sig ** 2) * dt,
        annual_sig * math.sqrt(dt),
        n_days,
    )
    close_prices = base * np.exp(np.cumsum(log_rets))

    # Synthesise OHLV from close
    daily_vol = annual_sig * math.sqrt(dt)
    opens   = close_prices * np.exp(rng.normal(0, daily_vol * 0.3, n_days))
    highs   = np.maximum(close_prices, opens) * (1 + rng.uniform(0, daily_vol, n_days))
    lows    = np.minimum(close_prices, opens) * (1 - rng.uniform(0, daily_vol, n_days))
    volumes = rng.integers(50_000, 500_000, n_days).astype(float)

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=n_days * 1.4)  # add weekend buffer
    dates      = pd.bdate_range(end=end_date, periods=n_days)

    df = pd.DataFrame({
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  close_prices,
        "Volume": volumes,
    }, index=dates)
    df.index.name = "Date"
    return df


def generate_current(commodity: str, history: pd.DataFrame) -> dict:
    """
    Generate a synthetic "live" price quote based on the last historical close.
    """
    last_close = float(history["Close"].iloc[-1])
    prev_close = float(history["Close"].iloc[-2])
    change     = last_close - prev_close
    change_pct = (change / prev_close) * 100

    meta = COMMODITIES.get(commodity, {})
    return {
        "commodity":      commodity,
        "ticker":         meta.get("ticker", "???"),
        "name":           meta.get("name",   commodity),
        "unit":           meta.get("unit",   "USD"),
        "price":          round(last_close, 4),
        "previous_close": round(prev_close, 4),
        "change":         round(change, 4),
        "change_pct":     round(change_pct, 4),
        "day_high":       round(float(history["High"].iloc[-1]), 4),
        "day_low":        round(float(history["Low"].iloc[-1]),  4),
        "volume":         int(history["Volume"].iloc[-1]),
        "timestamp":      datetime.utcnow().isoformat() + "Z (DEMO)",
    }
