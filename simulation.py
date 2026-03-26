"""
Monte Carlo simulation engine for commodity futures price prediction.
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from config import DEFAULT_SIMULATIONS, DEFAULT_FORECAST_DAYS, CONFIDENCE_INTERVALS

logger = logging.getLogger(__name__)


def geometric_brownian_motion(
    current_price: float,
    mu: float,         # annualised drift (log-return mean * 252)
    sigma: float,      # annualised volatility (log-return std  * sqrt(252))
    days: int,
    n_simulations: int,
    dt: float = 1 / 252,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate price paths using Geometric Brownian Motion.

    Returns an array of shape (n_simulations, days + 1).
    The first column is always `current_price`.
    """
    rng = np.random.default_rng(seed)
    paths = np.empty((n_simulations, days + 1))
    paths[:, 0] = current_price

    # Vectorised simulation
    z = rng.standard_normal((n_simulations, days))
    factor = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z)
    for t in range(1, days + 1):
        paths[:, t] = paths[:, t - 1] * factor[:, t - 1]

    return paths


def jump_diffusion(
    current_price: float,
    mu: float,
    sigma: float,
    days: int,
    n_simulations: int,
    jump_lambda: float = 5.0,   # expected jumps per year
    jump_mu: float = 0.0,       # mean log-jump size
    jump_sigma: float = 0.03,   # std of log-jump size
    dt: float = 1 / 252,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Merton Jump-Diffusion model.
    Adds Poisson-distributed jumps on top of GBM to model sudden price shocks.
    """
    rng = np.random.default_rng(seed)
    paths = np.empty((n_simulations, days + 1))
    paths[:, 0] = current_price

    # Drift adjustment for jump component (compensator)
    k = math.exp(jump_mu + 0.5 * jump_sigma ** 2) - 1
    mu_adj = mu - jump_lambda * k

    z = rng.standard_normal((n_simulations, days))
    n_jumps = rng.poisson(jump_lambda * dt, (n_simulations, days))
    jump_sizes = rng.normal(jump_mu, jump_sigma, (n_simulations, days))

    for t in range(1, days + 1):
        diff = (mu_adj - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z[:, t - 1]
        jump = n_jumps[:, t - 1] * jump_sizes[:, t - 1]
        paths[:, t] = paths[:, t - 1] * np.exp(diff + jump)

    return paths


def compute_simulation_stats(
    paths: np.ndarray,
    percentiles: list[float] = CONFIDENCE_INTERVALS,
) -> pd.DataFrame:
    """
    Summarise simulation paths into percentile bands.

    Returns a DataFrame with one row per time step (day 0 … day N),
    columns: mean, std, and one column per percentile (e.g. p5, p25, ...).
    """
    stats = pd.DataFrame(index=range(paths.shape[1]))
    stats.index.name = "day"
    stats["mean"] = paths.mean(axis=0)
    stats["std"]  = paths.std(axis=0)
    for p in percentiles:
        label = f"p{int(p * 100)}"
        stats[label] = np.percentile(paths, p * 100, axis=0)
    return stats


def run_simulation(
    historical_df: pd.DataFrame,
    current_price: float,
    forecast_days: int = DEFAULT_FORECAST_DAYS,
    n_simulations: int = DEFAULT_SIMULATIONS,
    model: str = "gbm",          # "gbm" or "jump_diffusion"
    seed: Optional[int] = 42,
) -> dict:
    """
    Full simulation pipeline.

    Parameters
    ----------
    historical_df : DataFrame with a 'Close' column (from scraper.py)
    current_price : latest observed price (from current_data.py)
    forecast_days : how many trading days ahead to simulate
    n_simulations : number of Monte Carlo paths
    model         : "gbm" (default) or "jump_diffusion"
    seed          : random seed for reproducibility

    Returns
    -------
    dict with keys:
        paths      - raw simulation array (n_simulations x forecast_days+1)
        stats      - summary DataFrame
        params     - calibrated model params
        var_95     - 95% VaR (downside)
        cvar_95    - Conditional VaR (expected shortfall)
    """
    # ── Calibrate from historical log-returns ──────────────────────────────
    close = historical_df["Close"].dropna()
    log_returns = np.log(close / close.shift(1)).dropna()

    daily_mu    = float(log_returns.mean())
    daily_sigma = float(log_returns.std())
    annual_mu    = daily_mu    * 252
    annual_sigma = daily_sigma * math.sqrt(252)

    logger.info(
        "Calibrated params — daily μ=%.5f σ=%.5f | annual μ=%.3f σ=%.3f",
        daily_mu, daily_sigma, annual_mu, annual_sigma,
    )

    # ── Run chosen model ───────────────────────────────────────────────────
    if model == "jump_diffusion":
        # Estimate jump parameters from large-move days (|return| > 2σ)
        threshold = 2 * daily_sigma
        large_moves = log_returns[log_returns.abs() > threshold]
        jump_lambda = len(large_moves) / len(log_returns) * 252
        jump_sigma  = float(large_moves.std()) if len(large_moves) > 1 else daily_sigma
        jump_mu     = float(large_moves.mean()) if len(large_moves) > 1 else 0.0

        paths = jump_diffusion(
            current_price=current_price,
            mu=annual_mu,
            sigma=annual_sigma,
            days=forecast_days,
            n_simulations=n_simulations,
            jump_lambda=jump_lambda,
            jump_mu=jump_mu,
            jump_sigma=jump_sigma,
            seed=seed,
        )
        params = dict(
            model="jump_diffusion",
            annual_mu=annual_mu, annual_sigma=annual_sigma,
            jump_lambda=jump_lambda, jump_mu=jump_mu, jump_sigma=jump_sigma,
        )
    else:
        paths = geometric_brownian_motion(
            current_price=current_price,
            mu=annual_mu,
            sigma=annual_sigma,
            days=forecast_days,
            n_simulations=n_simulations,
            seed=seed,
        )
        params = dict(model="gbm", annual_mu=annual_mu, annual_sigma=annual_sigma)

    # ── Risk metrics on terminal prices ───────────────────────────────────
    terminal = paths[:, -1]
    pnl = (terminal - current_price) / current_price  # relative P&L

    var_95  = float(np.percentile(pnl, 5))    # 5th percentile loss
    cvar_95 = float(pnl[pnl <= var_95].mean()) if (pnl <= var_95).any() else var_95

    stats = compute_simulation_stats(paths)

    return {
        "paths":   paths,
        "stats":   stats,
        "params":  params,
        "var_95":  var_95,
        "cvar_95": cvar_95,
        "current_price": current_price,
        "forecast_days": forecast_days,
        "n_simulations": n_simulations,
    }
