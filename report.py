"""
Terminal report rendering for commodity futures simulation results.
"""

from datetime import datetime

import pandas as pd
import numpy as np


def _fmt(val, decimals=2):
    return f"{val:,.{decimals}f}"


def print_header(commodity: str, meta: dict, live: dict) -> None:
    name  = meta["name"]
    unit  = meta["unit"]
    price = live["price"]
    chg   = live["change"]
    chg_p = live["change_pct"]
    sign  = "+" if chg >= 0 else ""
    ts    = live["timestamp"]

    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {name}  ({commodity.upper()})")
    print(f"  Current Price : {_fmt(price)} {unit}")
    print(f"  Day Change    : {sign}{_fmt(chg)} ({sign}{_fmt(chg_p)}%)")
    print(f"  Day Range     : {_fmt(live['day_low'])} – {_fmt(live['day_high'])}")
    print(f"  Timestamp     : {ts}")
    print(bar)


def print_simulation_summary(sim_result: dict) -> None:
    stats   = sim_result["stats"]
    params  = sim_result["params"]
    n_sim   = sim_result["n_simulations"]
    f_days  = sim_result["forecast_days"]
    cur     = sim_result["current_price"]
    var     = sim_result["var_95"]
    cvar    = sim_result["cvar_95"]

    terminal = stats.iloc[-1]
    mean_p   = terminal["mean"]
    p5, p25, p50, p75, p95 = (
        terminal.get("p5", np.nan),
        terminal.get("p25", np.nan),
        terminal.get("p50", np.nan),
        terminal.get("p75", np.nan),
        terminal.get("p95", np.nan),
    )

    print(f"\n--- Monte Carlo Simulation ({params['model'].upper()}) ---")
    print(f"  Simulations   : {n_sim:,}")
    print(f"  Forecast      : {f_days} trading days")
    print(f"  Annual μ      : {params['annual_mu']*100:.2f}%")
    print(f"  Annual σ      : {params['annual_sigma']*100:.2f}%")
    if params["model"] == "jump_diffusion":
        print(f"  Jump λ/yr     : {params['jump_lambda']:.1f}")
    print()
    print(f"  Forecast Price Distribution at Day {f_days}:")
    print(f"    Mean          : {_fmt(mean_p)}")
    print(f"    5th  pct (bear): {_fmt(p5)}")
    print(f"    25th pct       : {_fmt(p25)}")
    print(f"    50th pct       : {_fmt(p50)}")
    print(f"    75th pct       : {_fmt(p75)}")
    print(f"    95th pct (bull): {_fmt(p95)}")
    print()
    print(f"  Risk Metrics:")
    print(f"    95% VaR  (downside) : {var*100:.2f}%")
    print(f"    95% CVaR (exp loss) : {cvar*100:.2f}%")


def print_ml_predictions(pred_df: pd.DataFrame, current_price: float) -> None:
    if pred_df.empty:
        print("\n  [No ML predictions available]")
        return

    print(f"\n--- ML Predictions (model: {pred_df['model'].iloc[0]}) ---")
    print(f"  {'Horizon':>10}  {'Pred Price':>12}  {'Change %':>9}  {'Direction':>9}  {'Confidence':>10}  {'CV R²':>7}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*7}")
    for horizon, row in pred_df.iterrows():
        arrow = "↑" if row["direction"] == "UP" else "↓"
        print(
            f"  {f'{horizon}d':>10}  "
            f"{_fmt(row['predicted_price']):>12}  "
            f"{row['change_pct']:>+9.2f}%  "
            f"{arrow + ' ' + row['direction']:>9}  "
            f"{row['confidence']:>9.1f}%  "
            f"{row['cv_r2']:>7.4f}"
        )


def print_combined_outlook(sim_result: dict, pred_df: pd.DataFrame) -> None:
    """Print a synthesised outlook combining MC and ML results."""
    print(f"\n--- Combined Outlook ---")
    cur  = sim_result["current_price"]
    stats = sim_result["stats"]
    days  = sim_result["forecast_days"]

    if not pred_df.empty and days in pred_df.index:
        ml_row  = pred_df.loc[days]
        ml_pred = ml_row["predicted_price"]
        mc_row  = stats.iloc[-1]
        mc_mean = mc_row["mean"]
        mc_p5   = mc_row.get("p5", np.nan)
        mc_p95  = mc_row.get("p95", np.nan)

        blended = (ml_pred + mc_mean) / 2
        print(f"  Horizon       : {days} trading days")
        print(f"  MC Mean       : {_fmt(mc_mean)}  ({(mc_mean/cur-1)*100:+.2f}%)")
        print(f"  ML Forecast   : {_fmt(ml_pred)}  ({(ml_pred/cur-1)*100:+.2f}%)")
        print(f"  Blended Est.  : {_fmt(blended)}  ({(blended/cur-1)*100:+.2f}%)")
        print(f"  MC 90% Range  : {_fmt(mc_p5)} – {_fmt(mc_p95)}")
    else:
        print("  (Run with matching horizon to see blended outlook)")

    print()


def save_csv(sim_result: dict, pred_df: pd.DataFrame, commodity: str, output_dir: str = "output") -> None:
    """Save simulation stats and ML predictions to CSV files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_path = f"{output_dir}/{commodity}_sim_stats_{ts}.csv"
    pred_path  = f"{output_dir}/{commodity}_ml_pred_{ts}.csv"

    sim_result["stats"].to_csv(stats_path)
    if not pred_df.empty:
        pred_df.to_csv(pred_path)

    print(f"\n  Saved: {stats_path}")
    print(f"  Saved: {pred_path}")
