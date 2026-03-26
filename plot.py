"""
Visualization for commodity futures simulation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


def _style():
    plt.rcParams.update({
        "figure.facecolor": "#0f0f1a",
        "axes.facecolor":   "#0f0f1a",
        "axes.edgecolor":   "#333355",
        "axes.labelcolor":  "#ccccdd",
        "text.color":       "#ccccdd",
        "xtick.color":      "#888899",
        "ytick.color":      "#888899",
        "grid.color":       "#222233",
        "grid.linewidth":   0.5,
        "font.family":      "monospace",
    })


def plot_simulation(
    sim_result: dict,
    pred_df: pd.DataFrame,
    commodity: str,
    live: dict,
    max_paths: int = 200,
    save_path: str = None,
) -> None:
    """
    Plot a 4-panel dashboard:
      1. Monte Carlo price paths (fan chart)
      2. Terminal price distribution histogram
      3. ML predicted prices by horizon
      4. Historical close + current price
    """
    _style()

    paths  = sim_result["paths"]
    stats  = sim_result["stats"]
    params = sim_result["params"]
    cur    = sim_result["current_price"]
    f_days = sim_result["forecast_days"]
    name   = live.get("name", commodity)
    unit   = live.get("unit", "USD")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"{name}  |  Current: {cur:,.2f} {unit}  |  "
        f"Model: {params['model'].upper()}  |  "
        f"Horizon: {f_days}d",
        fontsize=13, color="#eeeeff", y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])   # MC paths
    ax2 = fig.add_subplot(gs[0, 1])   # terminal distribution
    ax3 = fig.add_subplot(gs[1, 0])   # ML predictions
    ax4 = fig.add_subplot(gs[1, 1])   # percentile bands over time

    days = np.arange(paths.shape[1])

    # ── Panel 1: Sample MC paths ───────────────────────────────────────────
    sample = paths[np.random.choice(len(paths), min(max_paths, len(paths)), replace=False)]
    for path in sample:
        ax1.plot(days, path, color="#3355aa", alpha=0.05, linewidth=0.6)
    ax1.plot(days, stats["mean"], color="#66aaff", linewidth=1.5, label="Mean")
    if "p5" in stats.columns and "p95" in stats.columns:
        ax1.fill_between(days, stats["p5"], stats["p95"], color="#3355aa", alpha=0.25, label="5–95%")
        ax1.fill_between(days, stats["p25"], stats["p75"], color="#4466cc", alpha=0.30, label="25–75%")
    ax1.axhline(cur, color="#ffaa33", linewidth=1, linestyle="--", label="Current")
    ax1.set_title("Monte Carlo Paths", color="#eeeeff")
    ax1.set_xlabel("Trading Days")
    ax1.set_ylabel(f"Price ({unit})")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True)

    # ── Panel 2: Terminal price histogram ─────────────────────────────────
    terminal = paths[:, -1]
    ax2.hist(terminal, bins=80, color="#3355aa", alpha=0.8, edgecolor="none")
    ax2.axvline(cur,              color="#ffaa33", linewidth=1.5, linestyle="--", label="Current")
    ax2.axvline(stats["mean"].iloc[-1], color="#66aaff", linewidth=1.5, label="Mean")
    if "p5" in stats.columns:
        ax2.axvline(stats["p5"].iloc[-1],  color="#ff4444", linewidth=1, linestyle=":", label="5th pct")
        ax2.axvline(stats["p95"].iloc[-1], color="#44ff88", linewidth=1, linestyle=":", label="95th pct")
    ax2.set_title(f"Terminal Price Distribution (Day {f_days})", color="#eeeeff")
    ax2.set_xlabel(f"Price ({unit})")
    ax2.set_ylabel("Frequency")
    ax2.legend(fontsize=7)
    ax2.grid(True)

    # ── Panel 3: ML predictions ────────────────────────────────────────────
    if pred_df is not None and not pred_df.empty:
        horizons = pred_df.index.tolist()
        prices   = pred_df["predicted_price"].tolist()
        colors   = ["#44ff88" if p >= cur else "#ff4444" for p in prices]

        ax3.bar(range(len(horizons)), prices, color=colors, alpha=0.7, width=0.6)
        ax3.axhline(cur, color="#ffaa33", linewidth=1.2, linestyle="--", label="Current")
        ax3.set_xticks(range(len(horizons)))
        ax3.set_xticklabels([f"{h}d" for h in horizons])
        ax3.set_title(f"ML Forecast  ({pred_df['model'].iloc[0]})", color="#eeeeff")
        ax3.set_ylabel(f"Price ({unit})")
        ax3.legend(fontsize=7)
        ax3.grid(True, axis="y")

        for i, (p, h) in enumerate(zip(prices, horizons)):
            chg = (p / cur - 1) * 100
            ax3.text(i, p + (max(prices) - min(prices)) * 0.01,
                     f"{chg:+.1f}%", ha="center", va="bottom", fontsize=7,
                     color="#44ff88" if chg >= 0 else "#ff4444")
    else:
        ax3.text(0.5, 0.5, "No ML predictions\n(run without --no-ml)",
                 ha="center", va="center", transform=ax3.transAxes, color="#888899")
        ax3.set_title("ML Forecast", color="#eeeeff")

    # ── Panel 4: Percentile band over time ────────────────────────────────
    ax4.fill_between(days, stats.get("p5",  stats["mean"]),
                           stats.get("p95", stats["mean"]),
                     color="#3355aa", alpha=0.20, label="5–95%")
    ax4.fill_between(days, stats.get("p25", stats["mean"]),
                           stats.get("p75", stats["mean"]),
                     color="#4466cc", alpha=0.35, label="25–75%")
    ax4.plot(days, stats["mean"],            color="#66aaff", linewidth=1.5, label="Mean")
    ax4.plot(days, stats.get("p50", stats["mean"]), color="#aaccff", linewidth=1,
             linestyle="--", label="Median")
    ax4.axhline(cur, color="#ffaa33", linewidth=1, linestyle="--", label="Current")

    # Overlay ML predictions as scatter dots
    if pred_df is not None and not pred_df.empty:
        for h, row in pred_df.iterrows():
            if h <= f_days:
                c = "#44ff88" if row["predicted_price"] >= cur else "#ff4444"
                ax4.scatter(h, row["predicted_price"], color=c, zorder=5, s=40)

    ax4.set_title("Price Fan Chart + ML Points", color="#eeeeff")
    ax4.set_xlabel("Trading Days")
    ax4.set_ylabel(f"Price ({unit})")
    ax4.legend(fontsize=7, loc="upper left")
    ax4.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"\n  Chart saved: {save_path}")
    else:
        plt.tight_layout()
        plt.show()
